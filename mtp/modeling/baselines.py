"""The eight baselines.

Each baseline is a function that takes an `Experiment`, trains its models and records
the test results. `BASELINES` maps each baseline number to its function.

0. Time-lagged linear regression from EEG to velocity.
1. EEG model trained on MI only.
2. EEG model pretrained on ME, then fine-tuned on MI.
3. EEG and IMU model pretrained on ME, then fine-tuned on MI with the IMU input set to
   zero. The EEG and IMU features are fused by concatenation.
4. A teacher (EEG and IMU) and a student (EEG only) trained together on ME, after which
   the student is fine-tuned on MI.
5. The full model: like 4, but with lag-aware fusion, a learned per-subject scale and
   shift of the EEG features, and alignment to ME prototypes during MI fine-tuning.
6. Like 3, with bidirectional cross-attention between the EEG and IMU features.
7. Like 3, with self-attention over the fused EEG and IMU features.

The final score of every baseline is on the MI test set. MI trials have no IMU, so no
model sees a real IMU signal when it is tested on MI.
"""

from dataclasses import dataclass
from functools import partial
import logging

from mtp.config import TARGET_LEN, TARGET_ORDER
from mtp.modeling.experiment import Experiment
from mtp.modeling.linear import run_baseline0_linear
from mtp.modeling.models import (
    DistillationSimpleModel,
    EEGOnlyModel,
    FullPrivilegedModel,
    FusionModel,
    freeze_all_except,
)
from mtp.modeling.objectives import (
    PrototypeAlignedStudent,
    Supervised,
    TeacherStudent,
    distill_student,
    eeg_only,
    fusion_real_imu,
    fusion_zero_imu,
)
from mtp.modeling.trainers import compute_me_prototypes

log = logging.getLogger(__name__)


def baseline_0(exp: Experiment):
    """Run Baseline 0: time-lagged ridge regression, fitted separately on ME and on MI."""
    name = "Baseline 0: Time-lagged MLR"
    for data, task, notes in (
        (exp.mi, "MI_test", "MI train+val -> MI test; time-lagged ridge-stabilized MLR"),
        (exp.me, "ME_test", "ME train+val -> ME test; time-lagged ridge-stabilized MLR"),
    ):
        metrics = run_baseline0_linear(data, exp.scaler, exp.dim_names, exp.cfg)
        exp.record(0, name, metrics, notes, task, "linear")


def baseline_1(exp: Experiment):
    """Run Baseline 1: train an EEG model on MI only."""
    name = "Baseline 1: MI-only EEG"
    objective = Supervised(eeg_only, exp.regression_loss())
    model = EEGOnlyModel(
        **exp.model_dims,
        target_len=TARGET_LEN,
        latent_dim=exp.cfg.latent_dim,
        dropout=exp.cfg.dropout,
    ).to(exp.device)

    exp.fit(name, model, "mi", objective, exp.finetune)

    exp.record(
        1,
        name,
        exp.evaluate(model, "mi_test", objective).flat(),
        "Train only on MI EEG; evaluate on MI test",
        "MI_test",
        "EEG_only",
    )
    exp.record(
        1,
        name,
        exp.evaluate(model, "me_eeg_test", objective).flat(),
        "Same MI-only trained EEG model evaluated on ME test",
        "ME_test_final_model",
        "EEG_only",
    )


def baseline_2(exp: Experiment):
    """Run Baseline 2: pretrain an EEG model on ME, then fine-tune it on MI."""
    name = "Baseline 2: ME EEG pretrain + MI fine-tune"
    objective = Supervised(eeg_only, exp.regression_loss())
    model = EEGOnlyModel(
        **exp.model_dims,
        target_len=TARGET_LEN,
        latent_dim=exp.cfg.latent_dim,
        dropout=exp.cfg.dropout,
    ).to(exp.device)

    exp.fit("B2 Stage 1: ME EEG-only pretraining", model, "me_eeg", objective, exp.pretrain)
    exp.record(
        2,
        name,
        exp.evaluate(model, "me_eeg_test", objective).flat(),
        "After ME EEG-only pretraining, before MI fine-tuning",
        "ME_test_after_pretrain",
        "EEG_only",
    )

    exp.fit("B2 Stage 2: MI EEG-only fine-tuning", model, "mi", objective, exp.finetune)
    exp.record(
        2,
        name,
        exp.evaluate(model, "mi_test", objective).flat(),
        "ME EEG-only pretraining, then MI EEG-only fine-tuning; final MI test",
        "MI_test",
        "EEG_only",
    )
    exp.record(
        2,
        name,
        exp.evaluate(model, "me_eeg_test", objective).flat(),
        "Final MI-fine-tuned model evaluated on ME test",
        "ME_test_after_MI_finetune",
        "EEG_only",
    )


@dataclass(frozen=True)
class FusionBaseline:
    """Settings for Baselines 3, 6 and 7.

    These baselines share one training procedure, `fusion_baseline`. They differ only in
    the fusion block and in the names and notes written to the results.

    Attributes
    ----------
    id : int
        Baseline number.
    name : str
        Baseline name in the results.
    fusion : str
        Fusion block for `FusionModel`: "concat", "cross_mha" or "self_mha".
    pretrain_stage : str
        Name of the ME pretraining stage in the "stages" sheet.
    notes_after_pretrain, notes_mi_test, notes_me_after_finetune : str
        Notes for the three evaluations in the "summary" sheet.
    branch_real_imu, branch_zero_imu : str
        Branch labels for the evaluations with the real IMU and with a zero IMU.
    """

    id: int
    name: str
    fusion: str
    pretrain_stage: str
    notes_after_pretrain: str
    notes_mi_test: str
    notes_me_after_finetune: str
    branch_real_imu: str
    branch_zero_imu: str


FUSION_BASELINES = [
    FusionBaseline(
        id=3,
        name="Baseline 3: ME EEG+IMU pretrain, no distill/proto",
        fusion="concat",
        pretrain_stage="B3 Stage 1: ME EEG+IMU pretraining",
        notes_after_pretrain="ME EEG+IMU pretrained model evaluated on ME test using real IMU",
        notes_mi_test="ME EEG+IMU pretraining; MI uses zero IMU; no distillation/prototype alignment",
        notes_me_after_finetune="Final MI-fine-tuned model evaluated on ME test with zero IMU input",
        branch_real_imu="EEG_IMU_fusion",
        branch_zero_imu="EEG_zeroIMU_fusion",
    ),
    FusionBaseline(
        id=6,
        name="Baseline 6: Bidirectional cross-modal MHA fusion",
        fusion="cross_mha",
        pretrain_stage="B6 Stage 1: ME EEG+IMU bidirectional cross-modal MHA pretraining",
        notes_after_pretrain="ME EEG+IMU pretrained model evaluated on ME test using real IMU; bidirectional cross-modal MHA fusion",
        notes_mi_test="ME EEG+IMU pretraining with bidirectional cross-modal MHA; MI uses zero IMU; no distillation/prototype alignment",
        notes_me_after_finetune="Final MI-fine-tuned model evaluated on ME test with zero IMU input; bidirectional cross-modal MHA fusion",
        branch_real_imu="EEG_IMU_BiCMHA_fusion",
        branch_zero_imu="EEG_zeroIMU_BiCMHA_fusion",
    ),
    FusionBaseline(
        id=7,
        name="Baseline 7: Normal MHA fusion",
        fusion="self_mha",
        pretrain_stage="B7 Stage 1: ME EEG+IMU normal MHA pretraining",
        notes_after_pretrain="ME EEG+IMU pretrained model evaluated on ME test using real IMU; normal MHA fusion over fused EEG+IMU tokens",
        notes_mi_test="ME EEG+IMU pretraining with normal MHA fusion; MI uses zero IMU; no distillation/prototype alignment",
        notes_me_after_finetune="Final MI-fine-tuned model evaluated on ME test with zero IMU input; normal MHA fusion",
        branch_real_imu="EEG_IMU_NormalMHA_fusion",
        branch_zero_imu="EEG_zeroIMU_NormalMHA_fusion",
    ),
]


def fusion_baseline(exp: Experiment, spec: FusionBaseline):
    """Run Baseline 3, 6 or 7.

    The model is pretrained on ME with the EEG and the real IMU, then fine-tuned on MI
    with the IMU input set to zero.

    Parameters
    ----------
    exp : Experiment
        The subject's data and results.
    spec : FusionBaseline
        Which of the three baselines to run.
    """
    attention_heads = {
        "cross_mha": exp.cfg.cross_attention_heads,
        "self_mha": exp.cfg.normal_mha_heads,
    }
    num_heads = attention_heads.get(spec.fusion)
    model = FusionModel(
        spec.fusion,
        **exp.model_dims,
        target_len=TARGET_LEN,
        latent_dim=exp.cfg.latent_dim,
        num_heads=num_heads,
        dropout=exp.cfg.dropout,
    ).to(exp.device)
    loss = exp.regression_loss()
    real_imu, zero_imu = Supervised(fusion_real_imu, loss), Supervised(fusion_zero_imu, loss)

    exp.fit(spec.pretrain_stage, model, "me", real_imu, exp.pretrain)
    exp.record(
        spec.id,
        spec.name,
        exp.evaluate(model, "me_test", real_imu).flat(),
        spec.notes_after_pretrain,
        "ME_test_after_pretrain",
        spec.branch_real_imu,
    )

    exp.fit(
        f"B{spec.id} Stage 2: MI EEG-only fine-tuning with zero IMU",
        model,
        "mi",
        zero_imu,
        exp.finetune,
    )
    exp.record(
        spec.id,
        spec.name,
        exp.evaluate(model, "mi_test", zero_imu).flat(),
        spec.notes_mi_test,
        "MI_test",
        spec.branch_zero_imu,
    )
    exp.record(
        spec.id,
        spec.name,
        exp.evaluate(model, "me_eeg_test", zero_imu).flat(),
        spec.notes_me_after_finetune,
        "ME_test_after_MI_finetune_zeroIMU",
        spec.branch_zero_imu,
    )


def baseline_4(exp: Experiment):
    """Run Baseline 4: teacher-student distillation on ME, then fine-tune the student on MI."""
    cfg = exp.cfg
    name = "Baseline 4: ME distillation + MI fine-tune"
    model = DistillationSimpleModel(
        **exp.model_dims,
        target_len=TARGET_LEN,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
    ).to(exp.device)
    loss = exp.regression_loss()
    distill = TeacherStudent(
        loss,
        cfg.teacher_loss_weight,
        cfg.student_sup_loss_weight,
        cfg.kd_loss_weight,
        kd_beta=cfg.kd_beta,
    )

    exp.fit("B4 Stage 1: ME teacher-student distillation", model, "me", distill, exp.pretrain)
    after_pretrain = exp.evaluate(model, "me_test", distill)
    exp.record(
        4,
        name,
        after_pretrain.metrics["teacher"],
        "ME EEG+IMU teacher evaluated on ME test after distillation pretraining",
        "ME_test_after_pretrain",
        "teacher_EEG_IMU",
    )
    exp.record(
        4,
        name,
        after_pretrain.metrics["student"],
        "EEG-only student evaluated on ME test after distillation pretraining",
        "ME_test_after_pretrain",
        "student_EEG_only",
    )

    freeze_all_except(model, model.eeg_encoder, model.student_head)
    student = Supervised(distill_student, loss)
    exp.fit("B4 Stage 2: MI student fine-tuning", model, "mi", student, exp.finetune)
    exp.record(
        4,
        name,
        exp.evaluate(model, "mi_test", student).flat(),
        "ME EEG+IMU teacher teaches EEG-only student; final MI test; no prototype alignment",
        "MI_test",
        "student_EEG_only",
    )
    exp.record(
        4,
        name,
        exp.evaluate(model, "me_eeg_test", student).flat(),
        "Final MI-fine-tuned EEG-only student evaluated on ME test",
        "ME_test_after_MI_finetune",
        "student_EEG_only",
    )


def baseline_5(exp: Experiment):
    """Run Baseline 5, the full model.

    Like Baseline 4, but the teacher uses lag-aware fusion, the student's EEG features
    get a learned per-subject scale and shift, and MI fine-tuning also pulls the
    student's features toward ME prototypes of the same target.
    """
    cfg = exp.cfg
    name = "Baseline 5: Full model"
    model = FullPrivilegedModel(
        **exp.model_dims,
        n_subjects=1,
        target_len=TARGET_LEN,
        latent_dim=cfg.latent_dim,
        lag_radius=cfg.lag_radius,
        dropout=cfg.dropout,
    ).to(exp.device)
    loss = exp.regression_loss()
    distill = TeacherStudent(
        loss,
        cfg.teacher_loss_weight,
        cfg.student_sup_loss_weight,
        cfg.kd_loss_weight,
        kd_beta=cfg.kd_beta,
    )

    exp.fit("B5 Stage 1: ME full privileged pretraining", model, "me", distill, exp.pretrain)
    after_pretrain = exp.evaluate(model, "me_test", distill)
    exp.record(
        5,
        name,
        after_pretrain.metrics["teacher"],
        "Lag-aware EEG+IMU teacher evaluated on ME test after full pretraining",
        "ME_test_after_pretrain",
        "teacher_lag_EEG_IMU",
    )
    exp.record(
        5,
        name,
        after_pretrain.metrics["student"],
        "EEG-only student evaluated on ME test after full pretraining",
        "ME_test_after_pretrain",
        "student_EEG_only",
    )

    prototypes = compute_me_prototypes(
        model, exp.loaders["me_train"], exp.device, num_targets=len(TARGET_ORDER)
    )
    log.info("Computed ME prototypes: %s", tuple(prototypes.shape))

    freeze_all_except(model, model.eeg_encoder, model.subject_film, model.student_head)
    exp.fit(
        "B5 Stage 2: MI fine-tuning with prototype alignment",
        model,
        "mi",
        PrototypeAlignedStudent(loss, prototypes, cfg.proto_loss_weight),
        exp.finetune,
    )

    # Evaluate with the regression loss alone, without the prototype term.
    student = PrototypeAlignedStudent(loss)
    exp.record(
        5,
        name,
        exp.evaluate(model, "mi_test", student).flat(),
        "Lag-aware fusion + distillation + FiLM subject adaptation + ME prototype alignment; final MI test",
        "MI_test",
        "student_EEG_only",
    )
    exp.record(
        5,
        name,
        exp.evaluate(model, "me_eeg_test", student).flat(),
        "Final MI-fine-tuned EEG-only student evaluated on ME test",
        "ME_test_after_MI_finetune",
        "student_EEG_only",
    )


BASELINES = {
    0: baseline_0,
    1: baseline_1,
    2: baseline_2,
    4: baseline_4,
    5: baseline_5,
    **{spec.id: partial(fusion_baseline, spec=spec) for spec in FUSION_BASELINES},
}
