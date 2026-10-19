"""
Baselines 0-7. Each one trains its model(s) and records test results via Experiment.

Baseline 0: Time-lagged Linear method (MLR / Ridge-stabilized MLR)
Baseline 1: MI-only EEG model
Baseline 2: ME EEG-only pretraining -> MI EEG-only fine-tuning
Baseline 3: ME EEG+IMU pretraining -> MI EEG-only fine-tuning with zero IMU
            No distillation, no prototype alignment
Baseline 4: ME EEG+IMU teacher + EEG-only student distillation -> MI fine-tuning
            No prototype alignment
Baseline 5: Full model
            ME EEG+IMU + lag-aware fusion + student distillation
            + subject adaptation + prototype alignment
Baseline 6: Baseline 3 with bidirectional cross-modal multi-head attention fusion
Baseline 7: Baseline 3 with normal multi-head self-attention over fused EEG+IMU tokens

Important
---------
- Final model evaluation is always on the MI test set.
- IMU is not used during MI testing for Baseline 4 and 5.
- Baselines 3, 6 and 7 use zero IMU during MI fine-tuning/testing, to test whether
  simple ME+IMU pretraining is enough without distillation/prototype alignment.
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
    set_distill_student_finetune_mode,
    set_full_student_finetune_mode,
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
    name = "Baseline 0: Time-lagged MLR"
    for data, task, notes in (
        (exp.mi, "MI_test", "MI train+val -> MI test; time-lagged ridge-stabilized MLR"),
        (exp.me, "ME_test", "ME train+val -> ME test; time-lagged ridge-stabilized MLR"),
    ):
        metrics = run_baseline0_linear(data, exp.scaler, exp.dim_names, exp.cfg)
        exp.record(0, name, metrics, notes, task, "linear")


def baseline_1(exp: Experiment):
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
    """Baselines 3, 6 and 7 differ only in the fusion block and their labels."""

    id: int
    name: str
    fusion: str  # "concat", "cross_mha" or "self_mha" (see models.FusionModel)
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
    """ME EEG+IMU pretraining -> MI fine-tuning with zero IMU; no distillation/prototypes."""
    num_heads = {
        "cross_mha": exp.cfg.cross_attention_heads,
        "self_mha": exp.cfg.normal_mha_heads,
    }.get(spec.fusion, 4)
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
    after_pretrain = exp.evaluate(model, "me_test", distill)  # one pass scores teacher and student
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

    set_distill_student_finetune_mode(model)
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
        use_subject_id=True,
    )

    exp.fit("B5 Stage 1: ME full privileged pretraining", model, "me", distill, exp.pretrain)
    after_pretrain = exp.evaluate(model, "me_test", distill)  # one pass scores teacher and student
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

    set_full_student_finetune_mode(model)
    exp.fit(
        "B5 Stage 2: MI fine-tuning with prototype alignment",
        model,
        "mi",
        PrototypeAlignedStudent(loss, prototypes, cfg.proto_loss_weight),
        exp.finetune,
    )

    student = PrototypeAlignedStudent(loss)  # evaluation: regression loss only
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
