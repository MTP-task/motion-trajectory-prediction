"""
Run Baseline 0-7 and save ME/MI correlation results to Excel.

Baselines
---------
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
Baseline 6: ME EEG+IMU pretraining -> MI EEG-only fine-tuning with zero IMU
            Bidirectional Cross-Modal Multi-Head Attention fusion
            No distillation, no prototype alignment
Baseline 7: ME EEG+IMU pretraining -> MI EEG-only fine-tuning with zero IMU
            Normal Multi-Head Self-Attention fusion over fused EEG+IMU tokens
            No distillation, no prototype alignment

Important
---------
- labels_clean is required for both ME and MI for stratified split / target class IDs.
- New trajectory files are already mapped to EEG epoch order:
  each key contains an array with shape (epoch, 150, 3) or (epoch, 150, 6 for IMU).
- EEG input epoch is expected to cover -2..7 s.
- Filter-bank filtering is applied to the full -2..7 s epoch first,
  then the model input is cropped to 0..5 s.
- Final model evaluation is always on the MI test set.
- IMU is not used during MI testing for Baseline 4 and 5.
- Baseline 3 uses zero IMU during MI fine-tuning/testing to test whether
  simple ME+IMU pretraining is enough without distillation/prototype alignment.
- Baseline 6 copies Baseline 3 training/evaluation, but replaces simple
  concatenation fusion with bidirectional cross-modal multi-head attention.
- Baseline 7 copies Baseline 3 training/evaluation, but replaces simple
  concatenation fusion with normal multi-head self-attention over fused EEG+IMU tokens.

How to run
----------
1. Change `subject` and paths in USER SETTINGS (mtp/config.py).
2. Make sure these files exist:
    EEG_data_<subject>_ME.npy
    EEG_data_<subject>_MI.npy
    trajectory_<subject>_ME.npy
    trajectory_<subject>_MI.npy
    labels_clean_<subject>_ME.npy
    labels_clean_<subject>_MI.npy
3. Run from the repo root:
    python -m mtp.modeling.train

Output
------
Excel file:
    baseline_results_<subject>.xlsx

Sheets:
    summary
    axis_metrics
    stages
    config
"""

import random
import time

import numpy as np

import torch
import torch.nn as nn

from mtp.config import (
    BATCH_SIZE, CROSS_ATTENTION_HEADS, FINETUNE_EPOCHS, FINETUNE_LR, LAG_RADIUS, LATENT_DIM,
    NORMAL_MHA_HEADS, PATIENCE_FINETUNE, PATIENCE_PRETRAIN, PRETRAIN_EPOCHS, PRETRAIN_LR,
    RUN_BASELINE_0, RUN_BASELINE_1, RUN_BASELINE_2, RUN_BASELINE_3, RUN_BASELINE_4,
    RUN_BASELINE_5, RUN_BASELINE_6, RUN_BASELINE_7, SEED, TARGET_LEN, WEIGHT_DECAY,
)
from mtp.dataset import (
    fit_target_scaler, make_eeg_imu_loader, make_eeg_loader, normalize_targets,
    prepare_me_dataset, prepare_mi_dataset,
)
from mtp.modeling.models import (
    CrossModalAttentionFusionModel, DistillationSimpleModel, EEGOnlyModel, FullPrivilegedModel,
    NormalMHAFusionModel, SimpleFusionModel,
)
from mtp.modeling.metrics import SequenceRegressionLoss, long_axis_rows
from mtp.modeling.trainers import (
    TrainConfig, compute_me_prototypes, run_distill_pretrain_epoch, run_distill_student_epoch,
    run_eeg_epoch, run_full_pretrain_epoch, run_full_student_epoch, run_fusion_epoch,
    train_distill_pretrain, train_distill_student, train_eeg_model, train_full_pretrain,
    train_full_student, train_fusion_model,
)
from mtp.modeling.linear import run_baseline0_linear
from mtp.modeling.results import save_results_to_excel, summarize_baseline


# ============================================================
# 1) REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 12) MAIN
# ============================================================

def main():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    me = prepare_me_dataset()
    mi = prepare_mi_dataset()

    if me["n_bands"] != mi["n_bands"]:
        raise ValueError("ME and MI have different number of filter-bank bands.")
    if me["n_channels"] != mi["n_channels"]:
        raise ValueError("ME and MI have different number of EEG channels.")
    if me["out_dim"] != mi["out_dim"]:
        raise ValueError("ME and MI targets have different output dimensions.")

    dim_names = me["dim_names"]
    if me["dim_names"] != mi["dim_names"]:
        print("Warning: ME and MI dim_names are different.")
        print("ME:", me["dim_names"])
        print("MI:", mi["dim_names"])

    scaler = fit_target_scaler(me["y_train"], mi["y_train"])

    for d in [me, mi]:
        d["y_train_n"] = normalize_targets(d["y_train"], scaler)
        d["y_val_n"] = normalize_targets(d["y_val"], scaler)
        d["y_test_n"] = normalize_targets(d["y_test"], scaler)

    me_train_loader = make_eeg_imu_loader(me["X_train"], me["imu_train"], me["y_train_n"], me["id_train"], me["sub_train"], BATCH_SIZE, True)
    me_val_loader = make_eeg_imu_loader(me["X_val"], me["imu_val"], me["y_val_n"], me["id_val"], me["sub_val"], BATCH_SIZE, False)
    me_test_loader = make_eeg_imu_loader(me["X_test"], me["imu_test"], me["y_test_n"], me["id_test"], me["sub_test"], BATCH_SIZE, False)

    me_eeg_train_loader = make_eeg_loader(me["X_train"], me["y_train_n"], me["id_train"], me["sub_train"], BATCH_SIZE, True)
    me_eeg_val_loader = make_eeg_loader(me["X_val"], me["y_val_n"], me["id_val"], me["sub_val"], BATCH_SIZE, False)
    me_eeg_test_loader = make_eeg_loader(me["X_test"], me["y_test_n"], me["id_test"], me["sub_test"], BATCH_SIZE, False)

    mi_train_loader = make_eeg_loader(mi["X_train"], mi["y_train_n"], mi["id_train"], mi["sub_train"], BATCH_SIZE, True)
    mi_val_loader = make_eeg_loader(mi["X_val"], mi["y_val_n"], mi["id_val"], mi["sub_val"], BATCH_SIZE, False)
    mi_test_loader = make_eeg_loader(mi["X_test"], mi["y_test_n"], mi["id_test"], mi["sub_test"], BATCH_SIZE, False)

    pretrain_cfg = TrainConfig(PRETRAIN_LR, WEIGHT_DECAY, PRETRAIN_EPOCHS, PATIENCE_PRETRAIN, device)
    finetune_cfg = TrainConfig(FINETUNE_LR, WEIGHT_DECAY, FINETUNE_EPOCHS, PATIENCE_FINETUNE, device)

    criterion = SequenceRegressionLoss(corr_weight=0.10)

    summary_rows = []
    axis_rows = []
    stage_rows_all = []

    # ------------------------------------------------------------
    # Baseline 0: Time-lagged MLR
    # ------------------------------------------------------------
    if RUN_BASELINE_0:
        set_seed(SEED)
        name = "Baseline 0: Time-lagged MLR"

        metrics_mi = run_baseline0_linear(mi, scaler, dim_names)
        print(f"\n{name} MI TEST: r_mean={metrics_mi['r_mean']:.4f}, RMSE={metrics_mi['rmse_mean']:.4f}, MAE={metrics_mi['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            0, name, metrics_mi,
            "MI train+val -> MI test; time-lagged ridge-stabilized MLR",
            evaluation_task="MI_test",
            branch="linear",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_mi, dim_names, evaluation_task="MI_test", branch="linear"))

        metrics_me = run_baseline0_linear(me, scaler, dim_names)
        print(f"\n{name} ME TEST: r_mean={metrics_me['r_mean']:.4f}, RMSE={metrics_me['rmse_mean']:.4f}, MAE={metrics_me['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            0, name, metrics_me,
            "ME train+val -> ME test; time-lagged ridge-stabilized MLR",
            evaluation_task="ME_test",
            branch="linear",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me, dim_names, evaluation_task="ME_test", branch="linear"))

    # ------------------------------------------------------------
    # Baseline 1: MI-only EEG
    # ------------------------------------------------------------
    if RUN_BASELINE_1:
        set_seed(SEED)
        name = "Baseline 1: MI-only EEG"
        model = EEGOnlyModel(me["n_bands"], me["n_channels"], me["out_dim"], TARGET_LEN, LATENT_DIM).to(device)
        model, rows = train_eeg_model(name, model, mi_train_loader, mi_val_loader, scaler, dim_names, finetune_cfg)
        stage_rows_all.extend(rows)

        metrics_mi = run_eeg_epoch(model, mi_test_loader, criterion, scaler, dim_names, None, device)
        print(f"\n{name} MI TEST: r_mean={metrics_mi['r_mean']:.4f}, RMSE={metrics_mi['rmse_mean']:.4f}, MAE={metrics_mi['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            1, name, metrics_mi,
            "Train only on MI EEG; evaluate on MI test",
            evaluation_task="MI_test",
            branch="EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_mi, dim_names, evaluation_task="MI_test", branch="EEG_only"))

        metrics_me = run_eeg_epoch(model, me_eeg_test_loader, criterion, scaler, dim_names, None, device)
        print(f"\n{name} ME TEST: r_mean={metrics_me['r_mean']:.4f}, RMSE={metrics_me['rmse_mean']:.4f}, MAE={metrics_me['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            1, name, metrics_me,
            "Same MI-only trained EEG model evaluated on ME test",
            evaluation_task="ME_test_final_model",
            branch="EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me, dim_names, evaluation_task="ME_test_final_model", branch="EEG_only"))

    # ------------------------------------------------------------
    # Baseline 2: ME EEG pretrain -> MI fine-tune
    # ------------------------------------------------------------
    if RUN_BASELINE_2:
        set_seed(SEED)
        name = "Baseline 2: ME EEG pretrain + MI fine-tune"
        model = EEGOnlyModel(me["n_bands"], me["n_channels"], me["out_dim"], TARGET_LEN, LATENT_DIM).to(device)

        model, rows = train_eeg_model("B2 Stage 1: ME EEG-only pretraining", model, me_eeg_train_loader, me_eeg_val_loader, scaler, dim_names, pretrain_cfg)
        stage_rows_all.extend(rows)

        metrics_me_pre = run_eeg_epoch(model, me_eeg_test_loader, criterion, scaler, dim_names, None, device)
        print(f"\n{name} ME TEST after pretraining: r_mean={metrics_me_pre['r_mean']:.4f}, RMSE={metrics_me_pre['rmse_mean']:.4f}, MAE={metrics_me_pre['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            2, name, metrics_me_pre,
            "After ME EEG-only pretraining, before MI fine-tuning",
            evaluation_task="ME_test_after_pretrain",
            branch="EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_pre, dim_names, evaluation_task="ME_test_after_pretrain", branch="EEG_only"))

        model, rows = train_eeg_model("B2 Stage 2: MI EEG-only fine-tuning", model, mi_train_loader, mi_val_loader, scaler, dim_names, finetune_cfg)
        stage_rows_all.extend(rows)

        metrics_mi = run_eeg_epoch(model, mi_test_loader, criterion, scaler, dim_names, None, device)
        print(f"\n{name} MI TEST: r_mean={metrics_mi['r_mean']:.4f}, RMSE={metrics_mi['rmse_mean']:.4f}, MAE={metrics_mi['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            2, name, metrics_mi,
            "ME EEG-only pretraining, then MI EEG-only fine-tuning; final MI test",
            evaluation_task="MI_test",
            branch="EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_mi, dim_names, evaluation_task="MI_test", branch="EEG_only"))

        metrics_me_final = run_eeg_epoch(model, me_eeg_test_loader, criterion, scaler, dim_names, None, device)
        print(f"\n{name} ME TEST after MI fine-tuning: r_mean={metrics_me_final['r_mean']:.4f}, RMSE={metrics_me_final['rmse_mean']:.4f}, MAE={metrics_me_final['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            2, name, metrics_me_final,
            "Final MI-fine-tuned model evaluated on ME test",
            evaluation_task="ME_test_after_MI_finetune",
            branch="EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_final, dim_names, evaluation_task="ME_test_after_MI_finetune", branch="EEG_only"))

    # ------------------------------------------------------------
    # Baseline 3: ME EEG+IMU pretrain, no distill/proto
    # ------------------------------------------------------------
    if RUN_BASELINE_3:
        set_seed(SEED)
        name = "Baseline 3: ME EEG+IMU pretrain, no distill/proto"
        model = SimpleFusionModel(me["n_bands"], me["n_channels"], me["out_dim"], TARGET_LEN, LATENT_DIM).to(device)

        model, rows = train_fusion_model("B3 Stage 1: ME EEG+IMU pretraining", model, me_train_loader, me_val_loader, scaler, dim_names, pretrain_cfg, use_real_imu=True)
        stage_rows_all.extend(rows)

        metrics_me_pre = run_fusion_epoch(model, me_test_loader, criterion, scaler, dim_names, None, device, use_real_imu=True)
        print(f"\n{name} ME TEST after pretraining: r_mean={metrics_me_pre['r_mean']:.4f}, RMSE={metrics_me_pre['rmse_mean']:.4f}, MAE={metrics_me_pre['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            3, name, metrics_me_pre,
            "ME EEG+IMU pretrained model evaluated on ME test using real IMU",
            evaluation_task="ME_test_after_pretrain",
            branch="EEG_IMU_fusion",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_pre, dim_names, evaluation_task="ME_test_after_pretrain", branch="EEG_IMU_fusion"))

        # Fine-tune on MI with zero IMU.
        model, rows = train_fusion_model("B3 Stage 2: MI EEG-only fine-tuning with zero IMU", model, mi_train_loader, mi_val_loader, scaler, dim_names, finetune_cfg, use_real_imu=False)
        stage_rows_all.extend(rows)

        metrics_mi = run_fusion_epoch(model, mi_test_loader, criterion, scaler, dim_names, None, device, use_real_imu=False)
        print(f"\n{name} MI TEST: r_mean={metrics_mi['r_mean']:.4f}, RMSE={metrics_mi['rmse_mean']:.4f}, MAE={metrics_mi['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            3, name, metrics_mi,
            "ME EEG+IMU pretraining; MI uses zero IMU; no distillation/prototype alignment",
            evaluation_task="MI_test",
            branch="EEG_zeroIMU_fusion",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_mi, dim_names, evaluation_task="MI_test", branch="EEG_zeroIMU_fusion"))

        metrics_me_final_zero = run_fusion_epoch(model, me_eeg_test_loader, criterion, scaler, dim_names, None, device, use_real_imu=False)
        print(f"\n{name} ME TEST after MI fine-tuning with zero IMU: r_mean={metrics_me_final_zero['r_mean']:.4f}, RMSE={metrics_me_final_zero['rmse_mean']:.4f}, MAE={metrics_me_final_zero['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            3, name, metrics_me_final_zero,
            "Final MI-fine-tuned model evaluated on ME test with zero IMU input",
            evaluation_task="ME_test_after_MI_finetune_zeroIMU",
            branch="EEG_zeroIMU_fusion",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_final_zero, dim_names, evaluation_task="ME_test_after_MI_finetune_zeroIMU", branch="EEG_zeroIMU_fusion"))

    # ------------------------------------------------------------
    # Baseline 4: ME pretrain + distillation
    # ------------------------------------------------------------
    if RUN_BASELINE_4:
        set_seed(SEED)
        name = "Baseline 4: ME distillation + MI fine-tune"
        model = DistillationSimpleModel(me["n_bands"], me["n_channels"], me["out_dim"], TARGET_LEN, LATENT_DIM).to(device)

        model, rows = train_distill_pretrain("B4 Stage 1: ME teacher-student distillation", model, me_train_loader, me_val_loader, scaler, dim_names, pretrain_cfg)
        stage_rows_all.extend(rows)

        me_pre = run_distill_pretrain_epoch(model, me_test_loader, SequenceRegressionLoss(corr_weight=0.10), nn.SmoothL1Loss(beta=0.5), scaler, dim_names, None, device)
        metrics_me_teacher = me_pre["teacher"]
        metrics_me_student = me_pre["student"]
        print(f"\n{name} ME TEST teacher after distillation: r_mean={metrics_me_teacher['r_mean']:.4f}, RMSE={metrics_me_teacher['rmse_mean']:.4f}, MAE={metrics_me_teacher['mae_mean']:.4f}")
        print(f"{name} ME TEST student after distillation: r_mean={metrics_me_student['r_mean']:.4f}, RMSE={metrics_me_student['rmse_mean']:.4f}, MAE={metrics_me_student['mae_mean']:.4f}")

        summary_rows.append(summarize_baseline(
            4, name, metrics_me_teacher,
            "ME EEG+IMU teacher evaluated on ME test after distillation pretraining",
            evaluation_task="ME_test_after_pretrain",
            branch="teacher_EEG_IMU",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_teacher, dim_names, evaluation_task="ME_test_after_pretrain", branch="teacher_EEG_IMU"))

        summary_rows.append(summarize_baseline(
            4, name, metrics_me_student,
            "EEG-only student evaluated on ME test after distillation pretraining",
            evaluation_task="ME_test_after_pretrain",
            branch="student_EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_student, dim_names, evaluation_task="ME_test_after_pretrain", branch="student_EEG_only"))

        model, rows = train_distill_student("B4 Stage 2: MI student fine-tuning", model, mi_train_loader, mi_val_loader, scaler, dim_names, finetune_cfg)
        stage_rows_all.extend(rows)

        metrics_mi = run_distill_student_epoch(model, mi_test_loader, criterion, scaler, dim_names, None, device)
        print(f"\n{name} MI TEST: r_mean={metrics_mi['r_mean']:.4f}, RMSE={metrics_mi['rmse_mean']:.4f}, MAE={metrics_mi['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            4, name, metrics_mi,
            "ME EEG+IMU teacher teaches EEG-only student; final MI test; no prototype alignment",
            evaluation_task="MI_test",
            branch="student_EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_mi, dim_names, evaluation_task="MI_test", branch="student_EEG_only"))

        metrics_me_final = run_distill_student_epoch(model, me_eeg_test_loader, criterion, scaler, dim_names, None, device)
        print(f"\n{name} ME TEST after MI fine-tuning: r_mean={metrics_me_final['r_mean']:.4f}, RMSE={metrics_me_final['rmse_mean']:.4f}, MAE={metrics_me_final['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            4, name, metrics_me_final,
            "Final MI-fine-tuned EEG-only student evaluated on ME test",
            evaluation_task="ME_test_after_MI_finetune",
            branch="student_EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_final, dim_names, evaluation_task="ME_test_after_MI_finetune", branch="student_EEG_only"))

    # ------------------------------------------------------------
    # Baseline 5: Full model
    # ------------------------------------------------------------
    if RUN_BASELINE_5:
        set_seed(SEED)
        name = "Baseline 5: Full model"
        model = FullPrivilegedModel(
            n_bands=me["n_bands"],
            n_channels=me["n_channels"],
            out_dim=me["out_dim"],
            n_subjects=1,
            target_len=TARGET_LEN,
            latent_dim=LATENT_DIM,
            lag_radius=LAG_RADIUS,
            dropout=0.20,
        ).to(device)

        model, rows = train_full_pretrain("B5 Stage 1: ME full privileged pretraining", model, me_train_loader, me_val_loader, scaler, dim_names, pretrain_cfg)
        stage_rows_all.extend(rows)

        me_pre = run_full_pretrain_epoch(model, me_test_loader, SequenceRegressionLoss(corr_weight=0.10), nn.SmoothL1Loss(beta=0.5), scaler, dim_names, None, device)
        metrics_me_teacher = me_pre["teacher"]
        metrics_me_student = me_pre["student"]
        print(f"\n{name} ME TEST teacher after full pretraining: r_mean={metrics_me_teacher['r_mean']:.4f}, RMSE={metrics_me_teacher['rmse_mean']:.4f}, MAE={metrics_me_teacher['mae_mean']:.4f}")
        print(f"{name} ME TEST student after full pretraining: r_mean={metrics_me_student['r_mean']:.4f}, RMSE={metrics_me_student['rmse_mean']:.4f}, MAE={metrics_me_student['mae_mean']:.4f}")

        summary_rows.append(summarize_baseline(
            5, name, metrics_me_teacher,
            "Lag-aware EEG+IMU teacher evaluated on ME test after full pretraining",
            evaluation_task="ME_test_after_pretrain",
            branch="teacher_lag_EEG_IMU",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_teacher, dim_names, evaluation_task="ME_test_after_pretrain", branch="teacher_lag_EEG_IMU"))

        summary_rows.append(summarize_baseline(
            5, name, metrics_me_student,
            "EEG-only student evaluated on ME test after full pretraining",
            evaluation_task="ME_test_after_pretrain",
            branch="student_EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_student, dim_names, evaluation_task="ME_test_after_pretrain", branch="student_EEG_only"))

        prototypes = compute_me_prototypes(model, me_train_loader, device, num_targets=4)
        print("Computed ME prototypes:", tuple(prototypes.shape))

        model, rows = train_full_student("B5 Stage 2: MI fine-tuning with prototype alignment", model, mi_train_loader, mi_val_loader, scaler, dim_names, prototypes, finetune_cfg)
        stage_rows_all.extend(rows)

        metrics_mi = run_full_student_epoch(model, mi_test_loader, criterion, scaler, dim_names, prototypes=None, proto_weight=0.0, optimizer=None, device=device)
        print(f"\n{name} MI TEST: r_mean={metrics_mi['r_mean']:.4f}, RMSE={metrics_mi['rmse_mean']:.4f}, MAE={metrics_mi['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            5, name, metrics_mi,
            "Lag-aware fusion + distillation + FiLM subject adaptation + ME prototype alignment; final MI test",
            evaluation_task="MI_test",
            branch="student_EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_mi, dim_names, evaluation_task="MI_test", branch="student_EEG_only"))

        metrics_me_final = run_full_student_epoch(model, me_eeg_test_loader, criterion, scaler, dim_names, prototypes=None, proto_weight=0.0, optimizer=None, device=device)
        print(f"\n{name} ME TEST after MI fine-tuning: r_mean={metrics_me_final['r_mean']:.4f}, RMSE={metrics_me_final['rmse_mean']:.4f}, MAE={metrics_me_final['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            5, name, metrics_me_final,
            "Final MI-fine-tuned EEG-only student evaluated on ME test",
            evaluation_task="ME_test_after_MI_finetune",
            branch="student_EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_final, dim_names, evaluation_task="ME_test_after_MI_finetune", branch="student_EEG_only"))


    # ------------------------------------------------------------
    # Baseline 6: ME EEG+IMU pretrain with bidirectional cross-modal MHA,
    #             no distillation/prototype alignment
    # ------------------------------------------------------------
    if RUN_BASELINE_6:
        set_seed(SEED)
        name = "Baseline 6: Bidirectional cross-modal MHA fusion"
        model = CrossModalAttentionFusionModel(
            me["n_bands"],
            me["n_channels"],
            me["out_dim"],
            TARGET_LEN,
            LATENT_DIM,
            CROSS_ATTENTION_HEADS,
        ).to(device)

        model, rows = train_fusion_model(
            "B6 Stage 1: ME EEG+IMU bidirectional cross-modal MHA pretraining",
            model,
            me_train_loader,
            me_val_loader,
            scaler,
            dim_names,
            pretrain_cfg,
            use_real_imu=True,
        )
        stage_rows_all.extend(rows)

        metrics_me_pre = run_fusion_epoch(
            model,
            me_test_loader,
            criterion,
            scaler,
            dim_names,
            None,
            device,
            use_real_imu=True,
        )
        print(
            f"\n{name} ME TEST after pretraining: "
            f"r_mean={metrics_me_pre['r_mean']:.4f}, "
            f"RMSE={metrics_me_pre['rmse_mean']:.4f}, "
            f"MAE={metrics_me_pre['mae_mean']:.4f}"
        )
        summary_rows.append(summarize_baseline(
            6, name, metrics_me_pre,
            "ME EEG+IMU pretrained model evaluated on ME test using real IMU; bidirectional cross-modal MHA fusion",
            evaluation_task="ME_test_after_pretrain",
            branch="EEG_IMU_BiCMHA_fusion",
        ))
        axis_rows.extend(long_axis_rows(
            name,
            metrics_me_pre,
            dim_names,
            evaluation_task="ME_test_after_pretrain",
            branch="EEG_IMU_BiCMHA_fusion",
        ))

        # Fine-tune on MI with zero IMU, exactly like Baseline 3.
        model, rows = train_fusion_model(
            "B6 Stage 2: MI EEG-only fine-tuning with zero IMU",
            model,
            mi_train_loader,
            mi_val_loader,
            scaler,
            dim_names,
            finetune_cfg,
            use_real_imu=False,
        )
        stage_rows_all.extend(rows)

        metrics_mi = run_fusion_epoch(
            model,
            mi_test_loader,
            criterion,
            scaler,
            dim_names,
            None,
            device,
            use_real_imu=False,
        )
        print(
            f"\n{name} MI TEST: "
            f"r_mean={metrics_mi['r_mean']:.4f}, "
            f"RMSE={metrics_mi['rmse_mean']:.4f}, "
            f"MAE={metrics_mi['mae_mean']:.4f}"
        )
        summary_rows.append(summarize_baseline(
            6, name, metrics_mi,
            "ME EEG+IMU pretraining with bidirectional cross-modal MHA; MI uses zero IMU; no distillation/prototype alignment",
            evaluation_task="MI_test",
            branch="EEG_zeroIMU_BiCMHA_fusion",
        ))
        axis_rows.extend(long_axis_rows(
            name,
            metrics_mi,
            dim_names,
            evaluation_task="MI_test",
            branch="EEG_zeroIMU_BiCMHA_fusion",
        ))

        metrics_me_final_zero = run_fusion_epoch(
            model,
            me_eeg_test_loader,
            criterion,
            scaler,
            dim_names,
            None,
            device,
            use_real_imu=False,
        )
        print(
            f"\n{name} ME TEST after MI fine-tuning with zero IMU: "
            f"r_mean={metrics_me_final_zero['r_mean']:.4f}, "
            f"RMSE={metrics_me_final_zero['rmse_mean']:.4f}, "
            f"MAE={metrics_me_final_zero['mae_mean']:.4f}"
        )
        summary_rows.append(summarize_baseline(
            6, name, metrics_me_final_zero,
            "Final MI-fine-tuned model evaluated on ME test with zero IMU input; bidirectional cross-modal MHA fusion",
            evaluation_task="ME_test_after_MI_finetune_zeroIMU",
            branch="EEG_zeroIMU_BiCMHA_fusion",
        ))
        axis_rows.extend(long_axis_rows(
            name,
            metrics_me_final_zero,
            dim_names,
            evaluation_task="ME_test_after_MI_finetune_zeroIMU",
            branch="EEG_zeroIMU_BiCMHA_fusion",
        ))



    # ------------------------------------------------------------
    # Baseline 7: ME EEG+IMU pretrain with normal MHA fusion,
    #             no distillation/prototype alignment
    # ------------------------------------------------------------
    if RUN_BASELINE_7:
        set_seed(SEED)
        name = "Baseline 7: Normal MHA fusion"
        model = NormalMHAFusionModel(
            me["n_bands"],
            me["n_channels"],
            me["out_dim"],
            TARGET_LEN,
            LATENT_DIM,
            NORMAL_MHA_HEADS,
        ).to(device)

        model, rows = train_fusion_model(
            "B7 Stage 1: ME EEG+IMU normal MHA pretraining",
            model,
            me_train_loader,
            me_val_loader,
            scaler,
            dim_names,
            pretrain_cfg,
            use_real_imu=True,
        )
        stage_rows_all.extend(rows)

        metrics_me_pre = run_fusion_epoch(
            model,
            me_test_loader,
            criterion,
            scaler,
            dim_names,
            None,
            device,
            use_real_imu=True,
        )
        print(
            f"\n{name} ME TEST after pretraining: "
            f"r_mean={metrics_me_pre['r_mean']:.4f}, "
            f"RMSE={metrics_me_pre['rmse_mean']:.4f}, "
            f"MAE={metrics_me_pre['mae_mean']:.4f}"
        )
        summary_rows.append(summarize_baseline(
            7, name, metrics_me_pre,
            "ME EEG+IMU pretrained model evaluated on ME test using real IMU; normal MHA fusion over fused EEG+IMU tokens",
            evaluation_task="ME_test_after_pretrain",
            branch="EEG_IMU_NormalMHA_fusion",
        ))
        axis_rows.extend(long_axis_rows(
            name,
            metrics_me_pre,
            dim_names,
            evaluation_task="ME_test_after_pretrain",
            branch="EEG_IMU_NormalMHA_fusion",
        ))

        # Fine-tune on MI with zero IMU, exactly like Baseline 3 and Baseline 6.
        model, rows = train_fusion_model(
            "B7 Stage 2: MI EEG-only fine-tuning with zero IMU",
            model,
            mi_train_loader,
            mi_val_loader,
            scaler,
            dim_names,
            finetune_cfg,
            use_real_imu=False,
        )
        stage_rows_all.extend(rows)

        metrics_mi = run_fusion_epoch(
            model,
            mi_test_loader,
            criterion,
            scaler,
            dim_names,
            None,
            device,
            use_real_imu=False,
        )
        print(
            f"\n{name} MI TEST: "
            f"r_mean={metrics_mi['r_mean']:.4f}, "
            f"RMSE={metrics_mi['rmse_mean']:.4f}, "
            f"MAE={metrics_mi['mae_mean']:.4f}"
        )
        summary_rows.append(summarize_baseline(
            7, name, metrics_mi,
            "ME EEG+IMU pretraining with normal MHA fusion; MI uses zero IMU; no distillation/prototype alignment",
            evaluation_task="MI_test",
            branch="EEG_zeroIMU_NormalMHA_fusion",
        ))
        axis_rows.extend(long_axis_rows(
            name,
            metrics_mi,
            dim_names,
            evaluation_task="MI_test",
            branch="EEG_zeroIMU_NormalMHA_fusion",
        ))

        metrics_me_final_zero = run_fusion_epoch(
            model,
            me_eeg_test_loader,
            criterion,
            scaler,
            dim_names,
            None,
            device,
            use_real_imu=False,
        )
        print(
            f"\n{name} ME TEST after MI fine-tuning with zero IMU: "
            f"r_mean={metrics_me_final_zero['r_mean']:.4f}, "
            f"RMSE={metrics_me_final_zero['rmse_mean']:.4f}, "
            f"MAE={metrics_me_final_zero['mae_mean']:.4f}"
        )
        summary_rows.append(summarize_baseline(
            7, name, metrics_me_final_zero,
            "Final MI-fine-tuned model evaluated on ME test with zero IMU input; normal MHA fusion",
            evaluation_task="ME_test_after_MI_finetune_zeroIMU",
            branch="EEG_zeroIMU_NormalMHA_fusion",
        ))
        axis_rows.extend(long_axis_rows(
            name,
            metrics_me_final_zero,
            dim_names,
            evaluation_task="ME_test_after_MI_finetune_zeroIMU",
            branch="EEG_zeroIMU_NormalMHA_fusion",
        ))

    save_results_to_excel(summary_rows, axis_rows, stage_rows_all, dim_names)

    elapsed = time.time() - start_time
    print(f"\nAll requested baselines finished in {elapsed/60:.2f} minutes.")


if __name__ == "__main__":
    main()
