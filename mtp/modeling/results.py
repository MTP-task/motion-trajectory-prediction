import numpy as np

from mtp.config import (
    BANDS, BASELINE0_RIDGE_ALPHA, BASELINE0_USE_RIDGE, BATCH_SIZE, CROSS_ATTENTION_HEADS,
    EEG_CROP_TMAX, EEG_CROP_TMIN, EEG_EPOCH_TMIN, EEG_FS, FINETUNE_EPOCHS, KD_LOSS_WEIGHT,
    LINEAR_LAG_WINDOW_SEC, LINEAR_LAGS_SEC, MODEL_EEG_SAMPLES, NORMAL_MHA_HEADS,
    PRETRAIN_EPOCHS, PROTO_LOSS_WEIGHT, RESULT_XLSX_PATH, SEED, TARGET_STREAM, TRAJ_FS,
    subject,
)


# ============================================================
# 11) EXPORT RESULTS
# ============================================================

def summarize_baseline(
    baseline_id,
    baseline_name,
    metrics,
    notes="",
    evaluation_task="MI_test_final",
    branch="final",
):
    row = {
        "subject": subject,
        "baseline_id": baseline_id,
        "baseline": baseline_name,
        "evaluation_task": evaluation_task,
        "branch": branch,
        "target_stream": TARGET_STREAM,
        "test_r_mean": metrics.get("r_mean", np.nan),
        "test_rmse_mean": metrics.get("rmse_mean", np.nan),
        "test_mae_mean": metrics.get("mae_mean", np.nan),
        "test_loss": metrics.get("loss", np.nan),
    }

    # Add per-axis correlation columns directly to the summary sheet.
    # Example columns:
    #   r_shoulder_Vx, r_shoulder_Vy, r_shoulder_Vz,
    #   r_elbow_Vx,    r_elbow_Vy,    r_elbow_Vz,
    #   r_wrist_Vx,    r_wrist_Vy,    r_wrist_Vz
    for k, v in metrics.items():
        if k.startswith("r_") and k != "r_mean":
            row[k] = v

    row["notes"] = notes
    return row


def save_results_to_excel(summary_rows, axis_rows, stage_rows, dim_names):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to save .xlsx results. Please install pandas and openpyxl.") from exc

    summary_df = pd.DataFrame(summary_rows)
    axis_df = pd.DataFrame(axis_rows)
    stages_df = pd.DataFrame(stage_rows)

    config_rows = [
        ("subject", subject),
        ("target_stream", TARGET_STREAM),
        ("dim_names", ", ".join(dim_names)),
        ("eeg_fs", EEG_FS),
        ("traj_fs", TRAJ_FS),
        ("input_epoch", f"{EEG_EPOCH_TMIN} to 7.0 s assumed"),
        ("model_crop", f"{EEG_CROP_TMIN} to {EEG_CROP_TMAX} s"),
        ("model_eeg_samples", MODEL_EEG_SAMPLES),
        ("bands", str(BANDS)),
        ("batch_size", BATCH_SIZE),
        ("pretrain_epochs", PRETRAIN_EPOCHS),
        ("finetune_epochs", FINETUNE_EPOCHS),
        ("seed", SEED),
        ("baseline0_lags_sec", str(LINEAR_LAGS_SEC)),
        ("baseline0_lag_window_sec", LINEAR_LAG_WINDOW_SEC),
        ("baseline0_lag_window_ms", LINEAR_LAG_WINDOW_SEC * 1000),
        ("baseline0_use_ridge", BASELINE0_USE_RIDGE),
        ("baseline0_ridge_alpha", BASELINE0_RIDGE_ALPHA),
        ("cross_attention_heads", CROSS_ATTENTION_HEADS),
        ("normal_mha_heads", NORMAL_MHA_HEADS),
        ("kd_loss_weight", KD_LOSS_WEIGHT),
        ("proto_loss_weight", PROTO_LOSS_WEIGHT),
    ]
    config_df = pd.DataFrame(config_rows, columns=["parameter", "value"])

    with pd.ExcelWriter(RESULT_XLSX_PATH, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        axis_df.to_excel(writer, sheet_name="axis_metrics", index=False)
        stages_df.to_excel(writer, sheet_name="stages", index=False)
        config_df.to_excel(writer, sheet_name="config", index=False)

        # Basic formatting if openpyxl is available through the writer.
        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col:
                    value = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, min(len(value), 50))
                ws.column_dimensions[col_letter].width = max(10, max_len + 2)

    print(f"\nSaved Excel results to:\n{RESULT_XLSX_PATH}")
