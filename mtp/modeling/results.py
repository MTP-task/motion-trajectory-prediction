"""Collect result rows for one subject and export them to Excel."""
from dataclasses import fields

import numpy as np
import pandas as pd

from mtp.config import BANDS, EEG_CROP_TMAX, EEG_CROP_TMIN, EEG_EPOCH_TMIN, EEG_FS, MODEL_EEG_SAMPLES, TRAJ_FS
from mtp.modeling.metrics import long_axis_rows


def summarize_baseline(
    subject,
    target_stream,
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
        "target_stream": target_stream,
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


class ResultsLog:
    """Result rows for one subject: the summary, axis_metrics and stages sheets."""

    def __init__(self, cfg, dim_names):
        self.cfg = cfg
        self.dim_names = dim_names
        self.summary_rows = []
        self.axis_rows = []
        self.stage_rows = []

    def add_evaluation(self, baseline_id, baseline_name, metrics, notes, evaluation_task, branch):
        print(
            f"\n{baseline_name} | {evaluation_task} | {branch}: "
            f"r_mean={metrics['r_mean']:.4f}, RMSE={metrics['rmse_mean']:.4f}, MAE={metrics['mae_mean']:.4f}"
        )
        self.summary_rows.append(summarize_baseline(
            self.cfg.subject, self.cfg.target_stream, baseline_id, baseline_name, metrics,
            notes, evaluation_task=evaluation_task, branch=branch,
        ))
        self.axis_rows.extend(long_axis_rows(
            self.cfg.subject, baseline_name, metrics, self.dim_names,
            evaluation_task=evaluation_task, branch=branch,
        ))

    def config_rows(self):
        rows = [
            ("dim_names", ", ".join(self.dim_names)),
            ("eeg_fs", EEG_FS),
            ("traj_fs", TRAJ_FS),
            ("input_epoch", f"{EEG_EPOCH_TMIN} to 7.0 s assumed"),
            ("model_crop", f"{EEG_CROP_TMIN} to {EEG_CROP_TMAX} s"),
            ("model_eeg_samples", MODEL_EEG_SAMPLES),
            ("bands", str(BANDS)),
        ]
        for f in fields(self.cfg):
            value = getattr(self.cfg, f.name)
            # Excel cells hold numbers, strings and booleans only.
            rows.append((f.name, value if isinstance(value, (int, float, str, bool)) else str(value)))
        return rows

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        sheets = {
            "summary": pd.DataFrame(self.summary_rows),
            "axis_metrics": pd.DataFrame(self.axis_rows),
            "stages": pd.DataFrame(self.stage_rows),
            "config": pd.DataFrame(self.config_rows(), columns=["parameter", "value"]),
        }

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Freeze the header row and fit column widths.
            for ws in writer.book.worksheets:
                ws.freeze_panes = "A2"
                for col in ws.columns:
                    max_len = 0
                    col_letter = col[0].column_letter
                    for cell in col:
                        value = "" if cell.value is None else str(cell.value)
                        max_len = max(max_len, min(len(value), 50))
                    ws.column_dimensions[col_letter].width = max(10, max_len + 2)

        print(f"\nSaved Excel results to:\n{path}")
