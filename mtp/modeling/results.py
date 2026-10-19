"""Collect the result rows of one subject and save them to Excel."""

from dataclasses import fields
import logging

import numpy as np
import pandas as pd

from mtp.config import (
    BANDS,
    EEG_CROP_TMAX,
    EEG_CROP_TMIN,
    EEG_EPOCH_TMAX,
    EEG_EPOCH_TMIN,
    EEG_FS,
    MODEL_EEG_SAMPLES,
    TRAJ_FS,
)

log = logging.getLogger(__name__)


class ResultsLog:
    """Collect the result rows of one subject and save them to Excel.

    Parameters
    ----------
    cfg : Config
        Run settings, written to the "config" sheet.
    dim_names : list of str
        Name of each output dimension.

    Attributes
    ----------
    summary_rows, axis_rows, stage_rows : list of dict
        Rows of the "summary", "axis_metrics" and "stages" sheets.
    """

    def __init__(self, cfg, dim_names):
        self.cfg = cfg
        self.dim_names = dim_names
        self.summary_rows = []
        self.axis_rows = []
        self.stage_rows = []

    def add_evaluation(self, baseline_id, baseline_name, metrics, notes, evaluation_task, branch):
        """Log one evaluation and add it to the "summary" and "axis_metrics" sheets.

        Parameters
        ----------
        baseline_id : int
            Baseline number.
        baseline_name : str
            Baseline name.
        metrics : dict
            Output of `compute_metrics_np`, optionally with a "loss".
        notes : str
            Description of the evaluation.
        evaluation_task, branch : str
            What was evaluated, such as "MI_test", and which part of the model, such as
            "student_EEG_only".
        """
        log.info(
            "RESULT %s | %s | %s: r_mean=%.4f, RMSE=%.4f, MAE=%.4f",
            baseline_name,
            evaluation_task,
            branch,
            metrics["r_mean"],
            metrics["rmse_mean"],
            metrics["mae_mean"],
        )
        self.summary_rows.append(
            {
                "subject": self.cfg.subject,
                "baseline_id": baseline_id,
                "baseline": baseline_name,
                "evaluation_task": evaluation_task,
                "branch": branch,
                "target_stream": self.cfg.target_stream,
                "test_r_mean": metrics["r_mean"],
                "test_rmse_mean": metrics["rmse_mean"],
                "test_mae_mean": metrics["mae_mean"],
                "test_loss": metrics.get("loss", np.nan),
                **{f"r_{name}": metrics[f"r_{name}"] for name in self.dim_names},
                "notes": notes,
            }
        )
        for name in self.dim_names:
            self.axis_rows.append(
                {
                    "subject": self.cfg.subject,
                    "baseline": baseline_name,
                    "evaluation_task": evaluation_task,
                    "branch": branch,
                    "axis": name,
                    "r": metrics[f"r_{name}"],
                    "rmse": metrics[f"rmse_{name}"],
                    "mae": metrics[f"mae_{name}"],
                }
            )

    def config_rows(self):
        """Return the rows of the "config" sheet: recording constants, then all settings."""
        rows = [
            ("dim_names", ", ".join(self.dim_names)),
            ("eeg_fs", EEG_FS),
            ("traj_fs", TRAJ_FS),
            ("input_epoch", f"{EEG_EPOCH_TMIN} to {EEG_EPOCH_TMAX} s"),
            ("model_crop", f"{EEG_CROP_TMIN} to {EEG_CROP_TMAX} s"),
            ("model_eeg_samples", MODEL_EEG_SAMPLES),
            ("bands", str(BANDS)),
        ]
        for f in fields(self.cfg):
            value = getattr(self.cfg, f.name)
            # Excel cells hold numbers, text and booleans, so paths and lists become text.
            rows.append(
                (f.name, value if isinstance(value, (int, float, str, bool)) else str(value))
            )
        return rows

    def save(self, path):
        """Write the four sheets to an Excel file, creating its folder if needed.

        The header row of each sheet is frozen and the columns are sized to fit their
        content, up to 50 characters.
        """
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

            for ws in writer.book.worksheets:
                ws.freeze_panes = "A2"
                for col in ws.columns:
                    max_len = 0
                    col_letter = col[0].column_letter
                    for cell in col:
                        value = "" if cell.value is None else str(cell.value)
                        max_len = max(max_len, min(len(value), 50))
                    ws.column_dimensions[col_letter].width = max(10, max_len + 2)

        log.info("Saved Excel results to %s", path)
