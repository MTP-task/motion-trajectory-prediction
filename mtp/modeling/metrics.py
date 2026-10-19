"""Training losses and evaluation metrics."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def sequence_corrcoef(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8):
    yp = y_pred.reshape(-1, y_pred.shape[-1])
    yt = y_true.reshape(-1, y_true.shape[-1])
    yp = yp - yp.mean(dim=0, keepdim=True)
    yt = yt - yt.mean(dim=0, keepdim=True)
    num = (yp * yt).sum(dim=0)
    den = torch.sqrt((yp**2).sum(dim=0) * (yt**2).sum(dim=0) + eps)
    return (num / (den + eps)).mean()


class SequenceRegressionLoss(nn.Module):
    """Huber loss plus corr_weight * (1 - mean Pearson r over the output dimensions)."""

    def __init__(self, corr_weight: float = 0.10, huber_beta: float = 0.5):
        super().__init__()
        self.corr_weight = corr_weight
        self.huber = nn.SmoothL1Loss(beta=huber_beta)

    def forward(self, y_pred, y_true):
        huber = self.huber(y_pred, y_true)
        corr = sequence_corrcoef(y_pred, y_true)
        return huber + self.corr_weight * (1.0 - corr)


def proto_alignment_loss(h_eeg, target_id, prototypes):
    z = h_eeg.mean(dim=-1)
    z = F.normalize(z, dim=-1)
    p = prototypes[target_id]
    p = F.normalize(p, dim=-1)
    return (1.0 - (z * p).sum(dim=-1)).mean()


def compute_metrics_np(y_true: np.ndarray, y_pred: np.ndarray, dim_names: list[str]):
    yt = y_true.reshape(-1, y_true.shape[-1])
    yp = y_pred.reshape(-1, y_pred.shape[-1])

    rmse = np.sqrt(np.mean((yt - yp) ** 2, axis=0))
    mae = np.mean(np.abs(yt - yp), axis=0)

    r_list = []
    for d in range(yt.shape[1]):
        if np.std(yt[:, d]) < 1e-9 or np.std(yp[:, d]) < 1e-9:
            r = 0.0
        else:
            r = float(np.corrcoef(yt[:, d], yp[:, d])[0, 1])
        r_list.append(r)

    metrics = {
        "r_mean": float(np.mean(r_list)),
        "rmse_mean": float(np.mean(rmse)),
        "mae_mean": float(np.mean(mae)),
    }

    for name, r, e, a in zip(dim_names, r_list, rmse, mae):
        metrics[f"r_{name}"] = float(r)
        metrics[f"rmse_{name}"] = float(e)
        metrics[f"mae_{name}"] = float(a)

    return metrics


def long_axis_rows(
    subject, baseline_name, metrics, dim_names, evaluation_task="MI_test_final", branch="final"
):
    """One row per output dimension (r, RMSE, MAE) for the axis_metrics sheet."""
    rows = []
    for name in dim_names:
        rows.append(
            {
                "subject": subject,
                "baseline": baseline_name,
                "evaluation_task": evaluation_task,
                "branch": branch,
                "axis": name,
                "r": metrics.get(f"r_{name}", np.nan),
                "rmse": metrics.get(f"rmse_{name}", np.nan),
                "mae": metrics.get(f"mae_{name}", np.nan),
            }
        )
    return rows
