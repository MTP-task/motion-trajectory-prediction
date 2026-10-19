"""Training losses and evaluation metrics."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def sequence_corrcoef(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8):
    """Return the Pearson correlation per output dimension, averaged over the dimensions.

    All trials and time steps are pooled before the correlation is computed.

    Parameters
    ----------
    y_pred, y_true : Tensor, shape (batch, n_times, n_outputs)
        Predicted and true velocities.
    eps : float
        Keeps the division stable when a signal is constant.

    Returns
    -------
    Tensor
        A scalar.
    """
    yp = y_pred.reshape(-1, y_pred.shape[-1])
    yt = y_true.reshape(-1, y_true.shape[-1])
    yp = yp - yp.mean(dim=0, keepdim=True)
    yt = yt - yt.mean(dim=0, keepdim=True)
    num = (yp * yt).sum(dim=0)
    den = torch.sqrt((yp**2).sum(dim=0) * (yt**2).sum(dim=0) + eps)
    return (num / (den + eps)).mean()


class SequenceRegressionLoss(nn.Module):
    """Regression loss used by all deep baselines.

    The loss is the Huber loss plus ``corr_weight * (1 - r)``, where r is the mean
    correlation from `sequence_corrcoef`. The correlation term rewards getting the shape
    of each velocity curve right, not only its values.

    Parameters
    ----------
    corr_weight : float
        Weight of the correlation term.
    huber_beta : float
        Threshold of the Huber (SmoothL1) loss.
    """

    def __init__(self, corr_weight: float = 0.10, huber_beta: float = 0.5):
        super().__init__()
        self.corr_weight = corr_weight
        self.huber = nn.SmoothL1Loss(beta=huber_beta)

    def forward(self, y_pred, y_true):
        """Compute the loss for predictions of shape (batch, n_times, n_outputs)."""
        huber = self.huber(y_pred, y_true)
        corr = sequence_corrcoef(y_pred, y_true)
        return huber + self.corr_weight * (1.0 - corr)


def proto_alignment_loss(h_eeg, target_id, prototypes):
    """Return how far each trial's EEG features are from its target's prototype.

    Parameters
    ----------
    h_eeg : Tensor, shape (batch, latent_dim, n_times)
        EEG features of the student model.
    target_id : Tensor of int, shape (batch,)
        Target index of each trial.
    prototypes : Tensor, shape (n_targets, latent_dim)
        One prototype per target, from `compute_me_prototypes`.

    Returns
    -------
    Tensor
        A scalar: 1 minus the cosine similarity between the time-averaged features and
        the prototype, averaged over the batch.
    """
    z = h_eeg.mean(dim=-1)
    z = F.normalize(z, dim=-1)
    p = prototypes[target_id]
    p = F.normalize(p, dim=-1)
    return (1.0 - (z * p).sum(dim=-1)).mean()


def compute_metrics_np(y_true: np.ndarray, y_pred: np.ndarray, dim_names: list[str]):
    """Score predicted velocities against the true ones.

    For each output dimension, all trials and time steps are pooled before computing the
    Pearson correlation r, the RMSE and the MAE. A dimension that is constant in either
    array gets r = 0.

    Parameters
    ----------
    y_true, y_pred : ndarray, shape (n_trials, n_times, n_outputs)
        True and predicted velocities, in their original units.
    dim_names : list of str
        Name of each output dimension, such as "wrist_Vx".

    Returns
    -------
    dict of float
        ``r_mean``, ``rmse_mean`` and ``mae_mean`` (averaged over the dimensions), and
        ``r_<name>``, ``rmse_<name>`` and ``mae_<name>`` for each dimension.
    """
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


def long_axis_rows(subject, baseline_name, metrics, dim_names, evaluation_task, branch):
    """Return one row per output dimension for the "axis_metrics" sheet.

    Parameters
    ----------
    subject : str
        Subject ID.
    baseline_name : str
        Name of the baseline, such as "Baseline 1: MI-only EEG".
    metrics : dict
        Output of `compute_metrics_np`.
    dim_names : list of str
        Output dimensions to report.
    evaluation_task, branch : str
        Labels of the evaluation, as in the "summary" sheet.

    Returns
    -------
    list of dict
        Rows with the r, RMSE and MAE of each dimension.
    """
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
