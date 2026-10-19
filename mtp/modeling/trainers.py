"""The training loop shared by every deep baseline.

What differs between baselines is the objective (see `mtp.modeling.objectives`); the
loop, optimizer, learning-rate schedule and early stopping are the same for all of them.
"""

import copy
from dataclasses import dataclass
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from mtp.dataset import denormalize_targets
from mtp.modeling.metrics import compute_metrics_np

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Optimizer and stopping settings for one training stage.

    Attributes
    ----------
    lr : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight decay.
    num_epochs : int
        Maximum number of epochs.
    patience : int
        Stop after this many epochs without a lower validation loss.
    device : str
        Device to train on, such as "cpu" or "cuda".
    grad_clip_norm : float
        Gradients are scaled down when their total norm is larger than this.
    lr_decay_factor : float
        The learning rate is multiplied by this factor when the validation loss stops
        improving.
    lr_decay_patience : int
        Number of epochs without improvement allowed before the learning rate decays.
    """

    lr: float
    weight_decay: float
    num_epochs: int
    patience: int
    device: str = "cpu"
    grad_clip_norm: float = 5.0
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 4


@dataclass
class EpochResult:
    """Losses and metrics from one pass over a loader.

    Attributes
    ----------
    losses : dict of float
        Average per trial of the total loss ("loss") and of any extra loss terms the
        objective reports, such as "proto_loss".
    metrics : dict of dict
        Output of `compute_metrics_np` for each prediction, keyed by its name: "y", or
        "teacher" and "student".
    """

    losses: dict
    metrics: dict

    def flat(self, pred: str = "y") -> dict:
        """Return the metrics of one prediction and the losses as a single dict."""
        return {**self.metrics[pred], **self.losses}


def trainable_parameters(model):
    """Return the parameters of `model` that are not frozen."""
    return [p for p in model.parameters() if p.requires_grad]


def run_epoch(
    model, loader, objective, scaler, dim_names, optimizer=None, device="cpu", grad_clip_norm=5.0
) -> EpochResult:
    """Make one pass over `loader`, training if an optimizer is given.

    Parameters
    ----------
    model : nn.Module
        Model to train or evaluate.
    loader : DataLoader
        Batches from `mtp.dataset.make_loader`.
    objective : callable
        One of the objectives in `mtp.modeling.objectives`.
    scaler : dict
        Target scaler, used to score predictions in their original units.
    dim_names : list of str
        Name of each output dimension.
    optimizer : torch.optim.Optimizer, optional
        If given, the model is trained; otherwise it is only evaluated.
    device : str
        Device to run on.
    grad_clip_norm : float
        Limit for the total gradient norm when training.

    Returns
    -------
    EpochResult
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    totals = {}
    y_true_all, preds_all = [], {}

    for batch in loader:
        batch = [t.to(device) for t in batch]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            out = objective(model, batch)
            if is_train:
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(model), max_norm=grad_clip_norm
                )
                optimizer.step()

        n = batch[0].shape[0]
        for name, value in {"loss": out.loss, **out.parts}.items():
            totals[name] = totals.get(name, 0.0) + value.item() * n
        y_true_all.append(out.y.detach().cpu().numpy())
        for name, pred in out.preds.items():
            preds_all.setdefault(name, []).append(pred.detach().cpu().numpy())

    y_true = denormalize_targets(np.concatenate(y_true_all, axis=0), scaler)
    metrics = {
        name: compute_metrics_np(
            y_true, denormalize_targets(np.concatenate(chunks, axis=0), scaler), dim_names
        )
        for name, chunks in preds_all.items()
    }
    losses = {name: total / len(loader.dataset) for name, total in totals.items()}
    return EpochResult(losses=losses, metrics=metrics)


def fit(
    stage_name, model, train_loader, val_loader, objective, scaler, dim_names, cfg: TrainConfig
):
    """Train a model and keep the weights with the lowest validation loss.

    The trainable parameters are optimized with AdamW. The learning rate decays when the
    validation loss stops improving, and training stops early after ``cfg.patience``
    epochs without improvement. At the end, the best weights are loaded back into
    `model`.

    Parameters
    ----------
    stage_name : str
        Name of the stage in the logs and the "stages" sheet.
    model : nn.Module
        Model to train, in place.
    train_loader, val_loader : DataLoader
        Training and validation batches.
    objective : callable
        One of the objectives in `mtp.modeling.objectives`.
    scaler : dict
        Target scaler, used to score predictions in their original units.
    dim_names : list of str
        Name of each output dimension.
    cfg : TrainConfig
        Optimizer and stopping settings.

    Returns
    -------
    list of dict
        One row per epoch for the "stages" sheet.
    """
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(
        trainable_parameters(model), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_decay_factor, patience=cfg.lr_decay_patience
    )

    best_state = None
    best_val_loss = math.inf
    wait = 0
    stage_rows = []

    log.info(
        "===== %s ===== (%d trainable parameters)",
        stage_name,
        sum(p.numel() for p in trainable_parameters(model)),
    )

    for epoch in range(1, cfg.num_epochs + 1):
        train = run_epoch(
            model,
            train_loader,
            objective,
            scaler,
            dim_names,
            optimizer,
            cfg.device,
            cfg.grad_clip_norm,
        )
        val = run_epoch(model, val_loader, objective, scaler, dim_names, None, cfg.device)
        scheduler.step(val.losses["loss"])

        details = objective.stage_row(train, val)
        log.info(
            "%s | Epoch %03d | train_loss=%.4f, val_loss=%.4f | %s",
            stage_name,
            epoch,
            train.losses["loss"],
            val.losses["loss"],
            ", ".join(f"{k}={v:.4f}" for k, v in details.items()),
        )
        stage_rows.append(
            {
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": train.losses["loss"],
                "val_loss": val.losses["loss"],
                **details,
            }
        )

        if val.losses["loss"] < best_val_loss:
            best_val_loss = val.losses["loss"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                log.info("%s: early stopping after epoch %d.", stage_name, epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return stage_rows


@torch.no_grad()
def compute_me_prototypes(model, loader, device, num_targets: int = 4):
    """Compute one prototype per target from ME data (Baseline 5).

    A prototype is the student's EEG features averaged over time and over all ME
    training trials of that target, scaled to unit length.

    Parameters
    ----------
    model : FullPrivilegedModel
        Model after ME pretraining.
    loader : DataLoader
        ME training batches with IMU.
    device : str
        Device to run on.
    num_targets : int
        Number of targets.

    Returns
    -------
    Tensor, shape (num_targets, latent_dim)

    Raises
    ------
    ValueError
        If a target has no trials.
    """
    model.eval()
    z_list, id_list = [], []

    for eeg, _imu, _y, target_id, subject_id in loader:
        eeg, subject_id = eeg.to(device), subject_id.to(device)
        _y_pred, h = model.forward_student(eeg, subject_id, return_latent=True)
        z = h.mean(dim=-1)
        z_list.append(z.detach().cpu())
        id_list.append(target_id.detach().cpu())

    z_all = torch.cat(z_list, dim=0)
    id_all = torch.cat(id_list, dim=0)

    latent_dim = z_all.shape[1]
    prototypes = torch.zeros(num_targets, latent_dim)

    for k in range(num_targets):
        mask = id_all == k
        if mask.sum() == 0:
            raise ValueError(f"No samples found for target id {k}.")
        prototypes[k] = z_all[mask].mean(dim=0)

    return F.normalize(prototypes, dim=-1)
