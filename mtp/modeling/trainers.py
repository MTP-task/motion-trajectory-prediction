"""The training loop shared by every deep baseline (objectives.py holds what differs)."""
import copy
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from mtp.dataset import denormalize_targets
from mtp.modeling.metrics import compute_metrics_np


@dataclass
class TrainConfig:
    lr: float
    weight_decay: float
    num_epochs: int
    patience: int
    device: str = "cpu"


@dataclass
class EpochResult:
    # "loss" plus any extra loss terms from the objective, averaged per sample
    losses: dict
    # prediction name ("y", "teacher", "student") -> metrics from compute_metrics_np
    metrics: dict

    def flat(self, pred: str = "y") -> dict:
        """Metrics of one prediction merged with the losses, as used for result rows."""
        return {**self.metrics[pred], **self.losses}


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def run_epoch(model, loader, objective, scaler, dim_names, optimizer=None, device="cpu") -> EpochResult:
    """One pass over `loader`: trains when an optimizer is given, otherwise evaluates."""
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
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm=5.0)
                optimizer.step()

        n = batch[0].shape[0]
        for name, value in {"loss": out.loss, **out.parts}.items():
            totals[name] = totals.get(name, 0.0) + value.item() * n
        y_true_all.append(out.y.detach().cpu().numpy())
        for name, pred in out.preds.items():
            preds_all.setdefault(name, []).append(pred.detach().cpu().numpy())

    y_true = denormalize_targets(np.concatenate(y_true_all, axis=0), scaler)
    metrics = {
        name: compute_metrics_np(y_true, denormalize_targets(np.concatenate(chunks, axis=0), scaler), dim_names)
        for name, chunks in preds_all.items()
    }
    losses = {name: total / len(loader.dataset) for name, total in totals.items()}
    return EpochResult(losses=losses, metrics=metrics)


def fit(stage_name, model, train_loader, val_loader, objective, scaler, dim_names, cfg: TrainConfig):
    """
    Train the model's trainable parameters with AdamW, halve the learning rate when the
    validation loss plateaus, stop early after `cfg.patience` epochs without improvement,
    and load the weights with the best validation loss back into `model`.

    Returns one row per epoch for the "stages" sheet.
    """
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_state = None
    best_val_loss = math.inf
    wait = 0
    stage_rows = []

    print(f"\n===== {stage_name} =====")
    print("Trainable parameters:", sum(p.numel() for p in trainable_parameters(model)))

    for epoch in range(1, cfg.num_epochs + 1):
        train = run_epoch(model, train_loader, objective, scaler, dim_names, optimizer, cfg.device)
        val = run_epoch(model, val_loader, objective, scaler, dim_names, None, cfg.device)
        scheduler.step(val.losses["loss"])

        details = objective.stage_row(train, val)
        print(
            f"{stage_name} | Epoch {epoch:03d} | "
            f"train_loss={train.losses['loss']:.4f}, val_loss={val.losses['loss']:.4f} | "
            + ", ".join(f"{k}={v:.4f}" for k, v in details.items())
        )
        stage_rows.append({
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train.losses["loss"],
            "val_loss": val.losses["loss"],
            **details,
        })

        if val.losses["loss"] < best_val_loss:
            best_val_loss = val.losses["loss"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"{stage_name}: early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return stage_rows


@torch.no_grad()
def compute_me_prototypes(model, loader, device, num_targets: int = 4):
    """Mean normalized student EEG latent per target on ME data (Baseline 5)."""
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
