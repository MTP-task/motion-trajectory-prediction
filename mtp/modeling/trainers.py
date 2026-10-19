import math
import copy
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mtp.config import (
    KD_LOSS_WEIGHT, PROTO_LOSS_WEIGHT, STUDENT_SUP_LOSS_WEIGHT, TARGET_LEN,
    TEACHER_LOSS_WEIGHT,
)
from mtp.dataset import denormalize_targets
from mtp.modeling.models import set_distill_student_finetune_mode, set_full_student_finetune_mode
from mtp.modeling.metrics import SequenceRegressionLoss, compute_metrics_np, proto_alignment_loss


# ============================================================
# 8) TRAINING HELPERS
# ============================================================

@dataclass
class TrainConfig:
    lr: float
    weight_decay: float
    num_epochs: int
    patience: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_eeg_epoch(model, loader, criterion, scaler, dim_names, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true_all, y_pred_all = [], []

    for eeg, y, _target_id, _subject_id in loader:
        eeg = eeg.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            y_pred = model(eeg)
            loss = criterion(y_pred, y)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += loss.item() * eeg.shape[0]
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(y_pred.detach().cpu().numpy())

    y_true_all = denormalize_targets(np.concatenate(y_true_all, axis=0), scaler)
    y_pred_all = denormalize_targets(np.concatenate(y_pred_all, axis=0), scaler)

    metrics = compute_metrics_np(y_true_all, y_pred_all, dim_names)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_eeg_model(stage_name, model, train_loader, val_loader, scaler, dim_names, cfg):
    model.to(cfg.device)
    criterion = SequenceRegressionLoss(corr_weight=0.10)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_state = None
    best_val_loss = math.inf
    wait = 0
    stage_rows = []

    print(f"\n===== {stage_name} =====")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    for epoch in range(1, cfg.num_epochs + 1):
        train_m = run_eeg_epoch(model, train_loader, criterion, scaler, dim_names, optimizer, cfg.device)
        val_m = run_eeg_epoch(model, val_loader, criterion, scaler, dim_names, None, cfg.device)
        scheduler.step(val_m["loss"])

        print(
            f"{stage_name} | Epoch {epoch:03d} | "
            f"train_loss={train_m['loss']:.4f}, val_loss={val_m['loss']:.4f} | "
            f"train_r={train_m['r_mean']:.4f}, val_r={val_m['r_mean']:.4f}"
        )

        stage_rows.append({
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "train_r_mean": train_m["r_mean"],
            "val_r_mean": val_m["r_mean"],
            "val_rmse_mean": val_m["rmse_mean"],
            "val_mae_mean": val_m["mae_mean"],
        })

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"{stage_name}: early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, stage_rows


def run_fusion_epoch(model, loader, criterion, scaler, dim_names, optimizer=None, device="cpu", use_real_imu=True):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true_all, y_pred_all = [], []

    for batch in loader:
        if use_real_imu:
            eeg, imu, y, _target_id, _subject_id = batch
            imu = imu.to(device)
        else:
            eeg, y, _target_id, _subject_id = batch
            imu = torch.zeros((eeg.shape[0], TARGET_LEN, 6), dtype=eeg.dtype)

        eeg = eeg.to(device)
        imu = imu.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            y_pred = model(eeg, imu)
            loss = criterion(y_pred, y)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += loss.item() * eeg.shape[0]
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(y_pred.detach().cpu().numpy())

    y_true_all = denormalize_targets(np.concatenate(y_true_all, axis=0), scaler)
    y_pred_all = denormalize_targets(np.concatenate(y_pred_all, axis=0), scaler)

    metrics = compute_metrics_np(y_true_all, y_pred_all, dim_names)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_fusion_model(stage_name, model, train_loader, val_loader, scaler, dim_names, cfg, use_real_imu):
    model.to(cfg.device)
    criterion = SequenceRegressionLoss(corr_weight=0.10)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_state = None
    best_val_loss = math.inf
    wait = 0
    stage_rows = []

    print(f"\n===== {stage_name} =====")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    for epoch in range(1, cfg.num_epochs + 1):
        train_m = run_fusion_epoch(model, train_loader, criterion, scaler, dim_names, optimizer, cfg.device, use_real_imu)
        val_m = run_fusion_epoch(model, val_loader, criterion, scaler, dim_names, None, cfg.device, use_real_imu)
        scheduler.step(val_m["loss"])

        print(
            f"{stage_name} | Epoch {epoch:03d} | "
            f"train_loss={train_m['loss']:.4f}, val_loss={val_m['loss']:.4f} | "
            f"train_r={train_m['r_mean']:.4f}, val_r={val_m['r_mean']:.4f}"
        )

        stage_rows.append({
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "train_r_mean": train_m["r_mean"],
            "val_r_mean": val_m["r_mean"],
            "val_rmse_mean": val_m["rmse_mean"],
            "val_mae_mean": val_m["mae_mean"],
        })

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"{stage_name}: early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, stage_rows


def run_distill_pretrain_epoch(model, loader, reg_criterion, kd_criterion, scaler, dim_names, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true_all, y_teacher_all, y_student_all = [], [], []

    for eeg, imu, y, _target_id, _subject_id in loader:
        eeg, imu, y = eeg.to(device), imu.to(device), y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            y_teacher, y_student = model.forward_teacher_student(eeg, imu)
            teacher_loss = reg_criterion(y_teacher, y)
            student_loss = reg_criterion(y_student, y)
            kd_loss = kd_criterion(y_student, y_teacher.detach())

            loss = (
                TEACHER_LOSS_WEIGHT * teacher_loss
                + STUDENT_SUP_LOSS_WEIGHT * student_loss
                + KD_LOSS_WEIGHT * kd_loss
            )

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += loss.item() * eeg.shape[0]
        y_true_all.append(y.detach().cpu().numpy())
        y_teacher_all.append(y_teacher.detach().cpu().numpy())
        y_student_all.append(y_student.detach().cpu().numpy())

    y_true_orig = denormalize_targets(np.concatenate(y_true_all, axis=0), scaler)
    y_teacher_orig = denormalize_targets(np.concatenate(y_teacher_all, axis=0), scaler)
    y_student_orig = denormalize_targets(np.concatenate(y_student_all, axis=0), scaler)

    return {
        "loss": total_loss / len(loader.dataset),
        "teacher": compute_metrics_np(y_true_orig, y_teacher_orig, dim_names),
        "student": compute_metrics_np(y_true_orig, y_student_orig, dim_names),
    }


def train_distill_pretrain(stage_name, model, train_loader, val_loader, scaler, dim_names, cfg):
    model.to(cfg.device)
    reg_criterion = SequenceRegressionLoss(corr_weight=0.10)
    kd_criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_state = None
    best_val_loss = math.inf
    wait = 0
    stage_rows = []

    print(f"\n===== {stage_name} =====")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    for epoch in range(1, cfg.num_epochs + 1):
        train_m = run_distill_pretrain_epoch(model, train_loader, reg_criterion, kd_criterion, scaler, dim_names, optimizer, cfg.device)
        val_m = run_distill_pretrain_epoch(model, val_loader, reg_criterion, kd_criterion, scaler, dim_names, None, cfg.device)
        scheduler.step(val_m["loss"])

        print(
            f"{stage_name} | Epoch {epoch:03d} | "
            f"train_loss={train_m['loss']:.4f}, val_loss={val_m['loss']:.4f} | "
            f"teacher_val_r={val_m['teacher']['r_mean']:.4f}, student_val_r={val_m['student']['r_mean']:.4f}"
        )

        stage_rows.append({
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "teacher_val_r_mean": val_m["teacher"]["r_mean"],
            "student_val_r_mean": val_m["student"]["r_mean"],
            "student_val_rmse_mean": val_m["student"]["rmse_mean"],
            "student_val_mae_mean": val_m["student"]["mae_mean"],
        })

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"{stage_name}: early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, stage_rows


def run_distill_student_epoch(model, loader, criterion, scaler, dim_names, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true_all, y_pred_all = [], []

    for eeg, y, _target_id, _subject_id in loader:
        eeg, y = eeg.to(device), y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            y_pred = model.forward_student(eeg)
            loss = criterion(y_pred, y)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=5.0)
                optimizer.step()

        total_loss += loss.item() * eeg.shape[0]
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(y_pred.detach().cpu().numpy())

    y_true_orig = denormalize_targets(np.concatenate(y_true_all, axis=0), scaler)
    y_pred_orig = denormalize_targets(np.concatenate(y_pred_all, axis=0), scaler)
    metrics = compute_metrics_np(y_true_orig, y_pred_orig, dim_names)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_distill_student(stage_name, model, train_loader, val_loader, scaler, dim_names, cfg):
    model.to(cfg.device)
    set_distill_student_finetune_mode(model)
    criterion = SequenceRegressionLoss(corr_weight=0.10)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_state = None
    best_val_loss = math.inf
    wait = 0
    stage_rows = []

    print(f"\n===== {stage_name} =====")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    for epoch in range(1, cfg.num_epochs + 1):
        train_m = run_distill_student_epoch(model, train_loader, criterion, scaler, dim_names, optimizer, cfg.device)
        val_m = run_distill_student_epoch(model, val_loader, criterion, scaler, dim_names, None, cfg.device)
        scheduler.step(val_m["loss"])

        print(
            f"{stage_name} | Epoch {epoch:03d} | "
            f"train_loss={train_m['loss']:.4f}, val_loss={val_m['loss']:.4f} | "
            f"train_r={train_m['r_mean']:.4f}, val_r={val_m['r_mean']:.4f}"
        )

        stage_rows.append({
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "train_r_mean": train_m["r_mean"],
            "val_r_mean": val_m["r_mean"],
            "val_rmse_mean": val_m["rmse_mean"],
            "val_mae_mean": val_m["mae_mean"],
        })

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"{stage_name}: early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, stage_rows


def run_full_pretrain_epoch(model, loader, reg_criterion, kd_criterion, scaler, dim_names, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true_all, y_teacher_all, y_student_all = [], [], []

    for eeg, imu, y, _target_id, subject_id in loader:
        eeg, imu, y, subject_id = eeg.to(device), imu.to(device), y.to(device), subject_id.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            y_teacher, y_student, _h_eeg, _h_teacher = model.forward_teacher_student(eeg, imu, subject_id)
            teacher_loss = reg_criterion(y_teacher, y)
            student_loss = reg_criterion(y_student, y)
            kd_loss = kd_criterion(y_student, y_teacher.detach())
            loss = (
                TEACHER_LOSS_WEIGHT * teacher_loss
                + STUDENT_SUP_LOSS_WEIGHT * student_loss
                + KD_LOSS_WEIGHT * kd_loss
            )
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += loss.item() * eeg.shape[0]
        y_true_all.append(y.detach().cpu().numpy())
        y_teacher_all.append(y_teacher.detach().cpu().numpy())
        y_student_all.append(y_student.detach().cpu().numpy())

    y_true_orig = denormalize_targets(np.concatenate(y_true_all, axis=0), scaler)
    y_teacher_orig = denormalize_targets(np.concatenate(y_teacher_all, axis=0), scaler)
    y_student_orig = denormalize_targets(np.concatenate(y_student_all, axis=0), scaler)

    return {
        "loss": total_loss / len(loader.dataset),
        "teacher": compute_metrics_np(y_true_orig, y_teacher_orig, dim_names),
        "student": compute_metrics_np(y_true_orig, y_student_orig, dim_names),
    }


def train_full_pretrain(stage_name, model, train_loader, val_loader, scaler, dim_names, cfg):
    model.to(cfg.device)
    reg_criterion = SequenceRegressionLoss(corr_weight=0.10)
    kd_criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_state = None
    best_val_loss = math.inf
    wait = 0
    stage_rows = []

    print(f"\n===== {stage_name} =====")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    for epoch in range(1, cfg.num_epochs + 1):
        train_m = run_full_pretrain_epoch(model, train_loader, reg_criterion, kd_criterion, scaler, dim_names, optimizer, cfg.device)
        val_m = run_full_pretrain_epoch(model, val_loader, reg_criterion, kd_criterion, scaler, dim_names, None, cfg.device)
        scheduler.step(val_m["loss"])

        print(
            f"{stage_name} | Epoch {epoch:03d} | "
            f"train_loss={train_m['loss']:.4f}, val_loss={val_m['loss']:.4f} | "
            f"teacher_val_r={val_m['teacher']['r_mean']:.4f}, student_val_r={val_m['student']['r_mean']:.4f}"
        )

        stage_rows.append({
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "teacher_val_r_mean": val_m["teacher"]["r_mean"],
            "student_val_r_mean": val_m["student"]["r_mean"],
            "student_val_rmse_mean": val_m["student"]["rmse_mean"],
            "student_val_mae_mean": val_m["student"]["mae_mean"],
        })

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"{stage_name}: early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, stage_rows


@torch.no_grad()
def compute_me_prototypes(model, loader, device, num_targets: int = 4):
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


def run_full_student_epoch(model, loader, criterion, scaler, dim_names, prototypes=None, proto_weight=0.0, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    if prototypes is not None:
        prototypes = prototypes.to(device)

    total_loss = 0.0
    total_reg_loss = 0.0
    total_proto_loss = 0.0
    y_true_all, y_pred_all = [], []

    for eeg, y, target_id, subject_id in loader:
        eeg, y, target_id, subject_id = eeg.to(device), y.to(device), target_id.to(device), subject_id.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            y_pred, h_eeg = model.forward_student(eeg, subject_id, return_latent=True)
            reg_loss = criterion(y_pred, y)

            if prototypes is not None and proto_weight > 0:
                p_loss = proto_alignment_loss(h_eeg, target_id, prototypes)
            else:
                p_loss = torch.tensor(0.0, device=device)

            loss = reg_loss + proto_weight * p_loss

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=5.0)
                optimizer.step()

        total_loss += loss.item() * eeg.shape[0]
        total_reg_loss += reg_loss.item() * eeg.shape[0]
        total_proto_loss += p_loss.item() * eeg.shape[0]
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(y_pred.detach().cpu().numpy())

    y_true_orig = denormalize_targets(np.concatenate(y_true_all, axis=0), scaler)
    y_pred_orig = denormalize_targets(np.concatenate(y_pred_all, axis=0), scaler)

    metrics = compute_metrics_np(y_true_orig, y_pred_orig, dim_names)
    metrics["loss"] = total_loss / len(loader.dataset)
    metrics["reg_loss"] = total_reg_loss / len(loader.dataset)
    metrics["proto_loss"] = total_proto_loss / len(loader.dataset)
    return metrics


def train_full_student(stage_name, model, train_loader, val_loader, scaler, dim_names, prototypes, cfg):
    model.to(cfg.device)
    set_full_student_finetune_mode(model)
    criterion = SequenceRegressionLoss(corr_weight=0.10)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    proto_weight = PROTO_LOSS_WEIGHT if prototypes is not None else 0.0

    best_state = None
    best_val_loss = math.inf
    wait = 0
    stage_rows = []

    print(f"\n===== {stage_name} =====")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Prototype loss weight:", proto_weight)

    for epoch in range(1, cfg.num_epochs + 1):
        train_m = run_full_student_epoch(model, train_loader, criterion, scaler, dim_names, prototypes, proto_weight, optimizer, cfg.device)
        val_m = run_full_student_epoch(model, val_loader, criterion, scaler, dim_names, prototypes, proto_weight, None, cfg.device)
        scheduler.step(val_m["loss"])

        print(
            f"{stage_name} | Epoch {epoch:03d} | "
            f"train_loss={train_m['loss']:.4f}, val_loss={val_m['loss']:.4f} | "
            f"train_r={train_m['r_mean']:.4f}, val_r={val_m['r_mean']:.4f}, val_proto={val_m['proto_loss']:.4f}"
        )

        stage_rows.append({
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "train_r_mean": train_m["r_mean"],
            "val_r_mean": val_m["r_mean"],
            "val_rmse_mean": val_m["rmse_mean"],
            "val_mae_mean": val_m["mae_mean"],
            "val_proto_loss": val_m["proto_loss"],
        })

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"{stage_name}: early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, stage_rows
