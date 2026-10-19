"""Per-batch training objectives.

An objective takes a model and one batch, and returns the loss to optimise plus the
predictions to score. The training loop in trainers.py is the same for every baseline;
only the objective changes.

Batches come from mtp.dataset.make_loader:
    EEG-only loaders: (eeg, y, target_id, subject_id)
    EEG+IMU loaders:  (eeg, imu, y, target_id, subject_id)
"""

from dataclasses import dataclass, field

import torch
from torch import nn

from mtp.config import IMU_DIM, TARGET_LEN
from mtp.modeling.metrics import proto_alignment_loss


@dataclass
class StepOutput:
    loss: torch.Tensor
    y: torch.Tensor
    # Predicted trajectories to score, e.g. {"y": ...} or {"teacher": ..., "student": ...}
    preds: dict
    # Extra loss terms to average over the epoch, e.g. {"proto_loss": ...}
    parts: dict = field(default_factory=dict)


# ---- Forward passes: (model, batch) -> (prediction, target) ------------------


def eeg_only(model, batch):
    eeg, y, _target_id, _subject_id = batch
    return model(eeg), y


def fusion_real_imu(model, batch):
    eeg, imu, y, _target_id, _subject_id = batch
    return model(eeg, imu), y


def fusion_zero_imu(model, batch):
    """MI has no IMU recording, so fusion models get an all-zero IMU input."""
    eeg, y, _target_id, _subject_id = batch
    imu = torch.zeros((eeg.shape[0], TARGET_LEN, IMU_DIM), dtype=eeg.dtype, device=eeg.device)
    return model(eeg, imu), y


def distill_student(model, batch):
    eeg, y, _target_id, _subject_id = batch
    return model.forward_student(eeg), y


# ---- Objectives -------------------------------------------------------------


def _single_prediction_row(train, val):
    return {
        "train_r_mean": train.metrics["y"]["r_mean"],
        "val_r_mean": val.metrics["y"]["r_mean"],
        "val_rmse_mean": val.metrics["y"]["rmse_mean"],
        "val_mae_mean": val.metrics["y"]["mae_mean"],
    }


class Supervised:
    """Regress one prediction onto the target with the given regression loss."""

    def __init__(self, forward, criterion):
        self.forward = forward
        self.criterion = criterion

    def __call__(self, model, batch):
        y_pred, y = self.forward(model, batch)
        return StepOutput(loss=self.criterion(y_pred, y), y=y, preds={"y": y_pred})

    def stage_row(self, train, val):
        """Columns for the per-epoch "stages" sheet."""
        return _single_prediction_row(train, val)


class TeacherStudent:
    """
    ME pretraining with an EEG+IMU teacher and an EEG-only student (Baselines 4 and 5):
        loss = w_teacher * reg(teacher) + w_student * reg(student)
               + w_kd * SmoothL1(student, teacher.detach())
    `criterion` is the regression loss reg. Set use_subject_id for models whose
    forward_teacher_student takes subject IDs.
    """

    def __init__(
        self,
        criterion,
        teacher_weight,
        student_weight,
        kd_weight,
        kd_beta=0.5,
        use_subject_id=False,
    ):
        self.teacher_weight = teacher_weight
        self.student_weight = student_weight
        self.kd_weight = kd_weight
        self.use_subject_id = use_subject_id
        self.reg_criterion = criterion
        self.kd_criterion = nn.SmoothL1Loss(beta=kd_beta)

    def __call__(self, model, batch):
        eeg, imu, y, _target_id, subject_id = batch
        if self.use_subject_id:
            y_teacher, y_student, _h_eeg, _h_teacher = model.forward_teacher_student(
                eeg, imu, subject_id
            )
        else:
            y_teacher, y_student = model.forward_teacher_student(eeg, imu)

        teacher_loss = self.reg_criterion(y_teacher, y)
        student_loss = self.reg_criterion(y_student, y)
        kd_loss = self.kd_criterion(y_student, y_teacher.detach())
        loss = (
            self.teacher_weight * teacher_loss
            + self.student_weight * student_loss
            + self.kd_weight * kd_loss
        )
        return StepOutput(loss=loss, y=y, preds={"teacher": y_teacher, "student": y_student})

    def stage_row(self, train, val):
        return {
            "teacher_val_r_mean": val.metrics["teacher"]["r_mean"],
            "student_val_r_mean": val.metrics["student"]["r_mean"],
            "student_val_rmse_mean": val.metrics["student"]["rmse_mean"],
            "student_val_mae_mean": val.metrics["student"]["mae_mean"],
        }


class PrototypeAlignedStudent:
    """
    MI fine-tuning of the Baseline 5 student, pulling its EEG latent towards the ME
    prototype of each trial's target:
        loss = reg(student) + proto_weight * (1 - cos(latent, prototype[target]))
    With no prototypes (or weight 0) this is plain regression, as used for evaluation.
    """

    def __init__(self, criterion, prototypes=None, proto_weight=0.0):
        self.prototypes = prototypes
        self.proto_weight = proto_weight
        self.criterion = criterion

    def __call__(self, model, batch):
        eeg, y, target_id, subject_id = batch
        y_pred, h_eeg = model.forward_student(eeg, subject_id, return_latent=True)
        reg_loss = self.criterion(y_pred, y)

        if self.prototypes is not None and self.proto_weight > 0:
            p_loss = proto_alignment_loss(h_eeg, target_id, self.prototypes.to(eeg.device))
        else:
            p_loss = torch.tensor(0.0, device=eeg.device)

        loss = reg_loss + self.proto_weight * p_loss
        return StepOutput(
            loss=loss, y=y, preds={"y": y_pred}, parts={"reg_loss": reg_loss, "proto_loss": p_loss}
        )

    def stage_row(self, train, val):
        return {**_single_prediction_row(train, val), "val_proto_loss": val.losses["proto_loss"]}
