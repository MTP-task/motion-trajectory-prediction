"""What each training stage computes for one batch.

An objective takes a model and a batch and returns the loss to optimize together with
the predictions to score. The training loop in `mtp.modeling.trainers` is the same for
every baseline; only the objective changes.

Batches come from `mtp.dataset.make_loader`. EEG-only loaders give (eeg, y, target_id,
subject_id) and EEG+IMU loaders give (eeg, imu, y, target_id, subject_id).

The forward functions `eeg_only`, `fusion_real_imu`, `fusion_zero_imu` and
`distill_student` each take a model and a batch and return the prediction and the target.
"""

from dataclasses import dataclass, field

import torch
from torch import nn

from mtp.config import IMU_DIM, TARGET_LEN
from mtp.modeling.metrics import proto_alignment_loss


@dataclass
class StepOutput:
    """What an objective returns for one batch.

    Attributes
    ----------
    loss : Tensor
        Loss to optimize.
    y : Tensor
        True velocities.
    preds : dict of Tensor
        Predicted velocities to score, keyed by name: "y", or "teacher" and "student"
        when one pass gives two predictions.
    parts : dict of Tensor
        Extra loss terms to report next to the total, such as "proto_loss".
    """

    loss: torch.Tensor
    y: torch.Tensor
    preds: dict
    parts: dict = field(default_factory=dict)


def eeg_only(model, batch):
    """Predict from the EEG alone."""
    eeg, y, _target_id, _subject_id = batch
    return model(eeg), y


def fusion_real_imu(model, batch):
    """Predict from the EEG and the recorded IMU (ME trials)."""
    eeg, imu, y, _target_id, _subject_id = batch
    return model(eeg, imu), y


def fusion_zero_imu(model, batch):
    """Predict from the EEG with an all-zero IMU input, since MI trials have no IMU."""
    eeg, y, _target_id, _subject_id = batch
    imu = torch.zeros((eeg.shape[0], TARGET_LEN, IMU_DIM), dtype=eeg.dtype, device=eeg.device)
    return model(eeg, imu), y


def distill_student(model, batch):
    """Predict with the EEG-only student of a teacher-student model."""
    eeg, y, _target_id, _subject_id = batch
    return model.forward_student(eeg), y


def _single_prediction_row(train, val):
    return {
        "train_r_mean": train.metrics["y"]["r_mean"],
        "val_r_mean": val.metrics["y"]["r_mean"],
        "val_rmse_mean": val.metrics["y"]["rmse_mean"],
        "val_mae_mean": val.metrics["y"]["mae_mean"],
    }


class Supervised:
    """Fit one prediction to the target with a regression loss.

    Parameters
    ----------
    forward : callable
        One of the forward functions in this module, such as `eeg_only`.
    criterion : nn.Module
        Regression loss, normally `SequenceRegressionLoss`.
    """

    def __init__(self, forward, criterion):
        self.forward = forward
        self.criterion = criterion

    def __call__(self, model, batch):
        """Return the loss and prediction for one batch as a `StepOutput`."""
        y_pred, y = self.forward(model, batch)
        return StepOutput(loss=self.criterion(y_pred, y), y=y, preds={"y": y_pred})

    def stage_row(self, train, val):
        """Return the columns this objective adds to the per-epoch "stages" sheet."""
        return _single_prediction_row(train, val)


class TeacherStudent:
    """Train a teacher and a student together on ME data (Baselines 4 and 5).

    The teacher sees EEG and IMU; the student sees only EEG. The loss is::

        teacher_weight * reg(teacher) + student_weight * reg(student)
            + kd_weight * SmoothL1(student, teacher)

    where reg is `criterion`. The teacher's output is detached in the last term, so that
    term only moves the student toward the teacher.

    Parameters
    ----------
    criterion : nn.Module
        Regression loss.
    teacher_weight, student_weight, kd_weight : float
        Weights of the three terms.
    kd_beta : float
        SmoothL1 threshold of the last term.
    use_subject_id : bool
        Pass subject IDs to ``model.forward_teacher_student``, as Baseline 5's model
        requires.
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
        """Return the loss and both predictions for one batch as a `StepOutput`."""
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
        """Return the columns this objective adds to the per-epoch "stages" sheet."""
        return {
            "teacher_val_r_mean": val.metrics["teacher"]["r_mean"],
            "student_val_r_mean": val.metrics["student"]["r_mean"],
            "student_val_rmse_mean": val.metrics["student"]["rmse_mean"],
            "student_val_mae_mean": val.metrics["student"]["mae_mean"],
        }


class PrototypeAlignedStudent:
    """Fine-tune Baseline 5's student on MI data, pulled toward the ME prototypes.

    The loss is::

        reg(student) + proto_weight * proto_alignment_loss(features, prototypes)

    Without prototypes, or with `proto_weight` 0, only the regression loss remains. That
    is how the model is evaluated.

    Parameters
    ----------
    criterion : nn.Module
        Regression loss.
    prototypes : Tensor, shape (n_targets, latent_dim), optional
        Prototypes from `compute_me_prototypes`.
    proto_weight : float
        Weight of the prototype term.
    """

    def __init__(self, criterion, prototypes=None, proto_weight=0.0):
        self.prototypes = prototypes
        self.proto_weight = proto_weight
        self.criterion = criterion

    def __call__(self, model, batch):
        """Return the loss, its two terms and the prediction for one batch."""
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
        """Return the columns this objective adds to the per-epoch "stages" sheet."""
        return {**_single_prediction_row(train, val), "val_proto_loss": val.losses["proto_loss"]}
