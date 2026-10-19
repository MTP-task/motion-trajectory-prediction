"""Losses, metrics, Baseline 0 features and model output shapes."""

import numpy as np
import pytest
import torch

from mtp.config import TARGET_LEN
from mtp.modeling.linear import make_time_lagged_linear_features
from mtp.modeling.metrics import SequenceRegressionLoss, compute_metrics_np, sequence_corrcoef
from mtp.modeling.models import (
    DistillationSimpleModel,
    EEGOnlyModel,
    FullPrivilegedModel,
    FusionModel,
)
from mtp.modeling.objectives import fusion_zero_imu

DIMS = {"n_bands": 2, "n_channels": 4, "out_dim": 9}


def test_metrics_match_numpy():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(5, 150, 3))
    y_pred = y_true + rng.normal(scale=0.5, size=y_true.shape)
    metrics = compute_metrics_np(y_true, y_pred, ["a", "b", "c"])

    flat_true, flat_pred = y_true.reshape(-1, 3), y_pred.reshape(-1, 3)
    for d, name in enumerate("abc"):
        err = flat_true[:, d] - flat_pred[:, d]
        assert metrics[f"r_{name}"] == pytest.approx(
            np.corrcoef(flat_true[:, d], flat_pred[:, d])[0, 1]
        )
        assert metrics[f"rmse_{name}"] == pytest.approx(np.sqrt(np.mean(err**2)))
        assert metrics[f"mae_{name}"] == pytest.approx(np.mean(np.abs(err)))
    assert metrics["r_mean"] == pytest.approx(np.mean([metrics[f"r_{n}"] for n in "abc"]))


def test_constant_prediction_has_zero_correlation():
    y_true = np.random.default_rng(1).normal(size=(2, 150, 1))
    assert compute_metrics_np(y_true, np.zeros_like(y_true), ["a"])["r_a"] == 0.0


def test_sequence_corrcoef_matches_numpy():
    g = torch.Generator().manual_seed(0)
    y_true = torch.randn(4, 150, 3, generator=g)
    y_pred = y_true + 0.5 * torch.randn(4, 150, 3, generator=g)
    expected = np.mean(
        [
            np.corrcoef(y_true.reshape(-1, 3)[:, d], y_pred.reshape(-1, 3)[:, d])[0, 1]
            for d in range(3)
        ]
    )
    assert sequence_corrcoef(y_pred, y_true).item() == pytest.approx(expected, abs=1e-5)


def test_regression_loss_is_zero_for_a_perfect_prediction():
    y = torch.randn(2, 150, 3)
    assert SequenceRegressionLoss()(y, y).item() == pytest.approx(0.0, abs=1e-6)


def test_lag_features_average_the_right_windows():
    # EEG whose value is its own sample index, so a window's mean is its centre index - 0.5.
    n_samples = 2560
    X = np.arange(n_samples, dtype=np.float32).reshape(1, 1, 1, n_samples)
    features = make_time_lagged_linear_features(X, lags_sec=[-0.1, 0.0], lag_window_sec=0.05)
    assert features.shape == (1, TARGET_LEN, 2)

    i = 60
    t = (i + 0.5) / 30
    for k, lag in enumerate([-0.1, 0.0]):
        centre = round((t + lag) * 512)
        assert features[0, i, k] == pytest.approx(centre - 0.5)


def test_lag_windows_before_the_crop_are_zero():
    X = np.ones((1, 1, 1, 2560), dtype=np.float32)
    features = make_time_lagged_linear_features(X, lags_sec=[-0.3], lag_window_sec=0.05)
    assert features[0, 0, 0] == 0.0  # sample 0 minus 300 ms lies before 0 s
    assert features[0, 30, 0] == 1.0


@pytest.mark.parametrize("fusion", ["concat", "cross_mha", "self_mha"])
def test_fusion_model_with_zero_imu(fusion):
    model = FusionModel(fusion, **DIMS).eval()
    batch = (
        torch.randn(3, 2, 4, 2560),
        torch.zeros(3, 150, 9),
        torch.zeros(3, dtype=torch.long),
        torch.zeros(3, dtype=torch.long),
    )
    y_pred, _ = fusion_zero_imu(model, batch)
    assert y_pred.shape == (3, 150, 9)


def test_unknown_fusion_is_rejected():
    with pytest.raises(ValueError, match="sum"):
        FusionModel("sum", **DIMS)


def test_eeg_only_and_teacher_student_shapes():
    eeg, imu = torch.randn(3, 2, 4, 2560), torch.randn(3, 150, 6)
    subject_id = torch.zeros(3, dtype=torch.long)
    assert EEGOnlyModel(**DIMS).eval()(eeg).shape == (3, 150, 9)

    teacher, student = DistillationSimpleModel(**DIMS).eval().forward_teacher_student(eeg, imu)
    assert teacher.shape == student.shape == (3, 150, 9)

    teacher, student, h_eeg, _ = (
        FullPrivilegedModel(**DIMS).eval().forward_teacher_student(eeg, imu, subject_id)
    )
    assert teacher.shape == student.shape == (3, 150, 9)
    assert h_eeg.shape == (3, 64, 150)
