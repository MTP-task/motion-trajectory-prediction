"""Baseline 0: time-lagged linear regression from EEG to velocity."""

import logging

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from mtp.config import EEG_FS, TARGET_LEN, TRAJ_FS
from mtp.dataset import denormalize_targets
from mtp.modeling.metrics import compute_metrics_np

log = logging.getLogger(__name__)


def make_time_lagged_linear_features(
    X: np.ndarray,
    lags_sec: list[float],
    lag_window_sec: float,
):
    """Build time-lagged EEG features for Baseline 0.

    For each of the 150 velocity samples, the EEG is averaged in a short window around
    each lag, for example 300, 200 and 100 ms before the sample and at the sample
    itself, separately for every band and channel. Only EEG close to each sample is
    used, so the model cannot predict the whole trajectory from the whole epoch, which
    could amount to recognizing a template. Neither the target label nor the time step
    is a feature.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_bands, n_channels, n_samples)
        Filter-bank EEG cropped to 0-5 s, from `make_eeg_input`.
    lags_sec : list of float
        Lags in seconds. Negative values are before the velocity sample.
    lag_window_sec : float
        Length in seconds of the window averaged around each lag.

    Returns
    -------
    ndarray of float32, shape (n_trials, 150, n_bands * n_channels * n_lags)
        A window that falls completely outside the 0-5 s crop gives zeros, the average
        of the z-scored EEG.
    """
    N, B, C, T_eeg = X.shape
    n_lags = len(lags_sec)
    n_feat = B * C * n_lags

    F = np.zeros((N, TARGET_LEN, n_feat), dtype=np.float32)

    half_window = max(1, round((lag_window_sec * EEG_FS) / 2.0))

    for ti in range(TARGET_LEN):
        # Time of the middle of velocity sample ti.
        t_sec = (ti + 0.5) / TRAJ_FS

        lag_features = []

        for lag_sec in lags_sec:
            center = round((t_sec + lag_sec) * EEG_FS)
            start = max(0, center - half_window)
            end = min(T_eeg, center + half_window)
            if end > start:
                feat = X[..., start:end].mean(axis=-1)
            else:
                feat = np.zeros((N, B, C), dtype=np.float32)
            lag_features.append(feat.reshape(N, -1))

        F[:, ti, :] = np.concatenate(lag_features, axis=1)

    return F


def run_baseline0_linear(data, scaler, dim_names, cfg):
    """Fit and test Baseline 0 on one condition.

    Every pair of a trial and a velocity sample is one row: its features come from
    `make_time_lagged_linear_features` and its target is the velocity at that sample.
    The model is fitted on the training and validation trials together, because
    nothing is tuned on the validation set, and scored on the test trials with the same
    metrics as the deep models.

    Parameters
    ----------
    data : PreparedCondition
        One condition from `prepare_dataset`, with normalized targets (``y_norm``).
    scaler : dict
        Target scaler, used to score the predictions in their original units.
    dim_names : list of str
        Name of each output dimension.
    cfg : Config
        Run settings. The ``baseline0_*`` and ``linear_*`` settings are used.

    Returns
    -------
    dict of float
        Test metrics from `compute_metrics_np`, with ``loss`` set to NaN.
    """
    log.info(
        "===== Baseline 0: Time-lagged MLR ===== (lags %s s, window %.1f ms)",
        cfg.linear_lags_sec,
        cfg.linear_lag_window_sec * 1000,
    )

    F_train, F_val, F_test = (
        make_time_lagged_linear_features(split.X, cfg.linear_lags_sec, cfg.linear_lag_window_sec)
        for split in data.splits()
    )
    n_features = F_train.shape[-1]
    n_outputs = data.train.y_norm.shape[-1]

    model = (
        Ridge(alpha=cfg.baseline0_ridge_alpha) if cfg.baseline0_use_ridge else LinearRegression()
    )
    model.fit(
        np.concatenate([F_train, F_val]).reshape(-1, n_features),
        np.concatenate([data.train.y_norm, data.val.y_norm]).reshape(-1, n_outputs),
    )
    y_pred = model.predict(F_test.reshape(-1, n_features)).reshape(data.test.y_norm.shape)

    metrics = compute_metrics_np(
        denormalize_targets(data.test.y_norm, scaler),
        denormalize_targets(y_pred, scaler),
        dim_names,
    )
    metrics["loss"] = np.nan
    return metrics
