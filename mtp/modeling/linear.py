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
            start = center - half_window
            end = center + half_window

            if end <= 0 or start >= T_eeg:
                feat = np.zeros((N, B, C), dtype=np.float32)
            else:
                start_clip = max(0, start)
                end_clip = min(T_eeg, end)

                if end_clip <= start_clip:
                    feat = np.zeros((N, B, C), dtype=np.float32)
                else:
                    feat = X[..., start_clip:end_clip].mean(axis=-1).astype(np.float32)

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

    Ftr, Fva, Fte = (
        make_time_lagged_linear_features(split.X, cfg.linear_lags_sec, cfg.linear_lag_window_sec)
        for split in data.splits()
    )

    Ntr, Ttr, P = Ftr.shape
    Nva, Tva, _ = Fva.shape
    Nte, Tte, _ = Fte.shape

    X_train_final = np.concatenate(
        [
            Ftr.reshape(Ntr * Ttr, P),
            Fva.reshape(Nva * Tva, P),
        ],
        axis=0,
    )

    y_train_final = np.concatenate(
        [
            data.train.y_norm.reshape(Ntr * Ttr, -1),
            data.val.y_norm.reshape(Nva * Tva, -1),
        ],
        axis=0,
    )

    X_test = Fte.reshape(Nte * Tte, P)

    if cfg.baseline0_use_ridge:
        model = Ridge(alpha=cfg.baseline0_ridge_alpha)
    else:
        model = LinearRegression()

    model.fit(X_train_final, y_train_final)

    y_pred_flat = model.predict(X_test)
    y_pred_n = y_pred_flat.reshape(data.test.y_norm.shape)

    y_true_orig = denormalize_targets(data.test.y_norm, scaler)
    y_pred_orig = denormalize_targets(y_pred_n, scaler)

    metrics = compute_metrics_np(y_true_orig, y_pred_orig, dim_names)
    metrics["loss"] = np.nan
    return metrics
