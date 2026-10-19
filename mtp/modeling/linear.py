from typing import List

import numpy as np
from sklearn.linear_model import Ridge, LinearRegression

from mtp.config import (
    BASELINE0_RIDGE_ALPHA, BASELINE0_USE_RIDGE, EEG_FS, LINEAR_LAG_WINDOW_SEC, LINEAR_LAGS_SEC,
    TARGET_LEN, TRAJ_FS,
)
from mtp.dataset import denormalize_targets
from mtp.modeling.metrics import compute_metrics_np


# ============================================================
# 9) BASELINE 0: TIME-LAGGED LINEAR MLR
# ============================================================

def make_time_lagged_linear_features(
    X: np.ndarray,
    lags_sec: List[float] = None,
    lag_window_sec: float = LINEAR_LAG_WINDOW_SEC,
):
    """
    Create time-lagged linear EEG features for MLR.

    Input:
        X: (N, B, C, T_eeg)
           Already filter-bank processed and cropped to 0..5 s.

    Output:
        F: (N, TARGET_LEN, B*C*n_lags)

    Meaning:
        For every trajectory sample y(t), the feature vector contains EEG values
        from several lagged time points:
            EEG(t - 300 ms), EEG(t - 200 ms), EEG(t - 100 ms), EEG(t)

        Each lag feature is the mean EEG value inside a short window
        around the lag center. This avoids using the whole 0..5 s EEG epoch
        to predict the whole trajectory, which was too strong and could behave
        like template decoding.

    Important:
        This is closer to classical time-lagged MLR:
            EEG(t + lag) -> velocity(t)

        It does NOT include target labels or time-bin IDs as input features.
    """
    if lags_sec is None:
        lags_sec = LINEAR_LAGS_SEC

    N, B, C, T_eeg = X.shape
    n_lags = len(lags_sec)
    n_feat = B * C * n_lags

    F = np.zeros((N, TARGET_LEN, n_feat), dtype=np.float32)

    half_window = max(1, int(round((lag_window_sec * EEG_FS) / 2.0)))

    for ti in range(TARGET_LEN):
        # Use center of each 30-Hz trajectory sample.
        t_sec = (ti + 0.5) / TRAJ_FS

        lag_features = []

        for lag_sec in lags_sec:
            center = int(round((t_sec + lag_sec) * EEG_FS))
            start = center - half_window
            end = center + half_window

            # If the lag window is fully outside the 0..5 s EEG crop,
            # use zeros. Since EEG features were z-scored, zero means average activity.
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


def run_baseline0_linear(data, scaler, dim_names):
    """
    Baseline 0: time-lagged MLR / Ridge regression.

    Previous version:
        whole EEG epoch -> whole trajectory
        This could be too powerful and template-like.

    New version:
        for each trajectory sample:
            EEG(t + lags) -> velocity(t)

    Training:
        rows = epochs * trajectory_time_samples
        X    = time-lagged EEG features
        y    = velocity at that time sample

    Evaluation:
        predictions are reshaped back to (epochs, 150, output_dim)
        and evaluated using the same r/RMSE/MAE metrics as the deep models.
    """
    print("\n===== Baseline 0: Time-lagged MLR =====")
    print(f"Using lags in seconds: {LINEAR_LAGS_SEC}")
    print(f"Lag window: {LINEAR_LAG_WINDOW_SEC * 1000:.1f} ms")

    Ftr = make_time_lagged_linear_features(data["X_train"])
    Fva = make_time_lagged_linear_features(data["X_val"])
    Fte = make_time_lagged_linear_features(data["X_test"])

    Ntr, Ttr, P = Ftr.shape
    Nva, Tva, _ = Fva.shape
    Nte, Tte, _ = Fte.shape

    # Use train + validation for final linear fit.
    X_train_final = np.concatenate([
        Ftr.reshape(Ntr * Ttr, P),
        Fva.reshape(Nva * Tva, P),
    ], axis=0)

    y_train_final = np.concatenate([
        data["y_train_n"].reshape(Ntr * Ttr, -1),
        data["y_val_n"].reshape(Nva * Tva, -1),
    ], axis=0)

    X_test = Fte.reshape(Nte * Tte, P)

    if BASELINE0_USE_RIDGE:
        model = Ridge(alpha=BASELINE0_RIDGE_ALPHA)
    else:
        model = LinearRegression()

    model.fit(X_train_final, y_train_final)

    y_pred_flat = model.predict(X_test)
    y_pred_n = y_pred_flat.reshape(data["y_test_n"].shape)

    y_true_orig = denormalize_targets(data["y_test_n"], scaler)
    y_pred_orig = denormalize_targets(y_pred_n, scaler)

    metrics = compute_metrics_np(y_true_orig, y_pred_orig, dim_names)
    metrics["loss"] = np.nan

    print(
        f"Baseline 0 time-lagged MLR test "
        f"r_mean={metrics['r_mean']:.4f}, "
        f"RMSE={metrics['rmse_mean']:.4f}, "
        f"MAE={metrics['mae_mean']:.4f}"
    )

    return metrics
