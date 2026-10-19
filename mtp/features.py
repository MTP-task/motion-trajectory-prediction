import numpy as np
from scipy.signal import butter, sosfiltfilt

from mtp.config import BANDS, EEG_CROP_TMAX, EEG_CROP_TMIN, EEG_EPOCH_TMIN, EEG_FS


# ============================================================
# 3) EEG / IMU PREPROCESSING
# ============================================================

def zscore_epochs_per_channel(x: np.ndarray, eps: float = 1e-8):
    mean = np.nanmean(x, axis=-1, keepdims=True)
    std = np.nanstd(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def clip_extreme_values(x: np.ndarray, clip_value: float = 8.0):
    return np.clip(x, -clip_value, clip_value)


def bandpass_epochs(x: np.ndarray, fs: float, low: float, high: float, order: int = 4):
    nyq = fs / 2.0
    if high >= nyq:
        high = nyq - 1e-3
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def get_crop_sample_indices(
    n_samples: int,
    fs: float = EEG_FS,
    epoch_tmin: float = EEG_EPOCH_TMIN,
    crop_tmin: float = EEG_CROP_TMIN,
    crop_tmax: float = EEG_CROP_TMAX,
):
    start = int(round((crop_tmin - epoch_tmin) * fs))
    length = int(round((crop_tmax - crop_tmin) * fs))
    end = start + length

    if start < 0:
        raise ValueError(
            f"Crop start sample is negative ({start}). "
            f"Check EEG_EPOCH_TMIN={epoch_tmin}, EEG_CROP_TMIN={crop_tmin}."
        )
    if end > n_samples:
        raise ValueError(
            f"Crop end sample ({end}) is larger than EEG length ({n_samples}). "
            f"Check input epoch length and crop settings."
        )

    return start, end


def make_eeg_input(
    eeg: np.ndarray,
    fs: float,
    use_filterbank: bool = True,
    epoch_tmin: float = EEG_EPOCH_TMIN,
    crop_tmin: float = EEG_CROP_TMIN,
    crop_tmax: float = EEG_CROP_TMAX,
):
    """
    Input:
        eeg: (N, C, T_full), expected to cover -2..7 s.
    Output:
        X: (N, B, C, T_crop), crop is 0..5 s.

    Filter-bank mode:
        filter full epoch first, then crop 0..5 s.
    """
    eeg = eeg.astype(np.float32)
    eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
    eeg = clip_extreme_values(eeg, clip_value=8.0)

    crop_start, crop_end = get_crop_sample_indices(
        n_samples=eeg.shape[-1],
        fs=fs,
        epoch_tmin=epoch_tmin,
        crop_tmin=crop_tmin,
        crop_tmax=crop_tmax,
    )

    print(
        f"EEG full epoch length: {eeg.shape[-1]} samples | "
        f"crop {crop_tmin:g}..{crop_tmax:g} s -> samples [{crop_start}:{crop_end}] "
        f"({crop_end - crop_start} samples)"
    )

    if not use_filterbank:
        x_crop = eeg[..., crop_start:crop_end]
        X = zscore_epochs_per_channel(x_crop)[:, None, :, :]
        X = clip_extreme_values(X, clip_value=8.0)
        return X.astype(np.float32)

    band_arrays = []
    for band_name, (low, high) in BANDS.items():
        band_name_lower = band_name.lower()
        order = 2 if band_name_lower in ["scp", "slow", "delta"] else 4

        print(
            f"Filtering {band_name}: {low}-{high} Hz on full epoch, "
            f"then crop to {crop_tmin:g}..{crop_tmax:g} s"
        )

        xb_full = bandpass_epochs(eeg, fs, low, high, order=order)
        xb = xb_full[..., crop_start:crop_end]
        xb = zscore_epochs_per_channel(xb)
        xb = clip_extreme_values(xb, clip_value=8.0)
        band_arrays.append(xb.astype(np.float32))

    X = np.stack(band_arrays, axis=1)
    return X.astype(np.float32)


def normalize_imu_train_only(imu_train, imu_val, imu_test, eps: float = 1e-8):
    mean = imu_train.mean(axis=(0, 1), keepdims=True)
    std = imu_train.std(axis=(0, 1), keepdims=True)

    imu_train_n = (imu_train - mean) / (std + eps)
    imu_val_n = (imu_val - mean) / (std + eps)
    imu_test_n = (imu_test - mean) / (std + eps)

    scaler = {"mean": mean.astype(np.float32), "std": (std + eps).astype(np.float32)}

    return imu_train_n.astype(np.float32), imu_val_n.astype(np.float32), imu_test_n.astype(np.float32), scaler
