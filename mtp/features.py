"""Turn EEG epochs into filter-bank model input, and normalize the IMU signals."""

import logging

import numpy as np
from scipy.signal import butter, sosfiltfilt

from mtp.config import BANDS, EEG_CROP_TMAX, EEG_CROP_TMIN, EEG_EPOCH_TMIN, EEG_FS

log = logging.getLogger(__name__)


def zscore_epochs_per_channel(x: np.ndarray, eps: float = 1e-8):
    """Scale each trial and channel to zero mean and unit variance over time."""
    mean = np.nanmean(x, axis=-1, keepdims=True)
    std = np.nanstd(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def clip_extreme_values(x: np.ndarray, clip_value: float = 8.0):
    """Limit values to the range from -clip_value to clip_value."""
    return np.clip(x, -clip_value, clip_value)


def bandpass_epochs(x: np.ndarray, fs: float, low: float, high: float, order: int = 4):
    """Band-pass filter along the last axis with a zero-phase Butterworth filter.

    Parameters
    ----------
    x : ndarray, shape (..., n_samples)
        Signals to filter.
    fs : float
        Sampling rate in Hz.
    low, high : float
        Pass band in Hz. A `high` at or above the Nyquist frequency is lowered to just
        below it.
    order : int
        Filter order.

    Returns
    -------
    ndarray
        Filtered signals, same shape as `x`.
    """
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
    """Return the first and one-past-last sample index of the crop window.

    Parameters
    ----------
    n_samples : int
        Number of samples in the full epoch.
    fs : float
        Sampling rate in Hz.
    epoch_tmin : float
        Time in seconds of the epoch's first sample, relative to movement onset.
    crop_tmin, crop_tmax : float
        Start and end of the crop window in seconds, relative to movement onset.

    Returns
    -------
    start, end : int

    Raises
    ------
    ValueError
        If the crop window does not fit inside the epoch.
    """
    start = round((crop_tmin - epoch_tmin) * fs)
    length = round((crop_tmax - crop_tmin) * fs)
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
    clip_value: float = 8.0,
):
    """Filter, crop and normalize EEG epochs for the models.

    Each band in `BANDS` is filtered on the full epoch before the crop, so the filter's
    edge effects fall outside the window the model sees. Each band is then z-scored per
    trial and channel and clipped.

    Parameters
    ----------
    eeg : ndarray, shape (n_trials, n_channels, n_samples)
        EEG epochs in volts, covering -2 to 7 s around movement onset.
    fs : float
        Sampling rate in Hz.
    use_filterbank : bool
        Split the EEG into the bands in `BANDS`. If False, return one broadband signal.
    epoch_tmin : float
        Time in seconds of the epoch's first sample.
    crop_tmin, crop_tmax : float
        Window to keep, in seconds from movement onset.
    clip_value : float
        Limit for the z-scored values, in standard deviations.

    Returns
    -------
    ndarray of float32, shape (n_trials, n_bands, n_channels, n_crop_samples)
        With the default settings, n_bands is 6 (1 without the filter bank) and
        n_crop_samples is 2560.
    """
    eeg = eeg.astype(np.float32)
    eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
    eeg = clip_extreme_values(eeg, clip_value=clip_value)

    crop_start, crop_end = get_crop_sample_indices(
        n_samples=eeg.shape[-1],
        fs=fs,
        epoch_tmin=epoch_tmin,
        crop_tmin=crop_tmin,
        crop_tmax=crop_tmax,
    )

    log.info(
        "EEG full epoch length: %d samples | crop %g..%g s -> samples [%d:%d] (%d samples)",
        eeg.shape[-1],
        crop_tmin,
        crop_tmax,
        crop_start,
        crop_end,
        crop_end - crop_start,
    )

    if not use_filterbank:
        x_crop = eeg[..., crop_start:crop_end]
        X = zscore_epochs_per_channel(x_crop)[:, None, :, :]
        X = clip_extreme_values(X, clip_value=clip_value)
        return X.astype(np.float32)

    band_arrays = []
    for band_name, (low, high) in BANDS.items():
        band_name_lower = band_name.lower()
        order = 2 if band_name_lower in ["scp", "slow", "delta"] else 4

        log.info("Filtering %s: %s-%s Hz on the full epoch, then cropping", band_name, low, high)

        xb_full = bandpass_epochs(eeg, fs, low, high, order=order)
        xb = xb_full[..., crop_start:crop_end]
        xb = zscore_epochs_per_channel(xb)
        xb = clip_extreme_values(xb, clip_value=clip_value)
        band_arrays.append(xb.astype(np.float32))

    X = np.stack(band_arrays, axis=1)
    return X.astype(np.float32)


def normalize_imu_train_only(imu_train, imu_val, imu_test, eps: float = 1e-8):
    """Normalize the IMU signals with statistics from the training trials only.

    Parameters
    ----------
    imu_train, imu_val, imu_test : ndarray, shape (n_trials, 150, 6)
        IMU rotation velocities of each split.
    eps : float
        Added to the standard deviation to avoid dividing by zero.

    Returns
    -------
    imu_train_n, imu_val_n, imu_test_n : ndarray of float32
        Normalized signals, same shapes as the inputs.
    scaler : dict
        "mean" and "std" of each signal, each of shape (1, 1, 6).
    """
    mean = imu_train.mean(axis=(0, 1), keepdims=True)
    std = imu_train.std(axis=(0, 1), keepdims=True)

    imu_train_n = (imu_train - mean) / (std + eps)
    imu_val_n = (imu_val - mean) / (std + eps)
    imu_test_n = (imu_test - mean) / (std + eps)

    scaler = {"mean": mean.astype(np.float32), "std": (std + eps).astype(np.float32)}

    return (
        imu_train_n.astype(np.float32),
        imu_val_n.astype(np.float32),
        imu_test_n.astype(np.float32),
        scaler,
    )
