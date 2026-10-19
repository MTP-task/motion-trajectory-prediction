"""Turn EEG epochs into filter-bank model input."""

import logging

import numpy as np
from scipy.signal import butter, sosfiltfilt

from mtp.config import (
    BANDS,
    EEG_CROP_TMAX,
    EEG_CROP_TMIN,
    EEG_EPOCH_TMIN,
    EEG_FS,
    MODEL_EEG_SAMPLES,
)

log = logging.getLogger(__name__)


def zscore_epochs_per_channel(x: np.ndarray, eps: float = 1e-8):
    """Scale each trial and channel to zero mean and unit variance over time."""
    mean = np.nanmean(x, axis=-1, keepdims=True)
    std = np.nanstd(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


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


def get_crop_sample_indices(n_samples: int):
    """Return the first and one-past-last sample of the model window in an EEG epoch.

    The window runs from `EEG_CROP_TMIN` to `EEG_CROP_TMAX` seconds after movement onset,
    and the epoch starts at `EEG_EPOCH_TMIN`.

    Parameters
    ----------
    n_samples : int
        Number of samples in the full epoch.

    Returns
    -------
    start, end : int

    Raises
    ------
    ValueError
        If the window does not fit inside the epoch.
    """
    start = round((EEG_CROP_TMIN - EEG_EPOCH_TMIN) * EEG_FS)
    end = start + MODEL_EEG_SAMPLES
    if start < 0 or end > n_samples:
        raise ValueError(
            f"The model window ({EEG_CROP_TMIN} to {EEG_CROP_TMAX} s, samples {start} to "
            f"{end}) does not fit in an epoch of {n_samples} samples starting at "
            f"{EEG_EPOCH_TMIN} s."
        )
    return start, end


def make_eeg_input(eeg: np.ndarray, use_filterbank: bool = True, clip_value: float = 8.0):
    """Filter, crop and normalize EEG epochs for the models.

    Each band in `BANDS` is filtered on the full epoch before the crop, so the filter's
    edge effects fall outside the window the model sees. Each band is then z-scored per
    trial and channel and clipped.

    Parameters
    ----------
    eeg : ndarray, shape (n_trials, n_channels, n_samples)
        EEG epochs in volts, sampled at `EEG_FS` and covering -2 to 7 s around movement
        onset.
    use_filterbank : bool
        Split the EEG into the bands in `BANDS`. If False, return one broadband signal.
    clip_value : float
        Limit for the z-scored values, in standard deviations.

    Returns
    -------
    ndarray of float32, shape (n_trials, n_bands, n_channels, n_crop_samples)
        With the default settings, n_bands is 6 (1 without the filter bank) and
        n_crop_samples is 2560.
    """
    eeg = np.nan_to_num(eeg.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    start, end = get_crop_sample_indices(eeg.shape[-1])
    log.info(
        "EEG epochs have %d samples; the models use %g to %g s, samples %d to %d",
        eeg.shape[-1],
        EEG_CROP_TMIN,
        EEG_CROP_TMAX,
        start,
        end,
    )

    if not use_filterbank:
        return _crop_and_scale(eeg, start, end, clip_value)[:, None]

    bands = []
    for band_name, (low, high) in BANDS.items():
        log.info("Filtering %s: %s-%s Hz on the full epoch, then cropping", band_name, low, high)
        order = 2 if band_name in ("slow", "delta") else 4
        filtered = bandpass_epochs(eeg, EEG_FS, low, high, order=order)
        bands.append(_crop_and_scale(filtered, start, end, clip_value))
    return np.stack(bands, axis=1)


def _crop_and_scale(x, start, end, clip_value):
    """Crop to the model window, z-score each trial and channel, and clip."""
    x = zscore_epochs_per_channel(x[..., start:end])
    return np.clip(x, -clip_value, clip_value).astype(np.float32)
