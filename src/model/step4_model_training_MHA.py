"""
Run Baseline 0-7 and save ME/MI correlation results to Excel.

Baselines
---------
Baseline 0: Time-lagged Linear method (MLR / Ridge-stabilized MLR)
Baseline 1: MI-only EEG model
Baseline 2: ME EEG-only pretraining -> MI EEG-only fine-tuning
Baseline 3: ME EEG+IMU pretraining -> MI EEG-only fine-tuning with zero IMU
            No distillation, no prototype alignment
Baseline 4: ME EEG+IMU teacher + EEG-only student distillation -> MI fine-tuning
            No prototype alignment
Baseline 5: Full model
            ME EEG+IMU + lag-aware fusion + student distillation
            + subject adaptation + prototype alignment
Baseline 6: ME EEG+IMU pretraining -> MI EEG-only fine-tuning with zero IMU
            Bidirectional Cross-Modal Multi-Head Attention fusion
            No distillation, no prototype alignment
Baseline 7: ME EEG+IMU pretraining -> MI EEG-only fine-tuning with zero IMU
            Normal Multi-Head Self-Attention fusion over fused EEG+IMU tokens
            No distillation, no prototype alignment

Important
---------
- labels_clean is required for both ME and MI for stratified split / target class IDs.
- New trajectory files are already mapped to EEG epoch order:
  each key contains an array with shape (epoch, 150, 3) or (epoch, 150, 6 for IMU).
- EEG input epoch is expected to cover -2..7 s.
- Filter-bank filtering is applied to the full -2..7 s epoch first,
  then the model input is cropped to 0..5 s.
- Final model evaluation is always on the MI test set.
- IMU is not used during MI testing for Baseline 4 and 5.
- Baseline 3 uses zero IMU during MI fine-tuning/testing to test whether
  simple ME+IMU pretraining is enough without distillation/prototype alignment.
- Baseline 6 copies Baseline 3 training/evaluation, but replaces simple
  concatenation fusion with bidirectional cross-modal multi-head attention.
- Baseline 7 copies Baseline 3 training/evaluation, but replaces simple
  concatenation fusion with normal multi-head self-attention over fused EEG+IMU tokens.

How to run
----------
1. Change `subject` and paths in USER SETTINGS.
2. Make sure these files exist:
    EEG_data_<subject>_ME.npy
    EEG_data_<subject>_MI.npy
    trajectory_<subject>_ME.npy
    trajectory_<subject>_MI.npy
    labels_clean_<subject>_ME.npy
    labels_clean_<subject>_MI.npy
3. Run:
    python run_all_baselines_save_excel.py

Output
------
Excel file:
    baseline_results_<subject>.xlsx

Sheets:
    summary
    axis_metrics
    stages
    config
"""

import os
import math
import copy
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional  

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0) USER SETTINGS
# ============================================================

subject = "s026"

# Set False if you want to quickly test only one or two baselines.
RUN_BASELINE_0 = True
RUN_BASELINE_1 = True
RUN_BASELINE_2 = True
RUN_BASELINE_3 = True
RUN_BASELINE_4 = False
RUN_BASELINE_5 = False
RUN_BASELINE_6 = True
RUN_BASELINE_7 = True

DATA_DIR = r"D:\MTP_task\model"

RESULTS_DIR = os.path.join(DATA_DIR, "baseline_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RESULT_XLSX_PATH = os.path.join(RESULTS_DIR, f"baseline_results_{subject}_multihead.xlsx")

ME_EEG_PATH = os.path.join(DATA_DIR, f"EEG_data_{subject}_ME.npy")
ME_TRAJ_PATH = os.path.join(DATA_DIR, f"trajectory_{subject}_ME.npy")
ME_LABELS_PATH = os.path.join(DATA_DIR, f"labels_clean_{subject}_ME.npy")

MI_EEG_PATH = os.path.join(DATA_DIR, f"EEG_data_{subject}_MI.npy")
MI_TRAJ_PATH = os.path.join(DATA_DIR, f"trajectory_{subject}_MI.npy")
MI_LABELS_PATH = os.path.join(DATA_DIR, f"labels_clean_{subject}_MI.npy")


EEG_FS = 512.0
TRAJ_FS = 30.0
EPOCH_SEC = 5.0
TARGET_LEN = int(TRAJ_FS * EPOCH_SEC)  # 150

# EEG input epoch timing.
# Input EEG epoch = -2..7 s, model uses 0..5 s after filter-bank filtering.
EEG_EPOCH_TMIN = -2.0
EEG_CROP_TMIN = 0.0
EEG_CROP_TMAX = 5.0
MODEL_EEG_SEC = EEG_CROP_TMAX - EEG_CROP_TMIN
MODEL_EEG_SAMPLES = int(round(EEG_FS * MODEL_EEG_SEC))  # 2560

# Main output target:
# "wrist"    -> wrist Vx,Vy,Vz
# "elbow"    -> elbow Vx,Vy,Vz
# "shoulder" -> shoulder Vx,Vy,Vz
# "all_xyz"  -> shoulder+elbow+wrist, output_dim=9
# "all_kin"  -> shoulder+elbow+wrist+IMU, output_dim=15
TARGET_STREAM = "all_xyz"

TRAJ_KEY_ALIAS = {
    "wrist": "wirst",
    "wirst": "wirst",
    "elbow": "elbow",
    "shoulder": "shoulder",
    "imu": "IMU",
    "IMU": "IMU",
}

TARGET_ORDER = ["Number 1", "Number 2", "Number 3", "Number 4"]

LABEL_TO_KEY = {
    101: "Number 1",
    102: "Number 2",
    103: "Number 3",
    104: "Number 4",
}
KEY_TO_TARGET_ID = {key: i for i, key in enumerate(TARGET_ORDER)}

USE_FILTERBANK = True
BANDS = {
    "slow": (0.5, 1.0),
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
    # If your final low-pass is exactly 40 Hz, you may change this to (30.0, 40.0).
    "low_gamma": (30.0, 45.0),
}

# Deep model settings
LATENT_DIM = 64
LAG_RADIUS = 3
CROSS_ATTENTION_HEADS = 4  # Used by Baseline 6 bidirectional cross-modal MHA fusion.
NORMAL_MHA_HEADS = 4        # Used by Baseline 7 normal MHA fusion.

BATCH_SIZE = 8
PRETRAIN_EPOCHS = 80
FINETUNE_EPOCHS = 60

PATIENCE_PRETRAIN = 12
PATIENCE_FINETUNE = 12

PRETRAIN_LR = 1e-3
FINETUNE_LR = 2e-4
WEIGHT_DECAY = 1e-4
SEED = 42

# Loss weights
TEACHER_LOSS_WEIGHT = 1.0
STUDENT_SUP_LOSS_WEIGHT = 1.0
KD_LOSS_WEIGHT = 0.5
PROTO_LOSS_WEIGHT = 0.2

# Baseline 0 settings: time-lagged MLR
BASELINE0_USE_RIDGE = True
BASELINE0_RIDGE_ALPHA = 10.0

# Time-lagged MLR setting:
# Each velocity sample y(t) is predicted from EEG features around t + lag.
# Negative lags mean EEG before the target velocity sample.
# Example: -0.300 means EEG 300 ms before the current velocity sample.
LINEAR_LAGS_SEC = [-0.300, -0.200, -0.100, 0.000]
LINEAR_LAG_WINDOW_SEC = 0.050  # average EEG within ±25 ms around each lag center



# ============================================================
# 1) REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 2) DATA LOADING AND TARGET CONSTRUCTION
# ============================================================

def _is_direct_trajectory_format(traj: dict) -> bool:
    """
    New trajectory format:
        traj["shoulder"] -> ndarray (N, 150, 3)
        traj["elbow"]    -> ndarray (N, 150, 3)
        traj["wirst"]    -> ndarray (N, 150, 3)
        traj["IMU"]      -> ndarray (N, 150, 6)

    Old trajectory format:
        traj["shoulder"]["Number 1"] -> ndarray (n1, 150, 3)
        ...
    """
    if not isinstance(traj, dict):
        return False

    required_keys = ["shoulder", "elbow", "wirst", "IMU"]
    if not all(k in traj for k in required_keys):
        return False

    return all(isinstance(traj[k], np.ndarray) for k in required_keys)


def load_eeg_and_trajectory(eeg_path: str, traj_path: str, name: str):
    if not os.path.exists(eeg_path):
        raise FileNotFoundError(f"{name} EEG file not found: {eeg_path}")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"{name} trajectory file not found: {traj_path}")

    eeg = np.load(eeg_path, allow_pickle=True)
    traj = np.load(traj_path, allow_pickle=True).item()

    print(f"\nLoaded {name} EEG:", eeg.shape, eeg.dtype)
    print(f"Loaded {name} trajectory keys:", list(traj.keys()))

    if _is_direct_trajectory_format(traj):
        print(f"{name} trajectory format: DIRECT epoch-aligned arrays")
        for stream_key in traj:
            arr = traj[stream_key]
            print(f"  {stream_key}: {arr.shape}, {arr.dtype}")
    else:
        print(f"{name} trajectory format: OLD nested Number 1-4 dictionary")
        for stream_key in traj:
            print(f"  {stream_key}:")
            for target_name in TARGET_ORDER:
                if target_name in traj[stream_key]:
                    print(f"    {target_name}: {traj[stream_key][target_name].shape}")

    return eeg, traj


def load_labels_clean(labels_path: str, name: str) -> np.ndarray:
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"{name} labels_clean file not found: {labels_path}\n"
            f"Please save labels_clean as .npy first."
        )

    labels = np.load(labels_path, allow_pickle=True)
    labels = np.asarray(labels).astype(np.int32).ravel()

    valid = set(LABEL_TO_KEY.keys())
    found = set(map(int, np.unique(labels)))
    unknown = found - valid
    if len(unknown) > 0:
        raise ValueError(f"{name}: unknown labels found {sorted(unknown)}. Expected only {sorted(valid)}.")

    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLoaded {name} labels_clean: {labels.shape}, {labels.dtype}")
    print(f"{name} label counts:")
    for lab, count in zip(unique, counts):
        print(f"  {lab} -> {LABEL_TO_KEY[int(lab)]}: {count}")

    return labels


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


def _make_dim_names(target_stream: str):
    if target_stream in ["wrist", "wirst", "elbow", "shoulder"]:
        clean_name = "wrist" if target_stream == "wirst" else target_stream
        return [f"{clean_name}_Vx", f"{clean_name}_Vy", f"{clean_name}_Vz"]

    if target_stream == "imu":
        return [
            "IMU_elbow_rotVx", "IMU_elbow_rotVy", "IMU_elbow_rotVz",
            "IMU_wrist_rotVx", "IMU_wrist_rotVy", "IMU_wrist_rotVz",
        ]

    if target_stream == "all_xyz":
        return [
            "shoulder_Vx", "shoulder_Vy", "shoulder_Vz",
            "elbow_Vx", "elbow_Vy", "elbow_Vz",
            "wrist_Vx", "wrist_Vy", "wrist_Vz",
        ]

    if target_stream == "all_kin":
        return [
            "shoulder_Vx", "shoulder_Vy", "shoulder_Vz",
            "elbow_Vx", "elbow_Vy", "elbow_Vz",
            "wrist_Vx", "wrist_Vy", "wrist_Vz",
            "IMU_elbow_rotVx", "IMU_elbow_rotVy", "IMU_elbow_rotVz",
            "IMU_wrist_rotVx", "IMU_wrist_rotVy", "IMU_wrist_rotVz",
        ]

    raise ValueError(f"Unknown TARGET_STREAM: {target_stream}")


def _get_target_epoch(traj: dict, target_stream: str, target_key: str, idx: int) -> np.ndarray:
    if target_stream in ["wrist", "wirst", "elbow", "shoulder"]:
        key = TRAJ_KEY_ALIAS[target_stream]
        return traj[key][target_key][idx]

    if target_stream == "imu":
        return traj["IMU"][target_key][idx]

    if target_stream == "all_xyz":
        return np.concatenate([
            traj["shoulder"][target_key][idx],
            traj["elbow"][target_key][idx],
            traj["wirst"][target_key][idx],
        ], axis=-1)

    if target_stream == "all_kin":
        return np.concatenate([
            traj["shoulder"][target_key][idx],
            traj["elbow"][target_key][idx],
            traj["wirst"][target_key][idx],
            traj["IMU"][target_key][idx],
        ], axis=-1)

    raise ValueError(f"Unknown TARGET_STREAM: {target_stream}")


def _check_label_counts_against_trajectory(labels_clean: np.ndarray, traj: dict, reference_key: str = "wirst"):
    for lab, target_key in LABEL_TO_KEY.items():
        used = int(np.sum(labels_clean == lab))
        available = int(traj[reference_key][target_key].shape[0])

        if used > available:
            raise ValueError(
                f"labels_clean has {used} epochs for {lab}->{target_key}, "
                f"but trajectory has only {available}."
            )

        if used < available:
            print(
                f"Warning: labels_clean uses {used}/{available} epochs for {lab}->{target_key}. "
                "This is okay only if EEG epochs were rejected but trajectory was not."
            )


def _target_ids_from_labels(labels_clean: np.ndarray, n_epochs: int, name: str = "") -> np.ndarray:
    """
    Convert labels_clean to target_ids 0..3.
    labels_clean is no longer used to pick trajectory epochs in the new format;
    it is used only for stratified splitting and class/prototype labels.
    """
    labels_clean = np.asarray(labels_clean).astype(np.int32).ravel()

    if labels_clean.shape[0] != n_epochs:
        raise ValueError(
            f"{name}: labels_clean length ({labels_clean.shape[0]}) does not match "
            f"number of epochs ({n_epochs})."
        )

    target_ids = []
    for lab in labels_clean:
        lab = int(lab)
        if lab not in LABEL_TO_KEY:
            raise ValueError(
                f"{name}: unknown label {lab}. Expected labels: {sorted(LABEL_TO_KEY.keys())}."
            )
        target_key = LABEL_TO_KEY[lab]
        target_ids.append(KEY_TO_TARGET_ID[target_key])

    return np.asarray(target_ids, dtype=np.int64)


def _get_direct_target_array(traj: dict, target_stream: str) -> np.ndarray:
    """
    Get target array from the new direct trajectory format.
    Output shape:
        single stream: (N, 150, 3)
        imu:           (N, 150, 6)
        all_xyz:       (N, 150, 9)
        all_kin:       (N, 150, 15)
    """
    if target_stream in ["wrist", "wirst", "elbow", "shoulder"]:
        key = TRAJ_KEY_ALIAS[target_stream]
        return traj[key]

    if target_stream == "imu":
        return traj["IMU"]

    if target_stream == "all_xyz":
        return np.concatenate(
            [traj["shoulder"], traj["elbow"], traj["wirst"]],
            axis=-1,
        )

    if target_stream == "all_kin":
        return np.concatenate(
            [traj["shoulder"], traj["elbow"], traj["wirst"], traj["IMU"]],
            axis=-1,
        )

    raise ValueError(f"Unknown TARGET_STREAM: {target_stream}")


def build_target_sequence(traj: dict, target_stream: str, labels_clean: np.ndarray):
    """
    Build target sequence Y.

    New direct trajectory format:
        The trajectory is already aligned with EEG epoch order.
        Example:
            traj["shoulder"].shape = (N, 150, 3)
            traj["elbow"].shape    = (N, 150, 3)
            traj["wirst"].shape    = (N, 150, 3)
            traj["IMU"].shape      = (N, 150, 6)

        In this format, labels_clean is NOT used to select trajectory epochs.
        labels_clean is used only to create target_ids for stratified train/val/test split
        and prototype alignment labels.

    Old nested format:
        The trajectory is grouped by "Number 1".."Number 4".
        In that case, labels_clean is used to pick the next available trajectory epoch
        from the matching target group.
    """
    labels_clean = np.asarray(labels_clean).astype(np.int32).ravel()
    dim_names = _make_dim_names(target_stream)

    # New direct epoch-aligned trajectory format
    if _is_direct_trajectory_format(traj):
        y = _get_direct_target_array(traj, target_stream).astype(np.float32)

        target_ids = _target_ids_from_labels(
            labels_clean=labels_clean,
            n_epochs=y.shape[0],
            name="direct trajectory target",
        )

        return y, target_ids, dim_names

    # Old nested Number 1-4 format
    _check_label_counts_against_trajectory(labels_clean, traj, reference_key="wirst")

    from collections import defaultdict
    counters = defaultdict(int)
    y_list = []
    target_ids = []

    for lab in labels_clean:
        lab = int(lab)
        target_key = LABEL_TO_KEY[lab]
        idx = counters[target_key]

        y_epoch = _get_target_epoch(traj, target_stream, target_key, idx)
        y_list.append(y_epoch.astype(np.float32))
        target_ids.append(KEY_TO_TARGET_ID[target_key])

        counters[target_key] += 1

    y = np.stack(y_list, axis=0).astype(np.float32)
    target_ids = np.asarray(target_ids, dtype=np.int64)

    return y, target_ids, dim_names


def build_imu_sequence(traj: dict, labels_clean: np.ndarray):
    """
    Build ME IMU input sequence.

    New direct format:
        traj["IMU"] is already epoch-aligned with EEG.
        Output: (N, 150, 6)

    Old nested format:
        traj["IMU"]["Number 1"] etc.
        labels_clean is used to align the order.
    """
    if "IMU" not in traj:
        raise KeyError("Trajectory file does not contain key 'IMU'.")

    labels_clean = np.asarray(labels_clean).astype(np.int32).ravel()

    # New direct epoch-aligned trajectory format
    if _is_direct_trajectory_format(traj):
        imu = traj["IMU"].astype(np.float32)

        # Validate label length for splitting / prototype target IDs
        _ = _target_ids_from_labels(
            labels_clean=labels_clean,
            n_epochs=imu.shape[0],
            name="direct IMU trajectory",
        )

        return imu

    # Old nested Number 1-4 format
    _check_label_counts_against_trajectory(labels_clean, traj, reference_key="IMU")

    from collections import defaultdict
    counters = defaultdict(int)
    imu_list = []

    for lab in labels_clean:
        target_key = LABEL_TO_KEY[int(lab)]
        idx = counters[target_key]
        imu_epoch = traj["IMU"][target_key][idx]
        imu_list.append(imu_epoch.astype(np.float32))
        counters[target_key] += 1

    return np.stack(imu_list, axis=0).astype(np.float32)


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


# ============================================================
# 4) SPLIT AND TARGET SCALING
# ============================================================

def stratified_split_indices(target_ids: np.ndarray, seed: int = 42):
    indices = np.arange(len(target_ids))
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=0.15,
        random_state=seed,
        stratify=target_ids,
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.1765,
        random_state=seed,
        stratify=target_ids[trainval_idx],
    )
    return train_idx, val_idx, test_idx


def fit_target_scaler(*y_train_arrays, eps: float = 1e-8):
    y_cat = np.concatenate(y_train_arrays, axis=0)
    mean = y_cat.mean(axis=(0, 1), keepdims=True)
    std = y_cat.std(axis=(0, 1), keepdims=True)
    return {"mean": mean.astype(np.float32), "std": (std + eps).astype(np.float32)}


def normalize_targets(y: np.ndarray, scaler: dict):
    return ((y - scaler["mean"]) / scaler["std"]).astype(np.float32)


def denormalize_targets(y_norm: np.ndarray, scaler: dict):
    return y_norm * scaler["std"] + scaler["mean"]


# ============================================================
# 5) DATASETS
# ============================================================

class EEGTrajectoryDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, target_id: np.ndarray, subject_id: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.target_id = torch.from_numpy(target_id).long()
        self.subject_id = torch.from_numpy(subject_id).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.target_id[idx], self.subject_id[idx]


class EEGIMUTrajectoryDataset(Dataset):
    def __init__(self, X: np.ndarray, imu: np.ndarray, y: np.ndarray, target_id: np.ndarray, subject_id: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.imu = torch.from_numpy(imu).float()
        self.y = torch.from_numpy(y).float()
        self.target_id = torch.from_numpy(target_id).long()
        self.subject_id = torch.from_numpy(subject_id).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.imu[idx], self.y[idx], self.target_id[idx], self.subject_id[idx]


def make_eeg_loader(X, y, ids, subject_ids, batch_size, shuffle):
    ds = EEGTrajectoryDataset(X, y, ids, subject_ids)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def make_eeg_imu_loader(X, imu, y, ids, subject_ids, batch_size, shuffle):
    ds = EEGIMUTrajectoryDataset(X, imu, y, ids, subject_ids)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ============================================================
# 6) MODEL BLOCKS
# ============================================================

class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.15):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class EEGEncoder(nn.Module):
    def __init__(
        self,
        n_bands: int,
        n_channels: int,
        target_len: int = 150,
        temporal_filters: int = 12,
        spatial_filters: int = 32,
        latent_dim: int = 64,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.target_len = target_len

        self.band_temporal = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, temporal_filters, kernel_size=(1, 31), padding=(0, 15), bias=False),
                nn.BatchNorm2d(temporal_filters),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
                nn.Dropout(dropout),
                nn.Conv2d(temporal_filters, temporal_filters, kernel_size=(1, 15), padding=(0, 7), bias=False),
                nn.BatchNorm2d(temporal_filters),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(dropout),
            )
            for _ in range(n_bands)
        ])

        self.spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=temporal_filters * n_bands,
                out_channels=spatial_filters,
                kernel_size=(n_channels, 1),
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.proj = nn.Sequential(
            nn.Conv1d(spatial_filters, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.tcn = nn.Sequential(
            TCNBlock(latent_dim, kernel_size=3, dilation=1, dropout=dropout),
            TCNBlock(latent_dim, kernel_size=3, dilation=2, dropout=dropout),
            TCNBlock(latent_dim, kernel_size=3, dilation=4, dropout=dropout),
            TCNBlock(latent_dim, kernel_size=3, dilation=8, dropout=dropout),
        )

    def forward(self, x):
        band_feats = []
        for b in range(self.n_bands):
            xb = x[:, b:b + 1, :, :]
            fb = self.band_temporal[b](xb)
            band_feats.append(fb)

        h = torch.cat(band_feats, dim=1)
        h = self.spatial(h)
        h = h.squeeze(2)
        h = self.proj(h)
        h = self.tcn(h)
        h = F.interpolate(h, size=self.target_len, mode="linear", align_corners=False)
        return h


class IMUEncoder(nn.Module):
    def __init__(self, in_dim: int = 6, latent_dim: int = 64, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, latent_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            TCNBlock(latent_dim, kernel_size=3, dilation=1, dropout=dropout),
            TCNBlock(latent_dim, kernel_size=3, dilation=2, dropout=dropout),
        )

    def forward(self, imu):
        x = imu.permute(0, 2, 1).contiguous()
        return self.net(x)


class SeqHead(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim, 64, kernel_size=1),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, out_dim, kernel_size=1),
        )

    def forward(self, h):
        y = self.net(h)
        return y.permute(0, 2, 1).contiguous()


class EEGOnlyModel(nn.Module):
    def __init__(self, n_bands, n_channels, out_dim, target_len=150, latent_dim=64, dropout=0.20):
        super().__init__()
        self.eeg_encoder = EEGEncoder(n_bands, n_channels, target_len, latent_dim=latent_dim, dropout=dropout)
        self.head = SeqHead(latent_dim, out_dim, dropout)

    def forward(self, eeg):
        h = self.eeg_encoder(eeg)
        y = self.head(h)
        return y


class SimpleFusionModel(nn.Module):
    """
    Baseline 3 model:
    ME: EEG + IMU -> simple fusion -> trajectory
    MI: EEG + zero IMU -> same model -> trajectory
    No student distillation, no prototype alignment.
    """
    def __init__(self, n_bands, n_channels, out_dim, target_len=150, latent_dim=64, dropout=0.20):
        super().__init__()
        self.eeg_encoder = EEGEncoder(n_bands, n_channels, target_len, latent_dim=latent_dim, dropout=dropout)
        self.imu_encoder = IMUEncoder(6, latent_dim, dropout)
        self.fusion = nn.Sequential(
            nn.Conv1d(latent_dim * 2, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            TCNBlock(latent_dim, kernel_size=3, dilation=1, dropout=dropout),
        )
        self.head = SeqHead(latent_dim, out_dim, dropout)

    def forward(self, eeg, imu):
        h_eeg = self.eeg_encoder(eeg)
        h_imu = self.imu_encoder(imu)
        h = self.fusion(torch.cat([h_eeg, h_imu], dim=1))
        return self.head(h)


class BidirectionalCrossModalAttentionFusion(nn.Module):
    """
    Bidirectional Cross-Modal Multi-Head Attention fusion for EEG and IMU features.

    Input tensors:
        h_eeg: (batch, latent_dim, time)
        h_imu: (batch, latent_dim, time)

    Fusion logic:
        1) EEG queries IMU: Q=EEG, K/V=IMU
        2) IMU queries EEG: Q=IMU, K/V=EEG
        3) Concatenate original and cross-attended features
        4) Project back to latent_dim and keep an EEG residual path

    The residual EEG path is important for Baseline 6 because MI fine-tuning/testing
    uses zero IMU, so the model should not become completely dependent on IMU.
    """
    def __init__(self, latent_dim: int, num_heads: int = 4, dropout: float = 0.20):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.eeg_to_imu = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.imu_to_eeg = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.eeg_norm = nn.LayerNorm(latent_dim)
        self.imu_norm = nn.LayerNorm(latent_dim)
        self.eeg_cross_norm = nn.LayerNorm(latent_dim)
        self.imu_cross_norm = nn.LayerNorm(latent_dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.fusion_proj = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.out = nn.Sequential(
            TCNBlock(latent_dim, kernel_size=3, dilation=1, dropout=dropout),
        )

    def forward(self, h_eeg, h_imu):
        # Convert from Conv1D format (B, D, T) to token format (B, T, D).
        eeg_tokens = h_eeg.permute(0, 2, 1).contiguous()
        imu_tokens = h_imu.permute(0, 2, 1).contiguous()

        eeg_q = self.eeg_norm(eeg_tokens)
        imu_q = self.imu_norm(imu_tokens)

        # EEG attends to IMU: use EEG as query and IMU as key/value.
        eeg_from_imu, _ = self.eeg_to_imu(
            query=eeg_q,
            key=imu_q,
            value=imu_q,
            need_weights=False,
        )
        eeg_from_imu = self.eeg_cross_norm(eeg_tokens + self.attn_dropout(eeg_from_imu))

        # IMU attends to EEG: use IMU as query and EEG as key/value.
        imu_from_eeg, _ = self.imu_to_eeg(
            query=imu_q,
            key=eeg_q,
            value=eeg_q,
            need_weights=False,
        )
        imu_from_eeg = self.imu_cross_norm(imu_tokens + self.attn_dropout(imu_from_eeg))

        # Fuse both original and cross-modal context features.
        fused_tokens = torch.cat(
            [eeg_tokens, imu_tokens, eeg_from_imu, imu_from_eeg],
            dim=-1,
        )
        fused_tokens = self.fusion_proj(fused_tokens)

        # EEG residual path helps preserve EEG-only decoding ability for MI with zero IMU.
        fused_tokens = fused_tokens + eeg_tokens

        # Convert back to Conv1D format (B, D, T), then refine temporally.
        fused = fused_tokens.permute(0, 2, 1).contiguous()
        return self.out(fused)


class CrossModalAttentionFusionModel(nn.Module):
    """
    Baseline 6 model:
    ME: EEG + IMU -> bidirectional cross-modal MHA fusion -> trajectory
    MI: EEG + zero IMU -> same model -> trajectory
    No student distillation, no prototype alignment.
    """
    def __init__(
        self,
        n_bands,
        n_channels,
        out_dim,
        target_len=150,
        latent_dim=64,
        num_heads=4,
        dropout=0.20,
    ):
        super().__init__()
        self.eeg_encoder = EEGEncoder(n_bands, n_channels, target_len, latent_dim=latent_dim, dropout=dropout)
        self.imu_encoder = IMUEncoder(6, latent_dim, dropout)
        self.fusion = BidirectionalCrossModalAttentionFusion(
            latent_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.head = SeqHead(latent_dim, out_dim, dropout)

    def forward(self, eeg, imu):
        h_eeg = self.eeg_encoder(eeg)
        h_imu = self.imu_encoder(imu)
        h = self.fusion(h_eeg, h_imu)
        return self.head(h)



class NormalMultiHeadAttentionFusion(nn.Module):
    """
    Normal Multi-Head Attention fusion for EEG and IMU features.

    Input tensors:
        h_eeg: (batch, latent_dim, time)
        h_imu: (batch, latent_dim, time)

    Fusion logic:
        1) Concatenate EEG and IMU latent features along the channel dimension
        2) Project concatenated features back to latent_dim
        3) Apply normal multi-head self-attention with Q=K=V=fused tokens
        4) Refine with a small feed-forward block and TCNBlock

    Difference from Baseline 6:
        - Baseline 6 uses cross-modal attention: EEG queries IMU and IMU queries EEG.
        - Baseline 7 uses normal self-attention after initial EEG+IMU feature fusion.

    The EEG residual path helps preserve an EEG-only route when MI uses zero IMU.
    """
    def __init__(self, latent_dim: int, num_heads: int = 4, dropout: float = 0.20):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.input_proj = nn.Sequential(
            nn.Conv1d(latent_dim * 2, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_norm = nn.LayerNorm(latent_dim)
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Dropout(dropout),
        )

        self.out = nn.Sequential(
            TCNBlock(latent_dim, kernel_size=3, dilation=1, dropout=dropout),
        )

    def forward(self, h_eeg, h_imu):
        # First make one fused latent stream in Conv1D format: (B, D, T).
        h0 = self.input_proj(torch.cat([h_eeg, h_imu], dim=1))

        # Convert to token format for PyTorch MultiheadAttention: (B, T, D).
        tokens = h0.permute(0, 2, 1).contiguous()

        # Normal MHA: Q, K, and V all come from the same fused EEG+IMU tokens.
        attn_in = self.attn_norm(tokens)
        attn_out, _ = self.self_attn(
            query=attn_in,
            key=attn_in,
            value=attn_in,
            need_weights=False,
        )
        tokens = tokens + self.attn_dropout(attn_out)

        # Small Transformer-style feed-forward refinement.
        ffn_in = self.ffn_norm(tokens)
        tokens = tokens + self.ffn(ffn_in)

        # EEG residual path helps MI fine-tuning/testing with zero IMU.
        eeg_tokens = h_eeg.permute(0, 2, 1).contiguous()
        tokens = tokens + eeg_tokens

        # Convert back to Conv1D format and refine temporally.
        fused = tokens.permute(0, 2, 1).contiguous()
        return self.out(fused)


class NormalMHAFusionModel(nn.Module):
    """
    Baseline 7 model:
    ME: EEG + IMU -> normal MHA fusion over fused EEG+IMU tokens -> trajectory
    MI: EEG + zero IMU -> same model -> trajectory
    No student distillation, no prototype alignment.
    """
    def __init__(
        self,
        n_bands,
        n_channels,
        out_dim,
        target_len=150,
        latent_dim=64,
        num_heads=4,
        dropout=0.20,
    ):
        super().__init__()
        self.eeg_encoder = EEGEncoder(n_bands, n_channels, target_len, latent_dim=latent_dim, dropout=dropout)
        self.imu_encoder = IMUEncoder(6, latent_dim, dropout)
        self.fusion = NormalMultiHeadAttentionFusion(
            latent_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.head = SeqHead(latent_dim, out_dim, dropout)

    def forward(self, eeg, imu):
        h_eeg = self.eeg_encoder(eeg)
        h_imu = self.imu_encoder(imu)
        h = self.fusion(h_eeg, h_imu)
        return self.head(h)


class DistillationSimpleModel(nn.Module):
    """
    Baseline 4:
    Teacher: EEG+IMU simple fusion
    Student: EEG only
    """
    def __init__(self, n_bands, n_channels, out_dim, target_len=150, latent_dim=64, dropout=0.20):
        super().__init__()
        self.eeg_encoder = EEGEncoder(n_bands, n_channels, target_len, latent_dim=latent_dim, dropout=dropout)
        self.imu_encoder = IMUEncoder(6, latent_dim, dropout)
        self.fusion = nn.Sequential(
            nn.Conv1d(latent_dim * 2, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            TCNBlock(latent_dim, kernel_size=3, dilation=1, dropout=dropout),
        )
        self.teacher_head = SeqHead(latent_dim, out_dim, dropout)
        self.student_head = SeqHead(latent_dim, out_dim, dropout)

    def forward_teacher_student(self, eeg, imu):
        h_eeg = self.eeg_encoder(eeg)
        h_imu = self.imu_encoder(imu)
        h_teacher = self.fusion(torch.cat([h_eeg, h_imu], dim=1))
        y_teacher = self.teacher_head(h_teacher)
        y_student = self.student_head(h_eeg)
        return y_teacher, y_student

    def forward_student(self, eeg):
        h_eeg = self.eeg_encoder(eeg)
        return self.student_head(h_eeg)


class SubjectFiLM(nn.Module):
    def __init__(self, n_subjects: int, latent_dim: int):
        super().__init__()
        self.gamma = nn.Embedding(n_subjects, latent_dim)
        self.beta = nn.Embedding(n_subjects, latent_dim)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, h, subject_id):
        gamma = self.gamma(subject_id).unsqueeze(-1)
        beta = self.beta(subject_id).unsqueeze(-1)
        return gamma * h + beta


class LagAwareFusion(nn.Module):
    def __init__(self, latent_dim: int, lag_radius: int = 3, dropout: float = 0.20):
        super().__init__()
        self.latent_dim = latent_dim
        self.lag_radius = lag_radius
        self.lags = list(range(-lag_radius, lag_radius + 1))

        self.q_proj = nn.Conv1d(latent_dim, latent_dim, kernel_size=1)
        self.k_proj = nn.Conv1d(latent_dim, latent_dim, kernel_size=1)
        self.v_proj = nn.Conv1d(latent_dim, latent_dim, kernel_size=1)

        self.context_proj = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.gate = nn.Sequential(
            nn.Conv1d(latent_dim * 3, latent_dim, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(latent_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.out = nn.Sequential(
            TCNBlock(latent_dim, kernel_size=3, dilation=1, dropout=dropout),
        )

    @staticmethod
    def shift_time(x, lag: int):
        if lag == 0:
            return x
        out = torch.zeros_like(x)
        if lag > 0:
            out[:, :, :-lag] = x[:, :, lag:]
        else:
            lag_abs = -lag
            out[:, :, lag_abs:] = x[:, :, :-lag_abs]
        return out

    def forward(self, h_eeg, h_imu):
        D = h_eeg.shape[1]
        q = self.q_proj(h_eeg)
        k = self.k_proj(h_imu)
        v = self.v_proj(h_imu)

        k_stack = torch.stack([self.shift_time(k, lag) for lag in self.lags], dim=1)
        v_stack = torch.stack([self.shift_time(v, lag) for lag in self.lags], dim=1)

        scores = (q.unsqueeze(1) * k_stack).sum(dim=2) / math.sqrt(D)
        alpha = torch.softmax(scores, dim=1)
        context = (alpha.unsqueeze(2) * v_stack).sum(dim=1)
        context = self.context_proj(context)

        gate_in = torch.cat([h_eeg, h_imu, context], dim=1)
        g = self.gate(gate_in)
        h_fused = h_eeg + g * context
        return self.out(h_fused)


class FullPrivilegedModel(nn.Module):
    def __init__(
        self,
        n_bands: int,
        n_channels: int,
        out_dim: int,
        n_subjects: int = 1,
        target_len: int = 150,
        latent_dim: int = 64,
        lag_radius: int = 3,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.eeg_encoder = EEGEncoder(n_bands, n_channels, target_len, latent_dim=latent_dim, dropout=dropout)
        self.subject_film = SubjectFiLM(n_subjects, latent_dim)
        self.imu_encoder = IMUEncoder(6, latent_dim, dropout)
        self.fusion = LagAwareFusion(latent_dim, lag_radius, dropout)
        self.teacher_head = SeqHead(latent_dim, out_dim, dropout)
        self.student_head = SeqHead(latent_dim, out_dim, dropout)

    def encode_eeg_student(self, eeg, subject_id):
        h = self.eeg_encoder(eeg)
        h = self.subject_film(h, subject_id)
        return h

    def forward_teacher_student(self, eeg, imu, subject_id):
        h_eeg = self.encode_eeg_student(eeg, subject_id)
        h_imu = self.imu_encoder(imu)
        h_teacher = self.fusion(h_eeg, h_imu)
        y_teacher = self.teacher_head(h_teacher)
        y_student = self.student_head(h_eeg)
        return y_teacher, y_student, h_eeg, h_teacher

    def forward_student(self, eeg, subject_id, return_latent: bool = False):
        h_eeg = self.encode_eeg_student(eeg, subject_id)
        y_student = self.student_head(h_eeg)
        if return_latent:
            return y_student, h_eeg
        return y_student


def set_full_student_finetune_mode(model: FullPrivilegedModel):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.eeg_encoder.parameters():
        p.requires_grad = True
    for p in model.subject_film.parameters():
        p.requires_grad = True
    for p in model.student_head.parameters():
        p.requires_grad = True


def set_distill_student_finetune_mode(model: DistillationSimpleModel):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.eeg_encoder.parameters():
        p.requires_grad = True
    for p in model.student_head.parameters():
        p.requires_grad = True


# ============================================================
# 7) LOSSES AND METRICS
# ============================================================

def sequence_corrcoef(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8):
    yp = y_pred.reshape(-1, y_pred.shape[-1])
    yt = y_true.reshape(-1, y_true.shape[-1])
    yp = yp - yp.mean(dim=0, keepdim=True)
    yt = yt - yt.mean(dim=0, keepdim=True)
    num = (yp * yt).sum(dim=0)
    den = torch.sqrt((yp ** 2).sum(dim=0) * (yt ** 2).sum(dim=0) + eps)
    return (num / (den + eps)).mean()


class SequenceRegressionLoss(nn.Module):
    def __init__(self, corr_weight: float = 0.10):
        super().__init__()
        self.corr_weight = corr_weight
        self.huber = nn.SmoothL1Loss(beta=0.5)

    def forward(self, y_pred, y_true):
        huber = self.huber(y_pred, y_true)
        corr = sequence_corrcoef(y_pred, y_true)
        return huber + self.corr_weight * (1.0 - corr)


def proto_alignment_loss(h_eeg, target_id, prototypes):
    z = h_eeg.mean(dim=-1)
    z = F.normalize(z, dim=-1)
    p = prototypes[target_id]
    p = F.normalize(p, dim=-1)
    return (1.0 - (z * p).sum(dim=-1)).mean()


def compute_metrics_np(y_true: np.ndarray, y_pred: np.ndarray, dim_names: List[str]):
    yt = y_true.reshape(-1, y_true.shape[-1])
    yp = y_pred.reshape(-1, y_pred.shape[-1])

    rmse = np.sqrt(np.mean((yt - yp) ** 2, axis=0))
    mae = np.mean(np.abs(yt - yp), axis=0)

    r_list = []
    for d in range(yt.shape[1]):
        if np.std(yt[:, d]) < 1e-9 or np.std(yp[:, d]) < 1e-9:
            r = 0.0
        else:
            r = float(np.corrcoef(yt[:, d], yp[:, d])[0, 1])
        r_list.append(r)

    metrics = {
        "r_mean": float(np.mean(r_list)),
        "rmse_mean": float(np.mean(rmse)),
        "mae_mean": float(np.mean(mae)),
    }

    for name, r, e, a in zip(dim_names, r_list, rmse, mae):
        metrics[f"r_{name}"] = float(r)
        metrics[f"rmse_{name}"] = float(e)
        metrics[f"mae_{name}"] = float(a)

    return metrics


def long_axis_rows(baseline_name, metrics, dim_names, evaluation_task="MI_test_final", branch="final"):
    rows = []
    for name in dim_names:
        rows.append({
            "subject": subject,
            "baseline": baseline_name,
            "evaluation_task": evaluation_task,
            "branch": branch,
            "axis": name,
            "r": metrics.get(f"r_{name}", np.nan),
            "rmse": metrics.get(f"rmse_{name}", np.nan),
            "mae": metrics.get(f"mae_{name}", np.nan),
        })
    return rows


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


# ============================================================
# 10) PREPARE DATA
# ============================================================

def prepare_me_dataset():
    eeg, traj = load_eeg_and_trajectory(ME_EEG_PATH, ME_TRAJ_PATH, "ME")
    labels_clean = load_labels_clean(ME_LABELS_PATH, "ME")

    if eeg.shape[0] != labels_clean.shape[0]:
        raise ValueError(f"ME: EEG epochs ({eeg.shape[0]}) != labels_clean length ({labels_clean.shape[0]}).")

    y, target_ids, dim_names = build_target_sequence(traj, TARGET_STREAM, labels_clean)
    imu = build_imu_sequence(traj, labels_clean)

    if eeg.shape[0] != y.shape[0]:
        raise ValueError(f"ME: EEG epochs ({eeg.shape[0]}) != target epochs ({y.shape[0]}).")
    if imu.shape[0] != y.shape[0]:
        raise ValueError(f"ME: IMU epochs ({imu.shape[0]}) != target epochs ({y.shape[0]}).")

    print("\nME after construction:")
    print("  EEG full epoch:", eeg.shape)
    print("  IMU:", imu.shape)
    print("  y:", y.shape)
    print("  target_ids:", target_ids.shape, np.bincount(target_ids, minlength=4))
    print("  dim_names:", dim_names)

    train_idx, val_idx, test_idx = stratified_split_indices(target_ids, SEED)

    print("\nME split:")
    print("  train:", len(train_idx), np.bincount(target_ids[train_idx], minlength=4))
    print("  val:  ", len(val_idx), np.bincount(target_ids[val_idx], minlength=4))
    print("  test: ", len(test_idx), np.bincount(target_ids[test_idx], minlength=4))

    X = make_eeg_input(eeg, EEG_FS, use_filterbank=USE_FILTERBANK)
    print("  EEG model input X:", X.shape)

    imu_train_n, imu_val_n, imu_test_n, imu_scaler = normalize_imu_train_only(
        imu[train_idx],
        imu[val_idx],
        imu[test_idx],
    )

    subject_ids = np.zeros(len(target_ids), dtype=np.int64)

    return {
        "X_train": X[train_idx],
        "X_val": X[val_idx],
        "X_test": X[test_idx],
        "imu_train": imu_train_n,
        "imu_val": imu_val_n,
        "imu_test": imu_test_n,
        "imu_scaler": imu_scaler,
        "y_train": y[train_idx],
        "y_val": y[val_idx],
        "y_test": y[test_idx],
        "id_train": target_ids[train_idx],
        "id_val": target_ids[val_idx],
        "id_test": target_ids[test_idx],
        "sub_train": subject_ids[train_idx],
        "sub_val": subject_ids[val_idx],
        "sub_test": subject_ids[test_idx],
        "dim_names": dim_names,
        "n_bands": X.shape[1],
        "n_channels": X.shape[2],
        "out_dim": y.shape[-1],
    }


def prepare_mi_dataset():
    eeg, traj = load_eeg_and_trajectory(MI_EEG_PATH, MI_TRAJ_PATH, "MI")
    labels_clean = load_labels_clean(MI_LABELS_PATH, "MI")

    if eeg.shape[0] != labels_clean.shape[0]:
        raise ValueError(f"MI: EEG epochs ({eeg.shape[0]}) != labels_clean length ({labels_clean.shape[0]}).")

    y, target_ids, dim_names = build_target_sequence(traj, TARGET_STREAM, labels_clean)

    if eeg.shape[0] != y.shape[0]:
        raise ValueError(f"MI: EEG epochs ({eeg.shape[0]}) != target epochs ({y.shape[0]}).")

    print("\nMI after construction:")
    print("  EEG full epoch:", eeg.shape)
    print("  y:", y.shape)
    print("  target_ids:", target_ids.shape, np.bincount(target_ids, minlength=4))
    print("  dim_names:", dim_names)

    train_idx, val_idx, test_idx = stratified_split_indices(target_ids, SEED)

    print("\nMI split:")
    print("  train:", len(train_idx), np.bincount(target_ids[train_idx], minlength=4))
    print("  val:  ", len(val_idx), np.bincount(target_ids[val_idx], minlength=4))
    print("  test: ", len(test_idx), np.bincount(target_ids[test_idx], minlength=4))

    X = make_eeg_input(eeg, EEG_FS, use_filterbank=USE_FILTERBANK)
    print("  EEG model input X:", X.shape)

    subject_ids = np.zeros(len(target_ids), dtype=np.int64)

    return {
        "X_train": X[train_idx],
        "X_val": X[val_idx],
        "X_test": X[test_idx],
        "y_train": y[train_idx],
        "y_val": y[val_idx],
        "y_test": y[test_idx],
        "id_train": target_ids[train_idx],
        "id_val": target_ids[val_idx],
        "id_test": target_ids[test_idx],
        "sub_train": subject_ids[train_idx],
        "sub_val": subject_ids[val_idx],
        "sub_test": subject_ids[test_idx],
        "dim_names": dim_names,
        "n_bands": X.shape[1],
        "n_channels": X.shape[2],
        "out_dim": y.shape[-1],
    }


# ============================================================
# 11) EXPORT RESULTS
# ============================================================

def summarize_baseline(
    baseline_id,
    baseline_name,
    metrics,
    notes="",
    evaluation_task="MI_test_final",
    branch="final",
):
    row = {
        "subject": subject,
        "baseline_id": baseline_id,
        "baseline": baseline_name,
        "evaluation_task": evaluation_task,
        "branch": branch,
        "target_stream": TARGET_STREAM,
        "test_r_mean": metrics.get("r_mean", np.nan),
        "test_rmse_mean": metrics.get("rmse_mean", np.nan),
        "test_mae_mean": metrics.get("mae_mean", np.nan),
        "test_loss": metrics.get("loss", np.nan),
    }

    # Add per-axis correlation columns directly to the summary sheet.
    # Example columns:
    #   r_shoulder_Vx, r_shoulder_Vy, r_shoulder_Vz,
    #   r_elbow_Vx,    r_elbow_Vy,    r_elbow_Vz,
    #   r_wrist_Vx,    r_wrist_Vy,    r_wrist_Vz
    for k, v in metrics.items():
        if k.startswith("r_") and k != "r_mean":
            row[k] = v

    row["notes"] = notes
    return row


def save_results_to_excel(summary_rows, axis_rows, stage_rows, dim_names):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to save .xlsx results. Please install pandas and openpyxl.") from exc

    summary_df = pd.DataFrame(summary_rows)
    axis_df = pd.DataFrame(axis_rows)
    stages_df = pd.DataFrame(stage_rows)

    config_rows = [
        ("subject", subject),
        ("target_stream", TARGET_STREAM),
        ("dim_names", ", ".join(dim_names)),
        ("eeg_fs", EEG_FS),
        ("traj_fs", TRAJ_FS),
        ("input_epoch", f"{EEG_EPOCH_TMIN} to 7.0 s assumed"),
        ("model_crop", f"{EEG_CROP_TMIN} to {EEG_CROP_TMAX} s"),
        ("model_eeg_samples", MODEL_EEG_SAMPLES),
        ("bands", str(BANDS)),
        ("batch_size", BATCH_SIZE),
        ("pretrain_epochs", PRETRAIN_EPOCHS),
        ("finetune_epochs", FINETUNE_EPOCHS),
        ("seed", SEED),
        ("baseline0_lags_sec", str(LINEAR_LAGS_SEC)),
        ("baseline0_lag_window_sec", LINEAR_LAG_WINDOW_SEC),
        ("baseline0_lag_window_ms", LINEAR_LAG_WINDOW_SEC * 1000),
        ("baseline0_use_ridge", BASELINE0_USE_RIDGE),
        ("baseline0_ridge_alpha", BASELINE0_RIDGE_ALPHA),
        ("cross_attention_heads", CROSS_ATTENTION_HEADS),
        ("normal_mha_heads", NORMAL_MHA_HEADS),
        ("kd_loss_weight", KD_LOSS_WEIGHT),
        ("proto_loss_weight", PROTO_LOSS_WEIGHT),
    ]
    config_df = pd.DataFrame(config_rows, columns=["parameter", "value"])

    with pd.ExcelWriter(RESULT_XLSX_PATH, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        axis_df.to_excel(writer, sheet_name="axis_metrics", index=False)
        stages_df.to_excel(writer, sheet_name="stages", index=False)
        config_df.to_excel(writer, sheet_name="config", index=False)

        # Basic formatting if openpyxl is available through the writer.
        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col:
                    value = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, min(len(value), 50))
                ws.column_dimensions[col_letter].width = max(10, max_len + 2)

    print(f"\nSaved Excel results to:\n{RESULT_XLSX_PATH}")


# ============================================================
# 12) MAIN
# ============================================================

def main():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    me = prepare_me_dataset()
    mi = prepare_mi_dataset()

    if me["n_bands"] != mi["n_bands"]:
        raise ValueError("ME and MI have different number of filter-bank bands.")
    if me["n_channels"] != mi["n_channels"]:
        raise ValueError("ME and MI have different number of EEG channels.")
    if me["out_dim"] != mi["out_dim"]:
        raise ValueError("ME and MI targets have different output dimensions.")

    dim_names = me["dim_names"]
    if me["dim_names"] != mi["dim_names"]:
        print("Warning: ME and MI dim_names are different.")
        print("ME:", me["dim_names"])
        print("MI:", mi["dim_names"])

    scaler = fit_target_scaler(me["y_train"], mi["y_train"])

    for d in [me, mi]:
        d["y_train_n"] = normalize_targets(d["y_train"], scaler)
        d["y_val_n"] = normalize_targets(d["y_val"], scaler)
        d["y_test_n"] = normalize_targets(d["y_test"], scaler)

    me_train_loader = make_eeg_imu_loader(me["X_train"], me["imu_train"], me["y_train_n"], me["id_train"], me["sub_train"], BATCH_SIZE, True)
    me_val_loader = make_eeg_imu_loader(me["X_val"], me["imu_val"], me["y_val_n"], me["id_val"], me["sub_val"], BATCH_SIZE, False)
    me_test_loader = make_eeg_imu_loader(me["X_test"], me["imu_test"], me["y_test_n"], me["id_test"], me["sub_test"], BATCH_SIZE, False)

    me_eeg_train_loader = make_eeg_loader(me["X_train"], me["y_train_n"], me["id_train"], me["sub_train"], BATCH_SIZE, True)
    me_eeg_val_loader = make_eeg_loader(me["X_val"], me["y_val_n"], me["id_val"], me["sub_val"], BATCH_SIZE, False)
    me_eeg_test_loader = make_eeg_loader(me["X_test"], me["y_test_n"], me["id_test"], me["sub_test"], BATCH_SIZE, False)

    mi_train_loader = make_eeg_loader(mi["X_train"], mi["y_train_n"], mi["id_train"], mi["sub_train"], BATCH_SIZE, True)
    mi_val_loader = make_eeg_loader(mi["X_val"], mi["y_val_n"], mi["id_val"], mi["sub_val"], BATCH_SIZE, False)
    mi_test_loader = make_eeg_loader(mi["X_test"], mi["y_test_n"], mi["id_test"], mi["sub_test"], BATCH_SIZE, False)

    pretrain_cfg = TrainConfig(PRETRAIN_LR, WEIGHT_DECAY, PRETRAIN_EPOCHS, PATIENCE_PRETRAIN, device)
    finetune_cfg = TrainConfig(FINETUNE_LR, WEIGHT_DECAY, FINETUNE_EPOCHS, PATIENCE_FINETUNE, device)

    criterion = SequenceRegressionLoss(corr_weight=0.10)

    summary_rows = []
    axis_rows = []
    stage_rows_all = []

    # ------------------------------------------------------------
    # Baseline 0: Time-lagged MLR
    # ------------------------------------------------------------
    if RUN_BASELINE_0:
        set_seed(SEED)
        name = "Baseline 0: Time-lagged MLR"

        metrics_mi = run_baseline0_linear(mi, scaler, dim_names)
        print(f"\n{name} MI TEST: r_mean={metrics_mi['r_mean']:.4f}, RMSE={metrics_mi['rmse_mean']:.4f}, MAE={metrics_mi['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            0, name, metrics_mi,
            "MI train+val -> MI test; time-lagged ridge-stabilized MLR",
            evaluation_task="MI_test",
            branch="linear",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_mi, dim_names, evaluation_task="MI_test", branch="linear"))

        metrics_me = run_baseline0_linear(me, scaler, dim_names)
        print(f"\n{name} ME TEST: r_mean={metrics_me['r_mean']:.4f}, RMSE={metrics_me['rmse_mean']:.4f}, MAE={metrics_me['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            0, name, metrics_me,
            "ME train+val -> ME test; time-lagged ridge-stabilized MLR",
            evaluation_task="ME_test",
            branch="linear",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me, dim_names, evaluation_task="ME_test", branch="linear"))

    # ------------------------------------------------------------
    # Baseline 1: MI-only EEG
    # ------------------------------------------------------------
    if RUN_BASELINE_1:
        set_seed(SEED)
        name = "Baseline 1: MI-only EEG"
        model = EEGOnlyModel(me["n_bands"], me["n_channels"], me["out_dim"], TARGET_LEN, LATENT_DIM).to(device)
        model, rows = train_eeg_model(name, model, mi_train_loader, mi_val_loader, scaler, dim_names, finetune_cfg)
        stage_rows_all.extend(rows)

        metrics_mi = run_eeg_epoch(model, mi_test_loader, criterion, scaler, dim_names, None, device)
        print(f"\n{name} MI TEST: r_mean={metrics_mi['r_mean']:.4f}, RMSE={metrics_mi['rmse_mean']:.4f}, MAE={metrics_mi['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            1, name, metrics_mi,
            "Train only on MI EEG; evaluate on MI test",
            evaluation_task="MI_test",
            branch="EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_mi, dim_names, evaluation_task="MI_test", branch="EEG_only"))

        metrics_me = run_eeg_epoch(model, me_eeg_test_loader, criterion, scaler, dim_names, None, device)
        print(f"\n{name} ME TEST: r_mean={metrics_me['r_mean']:.4f}, RMSE={metrics_me['rmse_mean']:.4f}, MAE={metrics_me['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            1, name, metrics_me,
            "Same MI-only trained EEG model evaluated on ME test",
            evaluation_task="ME_test_final_model",
            branch="EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me, dim_names, evaluation_task="ME_test_final_model", branch="EEG_only"))

    # ------------------------------------------------------------
    # Baseline 2: ME EEG pretrain -> MI fine-tune
    # ------------------------------------------------------------
    if RUN_BASELINE_2:
        set_seed(SEED)
        name = "Baseline 2: ME EEG pretrain + MI fine-tune"
        model = EEGOnlyModel(me["n_bands"], me["n_channels"], me["out_dim"], TARGET_LEN, LATENT_DIM).to(device)

        model, rows = train_eeg_model("B2 Stage 1: ME EEG-only pretraining", model, me_eeg_train_loader, me_eeg_val_loader, scaler, dim_names, pretrain_cfg)
        stage_rows_all.extend(rows)

        metrics_me_pre = run_eeg_epoch(model, me_eeg_test_loader, criterion, scaler, dim_names, None, device)
        print(f"\n{name} ME TEST after pretraining: r_mean={metrics_me_pre['r_mean']:.4f}, RMSE={metrics_me_pre['rmse_mean']:.4f}, MAE={metrics_me_pre['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            2, name, metrics_me_pre,
            "After ME EEG-only pretraining, before MI fine-tuning",
            evaluation_task="ME_test_after_pretrain",
            branch="EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_pre, dim_names, evaluation_task="ME_test_after_pretrain", branch="EEG_only"))

        model, rows = train_eeg_model("B2 Stage 2: MI EEG-only fine-tuning", model, mi_train_loader, mi_val_loader, scaler, dim_names, finetune_cfg)
        stage_rows_all.extend(rows)

        metrics_mi = run_eeg_epoch(model, mi_test_loader, criterion, scaler, dim_names, None, device)
        print(f"\n{name} MI TEST: r_mean={metrics_mi['r_mean']:.4f}, RMSE={metrics_mi['rmse_mean']:.4f}, MAE={metrics_mi['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            2, name, metrics_mi,
            "ME EEG-only pretraining, then MI EEG-only fine-tuning; final MI test",
            evaluation_task="MI_test",
            branch="EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_mi, dim_names, evaluation_task="MI_test", branch="EEG_only"))

        metrics_me_final = run_eeg_epoch(model, me_eeg_test_loader, criterion, scaler, dim_names, None, device)
        print(f"\n{name} ME TEST after MI fine-tuning: r_mean={metrics_me_final['r_mean']:.4f}, RMSE={metrics_me_final['rmse_mean']:.4f}, MAE={metrics_me_final['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            2, name, metrics_me_final,
            "Final MI-fine-tuned model evaluated on ME test",
            evaluation_task="ME_test_after_MI_finetune",
            branch="EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_final, dim_names, evaluation_task="ME_test_after_MI_finetune", branch="EEG_only"))

    # ------------------------------------------------------------
    # Baseline 3: ME EEG+IMU pretrain, no distill/proto
    # ------------------------------------------------------------
    if RUN_BASELINE_3:
        set_seed(SEED)
        name = "Baseline 3: ME EEG+IMU pretrain, no distill/proto"
        model = SimpleFusionModel(me["n_bands"], me["n_channels"], me["out_dim"], TARGET_LEN, LATENT_DIM).to(device)

        model, rows = train_fusion_model("B3 Stage 1: ME EEG+IMU pretraining", model, me_train_loader, me_val_loader, scaler, dim_names, pretrain_cfg, use_real_imu=True)
        stage_rows_all.extend(rows)

        metrics_me_pre = run_fusion_epoch(model, me_test_loader, criterion, scaler, dim_names, None, device, use_real_imu=True)
        print(f"\n{name} ME TEST after pretraining: r_mean={metrics_me_pre['r_mean']:.4f}, RMSE={metrics_me_pre['rmse_mean']:.4f}, MAE={metrics_me_pre['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            3, name, metrics_me_pre,
            "ME EEG+IMU pretrained model evaluated on ME test using real IMU",
            evaluation_task="ME_test_after_pretrain",
            branch="EEG_IMU_fusion",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_pre, dim_names, evaluation_task="ME_test_after_pretrain", branch="EEG_IMU_fusion"))

        # Fine-tune on MI with zero IMU.
        model, rows = train_fusion_model("B3 Stage 2: MI EEG-only fine-tuning with zero IMU", model, mi_train_loader, mi_val_loader, scaler, dim_names, finetune_cfg, use_real_imu=False)
        stage_rows_all.extend(rows)

        metrics_mi = run_fusion_epoch(model, mi_test_loader, criterion, scaler, dim_names, None, device, use_real_imu=False)
        print(f"\n{name} MI TEST: r_mean={metrics_mi['r_mean']:.4f}, RMSE={metrics_mi['rmse_mean']:.4f}, MAE={metrics_mi['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            3, name, metrics_mi,
            "ME EEG+IMU pretraining; MI uses zero IMU; no distillation/prototype alignment",
            evaluation_task="MI_test",
            branch="EEG_zeroIMU_fusion",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_mi, dim_names, evaluation_task="MI_test", branch="EEG_zeroIMU_fusion"))

        metrics_me_final_zero = run_fusion_epoch(model, me_eeg_test_loader, criterion, scaler, dim_names, None, device, use_real_imu=False)
        print(f"\n{name} ME TEST after MI fine-tuning with zero IMU: r_mean={metrics_me_final_zero['r_mean']:.4f}, RMSE={metrics_me_final_zero['rmse_mean']:.4f}, MAE={metrics_me_final_zero['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            3, name, metrics_me_final_zero,
            "Final MI-fine-tuned model evaluated on ME test with zero IMU input",
            evaluation_task="ME_test_after_MI_finetune_zeroIMU",
            branch="EEG_zeroIMU_fusion",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_final_zero, dim_names, evaluation_task="ME_test_after_MI_finetune_zeroIMU", branch="EEG_zeroIMU_fusion"))

    # ------------------------------------------------------------
    # Baseline 4: ME pretrain + distillation
    # ------------------------------------------------------------
    if RUN_BASELINE_4:
        set_seed(SEED)
        name = "Baseline 4: ME distillation + MI fine-tune"
        model = DistillationSimpleModel(me["n_bands"], me["n_channels"], me["out_dim"], TARGET_LEN, LATENT_DIM).to(device)

        model, rows = train_distill_pretrain("B4 Stage 1: ME teacher-student distillation", model, me_train_loader, me_val_loader, scaler, dim_names, pretrain_cfg)
        stage_rows_all.extend(rows)

        me_pre = run_distill_pretrain_epoch(model, me_test_loader, SequenceRegressionLoss(corr_weight=0.10), nn.SmoothL1Loss(beta=0.5), scaler, dim_names, None, device)
        metrics_me_teacher = me_pre["teacher"]
        metrics_me_student = me_pre["student"]
        print(f"\n{name} ME TEST teacher after distillation: r_mean={metrics_me_teacher['r_mean']:.4f}, RMSE={metrics_me_teacher['rmse_mean']:.4f}, MAE={metrics_me_teacher['mae_mean']:.4f}")
        print(f"{name} ME TEST student after distillation: r_mean={metrics_me_student['r_mean']:.4f}, RMSE={metrics_me_student['rmse_mean']:.4f}, MAE={metrics_me_student['mae_mean']:.4f}")

        summary_rows.append(summarize_baseline(
            4, name, metrics_me_teacher,
            "ME EEG+IMU teacher evaluated on ME test after distillation pretraining",
            evaluation_task="ME_test_after_pretrain",
            branch="teacher_EEG_IMU",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_teacher, dim_names, evaluation_task="ME_test_after_pretrain", branch="teacher_EEG_IMU"))

        summary_rows.append(summarize_baseline(
            4, name, metrics_me_student,
            "EEG-only student evaluated on ME test after distillation pretraining",
            evaluation_task="ME_test_after_pretrain",
            branch="student_EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_student, dim_names, evaluation_task="ME_test_after_pretrain", branch="student_EEG_only"))

        model, rows = train_distill_student("B4 Stage 2: MI student fine-tuning", model, mi_train_loader, mi_val_loader, scaler, dim_names, finetune_cfg)
        stage_rows_all.extend(rows)

        metrics_mi = run_distill_student_epoch(model, mi_test_loader, criterion, scaler, dim_names, None, device)
        print(f"\n{name} MI TEST: r_mean={metrics_mi['r_mean']:.4f}, RMSE={metrics_mi['rmse_mean']:.4f}, MAE={metrics_mi['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            4, name, metrics_mi,
            "ME EEG+IMU teacher teaches EEG-only student; final MI test; no prototype alignment",
            evaluation_task="MI_test",
            branch="student_EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_mi, dim_names, evaluation_task="MI_test", branch="student_EEG_only"))

        metrics_me_final = run_distill_student_epoch(model, me_eeg_test_loader, criterion, scaler, dim_names, None, device)
        print(f"\n{name} ME TEST after MI fine-tuning: r_mean={metrics_me_final['r_mean']:.4f}, RMSE={metrics_me_final['rmse_mean']:.4f}, MAE={metrics_me_final['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            4, name, metrics_me_final,
            "Final MI-fine-tuned EEG-only student evaluated on ME test",
            evaluation_task="ME_test_after_MI_finetune",
            branch="student_EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_final, dim_names, evaluation_task="ME_test_after_MI_finetune", branch="student_EEG_only"))

    # ------------------------------------------------------------
    # Baseline 5: Full model
    # ------------------------------------------------------------
    if RUN_BASELINE_5:
        set_seed(SEED)
        name = "Baseline 5: Full model"
        model = FullPrivilegedModel(
            n_bands=me["n_bands"],
            n_channels=me["n_channels"],
            out_dim=me["out_dim"],
            n_subjects=1,
            target_len=TARGET_LEN,
            latent_dim=LATENT_DIM,
            lag_radius=LAG_RADIUS,
            dropout=0.20,
        ).to(device)

        model, rows = train_full_pretrain("B5 Stage 1: ME full privileged pretraining", model, me_train_loader, me_val_loader, scaler, dim_names, pretrain_cfg)
        stage_rows_all.extend(rows)

        me_pre = run_full_pretrain_epoch(model, me_test_loader, SequenceRegressionLoss(corr_weight=0.10), nn.SmoothL1Loss(beta=0.5), scaler, dim_names, None, device)
        metrics_me_teacher = me_pre["teacher"]
        metrics_me_student = me_pre["student"]
        print(f"\n{name} ME TEST teacher after full pretraining: r_mean={metrics_me_teacher['r_mean']:.4f}, RMSE={metrics_me_teacher['rmse_mean']:.4f}, MAE={metrics_me_teacher['mae_mean']:.4f}")
        print(f"{name} ME TEST student after full pretraining: r_mean={metrics_me_student['r_mean']:.4f}, RMSE={metrics_me_student['rmse_mean']:.4f}, MAE={metrics_me_student['mae_mean']:.4f}")

        summary_rows.append(summarize_baseline(
            5, name, metrics_me_teacher,
            "Lag-aware EEG+IMU teacher evaluated on ME test after full pretraining",
            evaluation_task="ME_test_after_pretrain",
            branch="teacher_lag_EEG_IMU",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_teacher, dim_names, evaluation_task="ME_test_after_pretrain", branch="teacher_lag_EEG_IMU"))

        summary_rows.append(summarize_baseline(
            5, name, metrics_me_student,
            "EEG-only student evaluated on ME test after full pretraining",
            evaluation_task="ME_test_after_pretrain",
            branch="student_EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_student, dim_names, evaluation_task="ME_test_after_pretrain", branch="student_EEG_only"))

        prototypes = compute_me_prototypes(model, me_train_loader, device, num_targets=4)
        print("Computed ME prototypes:", tuple(prototypes.shape))

        model, rows = train_full_student("B5 Stage 2: MI fine-tuning with prototype alignment", model, mi_train_loader, mi_val_loader, scaler, dim_names, prototypes, finetune_cfg)
        stage_rows_all.extend(rows)

        metrics_mi = run_full_student_epoch(model, mi_test_loader, criterion, scaler, dim_names, prototypes=None, proto_weight=0.0, optimizer=None, device=device)
        print(f"\n{name} MI TEST: r_mean={metrics_mi['r_mean']:.4f}, RMSE={metrics_mi['rmse_mean']:.4f}, MAE={metrics_mi['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            5, name, metrics_mi,
            "Lag-aware fusion + distillation + FiLM subject adaptation + ME prototype alignment; final MI test",
            evaluation_task="MI_test",
            branch="student_EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_mi, dim_names, evaluation_task="MI_test", branch="student_EEG_only"))

        metrics_me_final = run_full_student_epoch(model, me_eeg_test_loader, criterion, scaler, dim_names, prototypes=None, proto_weight=0.0, optimizer=None, device=device)
        print(f"\n{name} ME TEST after MI fine-tuning: r_mean={metrics_me_final['r_mean']:.4f}, RMSE={metrics_me_final['rmse_mean']:.4f}, MAE={metrics_me_final['mae_mean']:.4f}")
        summary_rows.append(summarize_baseline(
            5, name, metrics_me_final,
            "Final MI-fine-tuned EEG-only student evaluated on ME test",
            evaluation_task="ME_test_after_MI_finetune",
            branch="student_EEG_only",
        ))
        axis_rows.extend(long_axis_rows(name, metrics_me_final, dim_names, evaluation_task="ME_test_after_MI_finetune", branch="student_EEG_only"))


    # ------------------------------------------------------------
    # Baseline 6: ME EEG+IMU pretrain with bidirectional cross-modal MHA,
    #             no distillation/prototype alignment
    # ------------------------------------------------------------
    if RUN_BASELINE_6:
        set_seed(SEED)
        name = "Baseline 6: Bidirectional cross-modal MHA fusion"
        model = CrossModalAttentionFusionModel(
            me["n_bands"],
            me["n_channels"],
            me["out_dim"],
            TARGET_LEN,
            LATENT_DIM,
            CROSS_ATTENTION_HEADS,
        ).to(device)

        model, rows = train_fusion_model(
            "B6 Stage 1: ME EEG+IMU bidirectional cross-modal MHA pretraining",
            model,
            me_train_loader,
            me_val_loader,
            scaler,
            dim_names,
            pretrain_cfg,
            use_real_imu=True,
        )
        stage_rows_all.extend(rows)

        metrics_me_pre = run_fusion_epoch(
            model,
            me_test_loader,
            criterion,
            scaler,
            dim_names,
            None,
            device,
            use_real_imu=True,
        )
        print(
            f"\n{name} ME TEST after pretraining: "
            f"r_mean={metrics_me_pre['r_mean']:.4f}, "
            f"RMSE={metrics_me_pre['rmse_mean']:.4f}, "
            f"MAE={metrics_me_pre['mae_mean']:.4f}"
        )
        summary_rows.append(summarize_baseline(
            6, name, metrics_me_pre,
            "ME EEG+IMU pretrained model evaluated on ME test using real IMU; bidirectional cross-modal MHA fusion",
            evaluation_task="ME_test_after_pretrain",
            branch="EEG_IMU_BiCMHA_fusion",
        ))
        axis_rows.extend(long_axis_rows(
            name,
            metrics_me_pre,
            dim_names,
            evaluation_task="ME_test_after_pretrain",
            branch="EEG_IMU_BiCMHA_fusion",
        ))

        # Fine-tune on MI with zero IMU, exactly like Baseline 3.
        model, rows = train_fusion_model(
            "B6 Stage 2: MI EEG-only fine-tuning with zero IMU",
            model,
            mi_train_loader,
            mi_val_loader,
            scaler,
            dim_names,
            finetune_cfg,
            use_real_imu=False,
        )
        stage_rows_all.extend(rows)

        metrics_mi = run_fusion_epoch(
            model,
            mi_test_loader,
            criterion,
            scaler,
            dim_names,
            None,
            device,
            use_real_imu=False,
        )
        print(
            f"\n{name} MI TEST: "
            f"r_mean={metrics_mi['r_mean']:.4f}, "
            f"RMSE={metrics_mi['rmse_mean']:.4f}, "
            f"MAE={metrics_mi['mae_mean']:.4f}"
        )
        summary_rows.append(summarize_baseline(
            6, name, metrics_mi,
            "ME EEG+IMU pretraining with bidirectional cross-modal MHA; MI uses zero IMU; no distillation/prototype alignment",
            evaluation_task="MI_test",
            branch="EEG_zeroIMU_BiCMHA_fusion",
        ))
        axis_rows.extend(long_axis_rows(
            name,
            metrics_mi,
            dim_names,
            evaluation_task="MI_test",
            branch="EEG_zeroIMU_BiCMHA_fusion",
        ))

        metrics_me_final_zero = run_fusion_epoch(
            model,
            me_eeg_test_loader,
            criterion,
            scaler,
            dim_names,
            None,
            device,
            use_real_imu=False,
        )
        print(
            f"\n{name} ME TEST after MI fine-tuning with zero IMU: "
            f"r_mean={metrics_me_final_zero['r_mean']:.4f}, "
            f"RMSE={metrics_me_final_zero['rmse_mean']:.4f}, "
            f"MAE={metrics_me_final_zero['mae_mean']:.4f}"
        )
        summary_rows.append(summarize_baseline(
            6, name, metrics_me_final_zero,
            "Final MI-fine-tuned model evaluated on ME test with zero IMU input; bidirectional cross-modal MHA fusion",
            evaluation_task="ME_test_after_MI_finetune_zeroIMU",
            branch="EEG_zeroIMU_BiCMHA_fusion",
        ))
        axis_rows.extend(long_axis_rows(
            name,
            metrics_me_final_zero,
            dim_names,
            evaluation_task="ME_test_after_MI_finetune_zeroIMU",
            branch="EEG_zeroIMU_BiCMHA_fusion",
        ))



    # ------------------------------------------------------------
    # Baseline 7: ME EEG+IMU pretrain with normal MHA fusion,
    #             no distillation/prototype alignment
    # ------------------------------------------------------------
    if RUN_BASELINE_7:
        set_seed(SEED)
        name = "Baseline 7: Normal MHA fusion"
        model = NormalMHAFusionModel(
            me["n_bands"],
            me["n_channels"],
            me["out_dim"],
            TARGET_LEN,
            LATENT_DIM,
            NORMAL_MHA_HEADS,
        ).to(device)

        model, rows = train_fusion_model(
            "B7 Stage 1: ME EEG+IMU normal MHA pretraining",
            model,
            me_train_loader,
            me_val_loader,
            scaler,
            dim_names,
            pretrain_cfg,
            use_real_imu=True,
        )
        stage_rows_all.extend(rows)

        metrics_me_pre = run_fusion_epoch(
            model,
            me_test_loader,
            criterion,
            scaler,
            dim_names,
            None,
            device,
            use_real_imu=True,
        )
        print(
            f"\n{name} ME TEST after pretraining: "
            f"r_mean={metrics_me_pre['r_mean']:.4f}, "
            f"RMSE={metrics_me_pre['rmse_mean']:.4f}, "
            f"MAE={metrics_me_pre['mae_mean']:.4f}"
        )
        summary_rows.append(summarize_baseline(
            7, name, metrics_me_pre,
            "ME EEG+IMU pretrained model evaluated on ME test using real IMU; normal MHA fusion over fused EEG+IMU tokens",
            evaluation_task="ME_test_after_pretrain",
            branch="EEG_IMU_NormalMHA_fusion",
        ))
        axis_rows.extend(long_axis_rows(
            name,
            metrics_me_pre,
            dim_names,
            evaluation_task="ME_test_after_pretrain",
            branch="EEG_IMU_NormalMHA_fusion",
        ))

        # Fine-tune on MI with zero IMU, exactly like Baseline 3 and Baseline 6.
        model, rows = train_fusion_model(
            "B7 Stage 2: MI EEG-only fine-tuning with zero IMU",
            model,
            mi_train_loader,
            mi_val_loader,
            scaler,
            dim_names,
            finetune_cfg,
            use_real_imu=False,
        )
        stage_rows_all.extend(rows)

        metrics_mi = run_fusion_epoch(
            model,
            mi_test_loader,
            criterion,
            scaler,
            dim_names,
            None,
            device,
            use_real_imu=False,
        )
        print(
            f"\n{name} MI TEST: "
            f"r_mean={metrics_mi['r_mean']:.4f}, "
            f"RMSE={metrics_mi['rmse_mean']:.4f}, "
            f"MAE={metrics_mi['mae_mean']:.4f}"
        )
        summary_rows.append(summarize_baseline(
            7, name, metrics_mi,
            "ME EEG+IMU pretraining with normal MHA fusion; MI uses zero IMU; no distillation/prototype alignment",
            evaluation_task="MI_test",
            branch="EEG_zeroIMU_NormalMHA_fusion",
        ))
        axis_rows.extend(long_axis_rows(
            name,
            metrics_mi,
            dim_names,
            evaluation_task="MI_test",
            branch="EEG_zeroIMU_NormalMHA_fusion",
        ))

        metrics_me_final_zero = run_fusion_epoch(
            model,
            me_eeg_test_loader,
            criterion,
            scaler,
            dim_names,
            None,
            device,
            use_real_imu=False,
        )
        print(
            f"\n{name} ME TEST after MI fine-tuning with zero IMU: "
            f"r_mean={metrics_me_final_zero['r_mean']:.4f}, "
            f"RMSE={metrics_me_final_zero['rmse_mean']:.4f}, "
            f"MAE={metrics_me_final_zero['mae_mean']:.4f}"
        )
        summary_rows.append(summarize_baseline(
            7, name, metrics_me_final_zero,
            "Final MI-fine-tuned model evaluated on ME test with zero IMU input; normal MHA fusion",
            evaluation_task="ME_test_after_MI_finetune_zeroIMU",
            branch="EEG_zeroIMU_NormalMHA_fusion",
        ))
        axis_rows.extend(long_axis_rows(
            name,
            metrics_me_final_zero,
            dim_names,
            evaluation_task="ME_test_after_MI_finetune_zeroIMU",
            branch="EEG_zeroIMU_NormalMHA_fusion",
        ))

    save_results_to_excel(summary_rows, axis_rows, stage_rows_all, dim_names)

    elapsed = time.time() - start_time
    print(f"\nAll requested baselines finished in {elapsed/60:.2f} minutes.")


if __name__ == "__main__":
    main()
