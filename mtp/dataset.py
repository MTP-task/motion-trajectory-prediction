import os

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from mtp.config import (
    EEG_FS, KEY_TO_TARGET_ID, LABEL_TO_KEY, ME_EEG_PATH, ME_LABELS_PATH, ME_TRAJ_PATH,
    MI_EEG_PATH, MI_LABELS_PATH, MI_TRAJ_PATH, SEED, TARGET_ORDER, TARGET_STREAM,
    TRAJ_KEY_ALIAS, USE_FILTERBANK,
)
from mtp.features import make_eeg_input, normalize_imu_train_only


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
