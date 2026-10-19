"""Load step 3 outputs for one subject and turn them into model-ready splits and loaders."""

import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from mtp.config import EEG_FS, KEY_TO_TARGET_ID, LABEL_TO_KEY, TRAJ_KEY_ALIAS
from mtp.features import make_eeg_input, normalize_imu_train_only

log = logging.getLogger(__name__)

# ---- Locating files ---------------------------------------------------------

# Step 3 writes three files per subject and condition (ME/MI). In data/processed they
# sit in these subfolders; on the lab PC (D:\MTP_task\model) they all sit in one folder.
DATA_FILES = {
    "eeg": ("EEG_data", "EEG_data_{subject}_{condition}.npy"),
    "trajectory": ("Kinematics_data/trajectory", "trajectory_{subject}_{condition}.npy"),
    "labels": ("Kinematics_data/label", "labels_clean_{subject}_{condition}.npy"),
}


def find_data_file(data_dir, kind: str, subject: str, condition: str) -> Path:
    subdir, pattern = DATA_FILES[kind]
    name = pattern.format(subject=subject, condition=condition)
    candidates = [Path(data_dir) / subdir / name, Path(data_dir) / name]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"{condition} {kind} file not found for {subject}. Looked for:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def list_subjects(data_dir) -> list[str]:
    """Subjects with ME EEG data in data_dir, e.g. ['s008', 's009', ...]."""
    subdir, _ = DATA_FILES["eeg"]
    prefix, suffix = "EEG_data_", "_ME.npy"
    subjects = set()
    for folder in (Path(data_dir) / subdir, Path(data_dir)):
        for path in folder.glob(f"{prefix}*{suffix}"):
            subjects.add(path.name[len(prefix) : -len(suffix)])
    if not subjects:
        raise FileNotFoundError(f"No {prefix}<subject>{suffix} files found in {data_dir}")
    return sorted(subjects)


# ---- Loading and target construction ----------------------------------------

TRAJECTORY_KEYS = ["shoulder", "elbow", "wirst", "IMU"]


def load_eeg_and_trajectory(eeg_path, traj_path, name: str):
    """
    Load EEG (N, C, T_full) and the trajectory dict written by step 3:
        traj["shoulder"] -> ndarray (N, 150, 3)
        traj["elbow"]    -> ndarray (N, 150, 3)
        traj["wirst"]    -> ndarray (N, 150, 3)
        traj["IMU"]      -> ndarray (N, 150, 6)
    Trajectories are already aligned with the EEG epoch order.
    """
    eeg = np.load(eeg_path, allow_pickle=True)
    traj = np.load(traj_path, allow_pickle=True).item()

    log.info("Loaded %s EEG: %s %s", name, eeg.shape, eeg.dtype)

    if not all(isinstance(traj.get(k), np.ndarray) for k in TRAJECTORY_KEYS):
        raise ValueError(
            f"{name} trajectory file {traj_path} must be a dict of epoch-aligned arrays "
            f"with keys {TRAJECTORY_KEYS} (the format written by step 3)."
        )
    log.info(
        "Loaded %s trajectory: %s", name, ", ".join(f"{k} {arr.shape}" for k, arr in traj.items())
    )

    return eeg, traj


def load_labels_clean(labels_path, name: str) -> np.ndarray:
    labels = np.load(labels_path, allow_pickle=True)
    labels = np.asarray(labels).astype(np.int32).ravel()

    valid = set(LABEL_TO_KEY.keys())
    found = set(map(int, np.unique(labels)))
    unknown = found - valid
    if len(unknown) > 0:
        raise ValueError(
            f"{name}: unknown labels found {sorted(unknown)}. Expected only {sorted(valid)}."
        )

    unique, counts = np.unique(labels, return_counts=True)
    log.info(
        "Loaded %s labels: %d trials (%s)",
        name,
        labels.shape[0],
        ", ".join(f"{LABEL_TO_KEY[int(lab)]}: {count}" for lab, count in zip(unique, counts)),
    )

    return labels


def _make_dim_names(target_stream: str):
    if target_stream in ["wrist", "wirst", "elbow", "shoulder"]:
        clean_name = "wrist" if target_stream == "wirst" else target_stream
        return [f"{clean_name}_Vx", f"{clean_name}_Vy", f"{clean_name}_Vz"]

    if target_stream == "imu":
        return [
            "IMU_elbow_rotVx",
            "IMU_elbow_rotVy",
            "IMU_elbow_rotVz",
            "IMU_wrist_rotVx",
            "IMU_wrist_rotVy",
            "IMU_wrist_rotVz",
        ]

    if target_stream == "all_xyz":
        return [
            "shoulder_Vx",
            "shoulder_Vy",
            "shoulder_Vz",
            "elbow_Vx",
            "elbow_Vy",
            "elbow_Vz",
            "wrist_Vx",
            "wrist_Vy",
            "wrist_Vz",
        ]

    if target_stream == "all_kin":
        return [
            "shoulder_Vx",
            "shoulder_Vy",
            "shoulder_Vz",
            "elbow_Vx",
            "elbow_Vy",
            "elbow_Vz",
            "wrist_Vx",
            "wrist_Vy",
            "wrist_Vz",
            "IMU_elbow_rotVx",
            "IMU_elbow_rotVy",
            "IMU_elbow_rotVz",
            "IMU_wrist_rotVx",
            "IMU_wrist_rotVy",
            "IMU_wrist_rotVz",
        ]

    raise ValueError(f"Unknown TARGET_STREAM: {target_stream}")


def _target_ids_from_labels(labels_clean: np.ndarray, n_epochs: int, name: str = "") -> np.ndarray:
    """
    Convert labels_clean to target_ids 0..3.
    labels_clean is used only for stratified splitting and class/prototype labels.
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


def _get_target_array(traj: dict, target_stream: str) -> np.ndarray:
    """
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
    """Return the target trajectories y, target_ids 0..3 (from labels_clean) and dim names."""
    labels_clean = np.asarray(labels_clean).astype(np.int32).ravel()
    dim_names = _make_dim_names(target_stream)

    y = _get_target_array(traj, target_stream).astype(np.float32)
    target_ids = _target_ids_from_labels(
        labels_clean=labels_clean,
        n_epochs=y.shape[0],
        name="trajectory target",
    )
    return y, target_ids, dim_names


def build_imu_sequence(traj: dict, labels_clean: np.ndarray):
    """Return the ME IMU input sequence (N, 150, 6)."""
    if "IMU" not in traj:
        raise KeyError("Trajectory file does not contain key 'IMU'.")

    imu = traj["IMU"].astype(np.float32)
    # Validates that labels_clean matches the number of IMU epochs.
    _target_ids_from_labels(
        labels_clean=labels_clean, n_epochs=imu.shape[0], name="IMU trajectory"
    )
    return imu


# ---- Split and target scaling -----------------------------------------------


def stratified_split_indices(
    target_ids: np.ndarray,
    seed: int = 42,
    test_fraction: float = 0.15,
    val_fraction: float = 0.1765,
):
    """
    Split trial indices into train / val / test, stratified by target.
    test_fraction is a share of all trials; val_fraction a share of the rest.
    """
    indices = np.arange(len(target_ids))
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=target_ids,
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_fraction,
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


# ---- Preparing one condition ------------------------------------------------

SPLITS = ("train", "val", "test")


def prepare_dataset(cfg, condition: str, with_imu: bool):
    """
    Load, split and preprocess one condition ("ME" or "MI") for cfg.subject.

    Returns a dict with X_<split> (N, bands, channels, samples), y_<split>, id_<split>
    (target 0..3), sub_<split> (subject index), imu_<split> if with_imu, plus dim_names,
    n_bands, n_channels and out_dim.
    """
    paths = {
        kind: find_data_file(cfg.data_dir, kind, cfg.subject, condition) for kind in DATA_FILES
    }
    eeg, traj = load_eeg_and_trajectory(paths["eeg"], paths["trajectory"], condition)
    labels_clean = load_labels_clean(paths["labels"], condition)

    if eeg.shape[0] != labels_clean.shape[0]:
        raise ValueError(
            f"{condition}: EEG epochs ({eeg.shape[0]}) != labels_clean length ({labels_clean.shape[0]})."
        )

    y, target_ids, dim_names = build_target_sequence(traj, cfg.target_stream, labels_clean)
    if eeg.shape[0] != y.shape[0]:
        raise ValueError(
            f"{condition}: EEG epochs ({eeg.shape[0]}) != target epochs ({y.shape[0]})."
        )

    if with_imu:
        imu = build_imu_sequence(traj, labels_clean)
        if imu.shape[0] != y.shape[0]:
            raise ValueError(
                f"{condition}: IMU epochs ({imu.shape[0]}) != target epochs ({y.shape[0]})."
            )
    log.info(
        "%s targets: y %s (%s)%s",
        condition,
        y.shape,
        ", ".join(dim_names),
        f", IMU {imu.shape}" if with_imu else "",
    )

    split_idx = dict(
        zip(
            SPLITS,
            stratified_split_indices(
                target_ids,
                cfg.seed,
                test_fraction=cfg.test_fraction,
                val_fraction=cfg.val_fraction,
            ),
        )
    )
    log.info(
        "%s split (trials per target 1-4): %s",
        condition,
        " | ".join(
            f"{split} {len(idx)} {np.bincount(target_ids[idx], minlength=4).tolist()}"
            for split, idx in split_idx.items()
        ),
    )

    X = make_eeg_input(
        eeg, EEG_FS, use_filterbank=cfg.use_filterbank, clip_value=cfg.eeg_clip_value
    )
    log.info("%s EEG model input: %s", condition, X.shape)

    subject_ids = np.zeros(len(target_ids), dtype=np.int64)
    data = {}
    for split, idx in split_idx.items():
        data[f"X_{split}"] = X[idx]
        data[f"y_{split}"] = y[idx]
        data[f"id_{split}"] = target_ids[idx]
        data[f"sub_{split}"] = subject_ids[idx]

    if with_imu:
        imu_train_n, imu_val_n, imu_test_n, imu_scaler = normalize_imu_train_only(
            *(imu[split_idx[split]] for split in SPLITS)
        )
        data.update(
            imu_train=imu_train_n, imu_val=imu_val_n, imu_test=imu_test_n, imu_scaler=imu_scaler
        )

    data.update(
        dim_names=dim_names, n_bands=X.shape[1], n_channels=X.shape[2], out_dim=y.shape[-1]
    )
    return data


# ---- PyTorch loaders --------------------------------------------------------


def make_loader(data: dict, split: str, batch_size: int, with_imu: bool = False):
    """
    DataLoader over one split of a prepared dataset; shuffled for "train" only.

    Batches are (eeg, y, target_id, subject_id), or (eeg, imu, y, target_id, subject_id)
    with with_imu=True. Targets come from the normalized y_<split>_n arrays.
    """
    tensors = [torch.from_numpy(data[f"X_{split}"]).float()]
    if with_imu:
        tensors.append(torch.from_numpy(data[f"imu_{split}"]).float())
    tensors += [
        torch.from_numpy(data[f"y_{split}_n"]).float(),
        torch.from_numpy(data[f"id_{split}"]).long(),
        torch.from_numpy(data[f"sub_{split}"]).long(),
    ]
    return DataLoader(
        TensorDataset(*tensors), batch_size=batch_size, shuffle=split == "train", drop_last=False
    )
