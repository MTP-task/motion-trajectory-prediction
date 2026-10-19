"""Load one subject's step 3 outputs and turn them into model inputs, splits and loaders.

Step 3 writes three files per subject and condition (ME or MI): EEG epochs, velocity
trajectories and target labels. In ``data/processed`` they sit in the subfolders listed
in `DATA_FILES`; on the lab PC (``D:/MTP_task/model``) all of them sit in one folder.
Both layouts work.

`prepare_dataset` returns a `PreparedCondition` with a `Split` each for training,
validation and testing; `make_loaders` turns it into PyTorch `Loaders`.
"""

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from mtp.config import LABEL_TO_TARGET_ID, N_TARGETS, TARGET_STREAMS, TRAJECTORY_STREAMS
from mtp.features import make_eeg_input

log = logging.getLogger(__name__)

DATA_FILES = {
    "eeg": ("EEG_data", "EEG_data_{subject}_{condition}.npy"),
    "trajectory": ("Kinematics_data/trajectory", "trajectory_{subject}_{condition}.npy"),
    "labels": ("Kinematics_data/label", "labels_clean_{subject}_{condition}.npy"),
}


def find_data_file(data_dir, kind: str, subject: str, condition: str) -> Path:
    """Find one of the step 3 output files.

    Parameters
    ----------
    data_dir : str or Path
        Folder with the step 3 outputs, in either layout.
    kind : {"eeg", "trajectory", "labels"}
        Which file to find.
    subject : str
        Subject ID, such as "s026".
    condition : {"ME", "MI"}
        Motor execution or motor imagery.

    Returns
    -------
    Path

    Raises
    ------
    FileNotFoundError
        If the file is in neither place. The message lists both paths that were tried.
    """
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
    """Return the IDs of all subjects that have an ME EEG file in `data_dir`.

    Parameters
    ----------
    data_dir : str or Path
        Folder with the step 3 outputs, in either layout.

    Returns
    -------
    list of str
        Sorted subject IDs, such as ["s008", "s009"].

    Raises
    ------
    FileNotFoundError
        If no subject is found.
    """
    subdir, _ = DATA_FILES["eeg"]
    prefix, suffix = "EEG_data_", "_ME.npy"
    subjects = set()
    for folder in (Path(data_dir) / subdir, Path(data_dir)):
        for path in folder.glob(f"{prefix}*{suffix}"):
            subjects.add(path.name[len(prefix) : -len(suffix)])
    if not subjects:
        raise FileNotFoundError(f"No {prefix}<subject>{suffix} files found in {data_dir}")
    return sorted(subjects)


TRAJECTORY_KEYS = [key for key, _names in TRAJECTORY_STREAMS.values()]


def load_eeg_and_trajectory(eeg_path, traj_path, name: str):
    """Load the EEG epochs and velocity trajectories of one condition.

    Parameters
    ----------
    eeg_path, traj_path : str or Path
        The EEG_data and trajectory files written by step 3.
    name : str
        Condition name for log messages and errors, such as "ME".

    Returns
    -------
    eeg : ndarray, shape (n_trials, n_channels, n_samples)
        EEG in volts, covering -2 to 7 s around movement onset.
    traj : dict of ndarray
        "shoulder", "elbow" and "wirst" velocities in m/s, each of shape
        (n_trials, 150, 3), and "IMU" rotation velocities in rad/s, of shape
        (n_trials, 150, 6). The trials are in the same order as in `eeg`.

    Raises
    ------
    ValueError
        If the trajectory file does not have this format.
    """
    eeg = np.load(eeg_path)
    # The trajectory file holds a Python dict, which np.save stores with pickle.
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
    """Load the target labels of one condition.

    Parameters
    ----------
    labels_path : str or Path
        The labels_clean file written by step 3.
    name : str
        Condition name for log messages, such as "ME".

    Returns
    -------
    ndarray of int32, shape (n_trials,)
        One label per trial, which should be 101 to 104 for balls 1 to 4.
        `build_target_sequence` rejects any other label.
    """
    labels = np.load(labels_path).astype(np.int32).ravel()
    unique, counts = np.unique(labels, return_counts=True)
    log.info(
        "Loaded %s labels: %d trials (%s)",
        name,
        len(labels),
        ", ".join(f"{label}: {count}" for label, count in zip(unique, counts)),
    )
    return labels


def build_target_sequence(traj: dict, target_stream: str, labels_clean: np.ndarray):
    """Build the velocity targets and the target index of each trial.

    Parameters
    ----------
    traj : dict of ndarray
        Trajectories from `load_eeg_and_trajectory`.
    target_stream : str
        Which velocities to predict; see `Config.target_stream`.
    labels_clean : ndarray, shape (n_trials,)
        Labels 101 to 104.

    Returns
    -------
    y : ndarray of float32, shape (n_trials, 150, n_outputs)
        Velocities to predict: 3 outputs for one joint, 6 for "imu", 9 for "all_xyz"
        and 15 for "all_kin".
    target_ids : ndarray of int64, shape (n_trials,)
        Target index (0 to 3) of each trial.
    dim_names : list of str
        Name of each output, such as "wrist_Vx".

    Raises
    ------
    ValueError
        If `target_stream` is unknown, a label is not 101 to 104, or the number of labels
        differs from the number of trials.
    """
    if target_stream not in TARGET_STREAMS:
        raise ValueError(
            f"Unknown target stream {target_stream!r}; expected one of {list(TARGET_STREAMS)}"
        )
    parts = [TRAJECTORY_STREAMS[part] for part in TARGET_STREAMS[target_stream]]
    y = np.concatenate([traj[key] for key, _names in parts], axis=-1).astype(np.float32)
    dim_names = [name for _key, names in parts for name in names]

    labels = np.asarray(labels_clean).ravel().tolist()
    unknown = sorted(set(labels) - LABEL_TO_TARGET_ID.keys())
    if unknown:
        raise ValueError(f"Unknown labels {unknown}; expected only {sorted(LABEL_TO_TARGET_ID)}.")
    if len(labels) != y.shape[0]:
        raise ValueError(f"{len(labels)} labels for {y.shape[0]} trajectory trials.")
    target_ids = np.array([LABEL_TO_TARGET_ID[label] for label in labels], dtype=np.int64)
    return y, target_ids, dim_names


def stratified_split_indices(
    target_ids: np.ndarray,
    seed: int = 42,
    test_fraction: float = 0.15,
    val_fraction: float = 0.1765,
):
    """Split trial indices into training, validation and test sets.

    Each set has the same mix of targets as the whole.

    Parameters
    ----------
    target_ids : ndarray, shape (n_trials,)
        Target index of each trial.
    seed : int
        Random seed for the split.
    test_fraction : float
        Share of all trials that go to the test set.
    val_fraction : float
        Share of the remaining trials that go to the validation set.

    Returns
    -------
    train_idx, val_idx, test_idx : ndarray of int
        Trial indices of each set.
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
    """Compute the mean and standard deviation of each output over the training targets.

    Parameters
    ----------
    *y_train_arrays : ndarray, shape (n_trials, n_times, n_outputs)
        Training targets. ME and MI are passed together, so both share one scale.
    eps : float
        Added to the standard deviation to avoid dividing by zero.

    Returns
    -------
    dict
        "mean" and "std", each of shape (1, 1, n_outputs).
    """
    y_cat = np.concatenate(y_train_arrays, axis=0)
    mean = y_cat.mean(axis=(0, 1), keepdims=True)
    std = y_cat.std(axis=(0, 1), keepdims=True)
    return {"mean": mean.astype(np.float32), "std": (std + eps).astype(np.float32)}


def normalize_targets(y: np.ndarray, scaler: dict):
    """Scale targets with a scaler from `fit_target_scaler`."""
    return ((y - scaler["mean"]) / scaler["std"]).astype(np.float32)


def denormalize_targets(y_norm: np.ndarray, scaler: dict):
    """Undo `normalize_targets`, giving velocities in their original units."""
    return y_norm * scaler["std"] + scaler["mean"]


SPLITS = ("train", "val", "test")


@dataclass
class Split:
    """The trials of one split (training, validation or test) of a prepared condition.

    Attributes
    ----------
    X : ndarray of float32, shape (n_trials, n_bands, n_channels, n_samples)
        EEG model input from `make_eeg_input`.
    y : ndarray of float32, shape (n_trials, 150, n_outputs)
        Velocity targets in their original units.
    target_id : ndarray of int64, shape (n_trials,)
        Target index (0 to 3) of each trial.
    subject_id : ndarray of int64, shape (n_trials,)
        Subject index of each trial. Always 0, since each run has one subject.
    imu : ndarray of float32, shape (n_trials, 150, 6), or None
        Normalized IMU signals. Only ME has them.
    y_norm : ndarray of float32, or None
        `y` scaled with the run's target scaler. `Experiment` sets it once the scaler is
        fitted on the training trials of both conditions.
    """

    X: np.ndarray
    y: np.ndarray
    target_id: np.ndarray
    subject_id: np.ndarray
    imu: np.ndarray | None = None
    y_norm: np.ndarray | None = None


@dataclass
class PreparedCondition:
    """One condition (ME or MI) of one subject, split and ready for the models.

    Attributes
    ----------
    train, val, test : Split
        The three splits.
    dim_names : list of str
        Name of each output, such as "wrist_Vx".
    """

    train: Split
    val: Split
    test: Split
    dim_names: list[str]

    def splits(self) -> list[Split]:
        """Return the training, validation and test splits, in that order."""
        return [self.train, self.val, self.test]

    @property
    def n_bands(self) -> int:
        """Number of filter-bank bands in the EEG input."""
        return self.train.X.shape[1]

    @property
    def n_channels(self) -> int:
        """Number of EEG channels."""
        return self.train.X.shape[2]

    @property
    def out_dim(self) -> int:
        """Number of velocity outputs."""
        return self.train.y.shape[-1]


def prepare_dataset(cfg, condition: str, *, with_imu: bool) -> PreparedCondition:
    """Load, split and preprocess one condition for ``cfg.subject``.

    Parameters
    ----------
    cfg : Config
        Run settings. The subject, data folder, target stream, split settings and EEG
        preprocessing settings are used.
    condition : {"ME", "MI"}
        Motor execution or motor imagery.
    with_imu : bool
        Also prepare the IMU signals as model input. Only ME has them.

    Returns
    -------
    PreparedCondition
    """
    eeg, y, target_ids, dim_names, imu = _load_condition(cfg, condition, with_imu=with_imu)
    split_idx = _split_trials(cfg, condition, target_ids)

    X = make_eeg_input(eeg, use_filterbank=cfg.use_filterbank, clip_value=cfg.eeg_clip_value)
    log.info("%s EEG model input: %s", condition, X.shape)

    imu_by_split = _normalize_imu(imu, split_idx) if with_imu else dict.fromkeys(SPLITS)
    subject_ids = np.zeros(len(target_ids), dtype=np.int64)
    splits = {
        name: Split(
            X=X[idx],
            y=y[idx],
            target_id=target_ids[idx],
            subject_id=subject_ids[idx],
            imu=imu_by_split[name],
        )
        for name, idx in split_idx.items()
    }
    return PreparedCondition(**splits, dim_names=dim_names)


def _load_condition(cfg, condition, *, with_imu):
    """Load the EEG, targets and, if `with_imu`, the IMU of one condition and check them."""
    paths = {
        kind: find_data_file(cfg.data_dir, kind, cfg.subject, condition) for kind in DATA_FILES
    }
    eeg, traj = load_eeg_and_trajectory(paths["eeg"], paths["trajectory"], condition)
    labels_clean = load_labels_clean(paths["labels"], condition)

    if eeg.shape[0] != len(labels_clean):
        raise ValueError(f"{condition}: {eeg.shape[0]} EEG epochs but {len(labels_clean)} labels.")

    y, target_ids, dim_names = build_target_sequence(traj, cfg.target_stream, labels_clean)

    imu = None
    if with_imu:
        imu = traj["IMU"].astype(np.float32)
        if imu.shape[0] != y.shape[0]:
            raise ValueError(f"{condition}: {imu.shape[0]} IMU epochs but {y.shape[0]} labels.")
    log.info(
        "%s targets: y %s (%s)%s",
        condition,
        y.shape,
        ", ".join(dim_names),
        f", IMU {imu.shape}" if with_imu else "",
    )
    return eeg, y, target_ids, dim_names, imu


def _split_trials(cfg, condition, target_ids):
    """Split the trials with `stratified_split_indices` and log how many each split got."""
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
            f"{split} {len(idx)} {np.bincount(target_ids[idx], minlength=N_TARGETS).tolist()}"
            for split, idx in split_idx.items()
        ),
    )
    return split_idx


def _normalize_imu(imu, split_idx, eps=1e-8):
    """Normalize the IMU of each split with the mean and std of the training trials."""
    train = imu[split_idx["train"]]
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True) + eps
    return {name: ((imu[idx] - mean) / std).astype(np.float32) for name, idx in split_idx.items()}


@dataclass
class Loaders:
    """Training, validation and test loaders over one condition."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def make_loader(split: Split, batch_size: int, *, shuffle: bool, with_imu: bool = False):
    """Make a DataLoader over one split.

    Parameters
    ----------
    split : Split
        A split whose `y_norm` has been set.
    batch_size : int
        Trials per batch.
    shuffle : bool
        Shuffle the trials in every epoch, as for training.
    with_imu : bool
        Include the IMU signals in each batch.

    Returns
    -------
    DataLoader
        Batches of (eeg, y, target_id, subject_id), or of (eeg, imu, y, target_id,
        subject_id) when `with_imu` is True, with ``y`` taken from ``split.y_norm``.
    """
    tensors = [torch.from_numpy(split.X).float()]
    if with_imu:
        tensors.append(torch.from_numpy(split.imu).float())
    tensors += [
        torch.from_numpy(split.y_norm).float(),
        torch.from_numpy(split.target_id).long(),
        torch.from_numpy(split.subject_id).long(),
    ]
    return DataLoader(
        TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, drop_last=False
    )


def make_loaders(condition: PreparedCondition, batch_size: int, *, with_imu: bool = False):
    """Make the training, validation and test loaders of a condition.

    Only the training loader is shuffled. See `make_loader` for what the batches hold.

    Returns
    -------
    Loaders
    """
    return Loaders(
        train=make_loader(condition.train, batch_size, shuffle=True, with_imu=with_imu),
        val=make_loader(condition.val, batch_size, shuffle=False, with_imu=with_imu),
        test=make_loader(condition.test, batch_size, shuffle=False, with_imu=with_imu),
    )
