"""Project settings.

The module-level constants describe the recordings and stay the same for every run. EEG
is sampled at ``EEG_FS`` = 512 Hz and the velocity targets at ``TRAJ_FS`` = 30 Hz. Each
epoch covers -2 to 7 s around movement onset, and the models use the part from 0 to 5 s:
``MODEL_EEG_SAMPLES`` = 2560 EEG samples and ``TARGET_LEN`` = 150 velocity samples. The
IMU gives ``IMU_DIM`` = 6 signals: rotation velocity around x, y and z at the elbow and
at the wrist.

`TRAJECTORY_STREAMS` lists the signals in the step 3 trajectory files, with their key in
the file and the names of their outputs. `TARGET_STREAMS` lists the signals that make up
each choice of `Config.target_stream`, in output order.

Each trial has a label from 101 to 104 for its target, ball 1 to 4.
`LABEL_TO_TARGET_ID` turns the labels into the target indices 0 to 3 used in the code.

`Config` holds the settings of one run. Every setting can also be given on the command
line; see ``python -m mtp.modeling.train --help``.
"""

import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path
import re
import typing

PROJ_ROOT = Path(__file__).resolve().parents[1]

EEG_FS = 512.0
TRAJ_FS = 30.0
IMU_DIM = 6

EEG_EPOCH_TMIN = -2.0
EEG_EPOCH_TMAX = 7.0
EEG_CROP_TMIN = 0.0
EEG_CROP_TMAX = 5.0
MODEL_EEG_SEC = EEG_CROP_TMAX - EEG_CROP_TMIN
MODEL_EEG_SAMPLES = round(EEG_FS * MODEL_EEG_SEC)
TARGET_LEN = round(TRAJ_FS * MODEL_EEG_SEC)

BANDS = {
    "slow": (0.5, 1.0),
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
    # If final low-pass is exactly 40 Hz, change this to (30.0, 40.0).
    "low_gamma": (30.0, 45.0),
}

TRAJECTORY_STREAMS = {
    "shoulder": ("shoulder", ["shoulder_Vx", "shoulder_Vy", "shoulder_Vz"]),
    "elbow": ("elbow", ["elbow_Vx", "elbow_Vy", "elbow_Vz"]),
    # The step 3 files spell the wrist key "wirst".
    "wrist": ("wirst", ["wrist_Vx", "wrist_Vy", "wrist_Vz"]),
    "imu": (
        "IMU",
        [
            "IMU_elbow_rotVx",
            "IMU_elbow_rotVy",
            "IMU_elbow_rotVz",
            "IMU_wrist_rotVx",
            "IMU_wrist_rotVy",
            "IMU_wrist_rotVz",
        ],
    ),
}

TARGET_STREAMS = {
    "shoulder": ["shoulder"],
    "elbow": ["elbow"],
    "wrist": ["wrist"],
    "imu": ["imu"],
    "all_xyz": ["shoulder", "elbow", "wrist"],
    "all_kin": ["shoulder", "elbow", "wrist", "imu"],
}

LABEL_TO_TARGET_ID = {101: 0, 102: 1, 103: 2, 104: 3}
N_TARGETS = len(LABEL_TO_TARGET_ID)

BASELINE_IDS = tuple(range(8))


@dataclass
class Config:
    """Settings for one training run on one subject.

    The defaults reproduce the original experiment.

    Attributes
    ----------
    subject : str
        Subject ID, such as "s026".
    data_dir : Path
        Folder with the step 3 output files.
    results_dir : Path
        Folder for the Excel results and log files.
    baselines : list of int
        Baselines to run, from 0 to 7. `mtp.modeling.baselines` describes each one.
    target_stream : str
        Which velocities the models predict. "shoulder", "elbow" or "wrist" gives the x,
        y and z velocity of one joint (3 outputs), and "imu" the 6 IMU rotation
        velocities. "all_xyz" covers all three joints (9 outputs) and "all_kin" adds the
        IMU (15 outputs).
    use_filterbank : bool
        Split the EEG into the frequency bands in `BANDS`. If False, the models get one
        broadband signal.
    eeg_clip_value : float
        EEG is z-scored per trial and channel, then clipped to this many standard
        deviations either side of zero, which limits the effect of leftover artifacts.
    test_fraction : float
        Share of all trials held out for testing.
    val_fraction : float
        Share of the remaining trials used for validation. 0.1765 of the remaining 85%
        is about 15% of all trials.
    device : str
        "auto" uses the GPU when there is one. Otherwise "cpu", "cuda" or
        "cuda:<index>".
    seed : int
        Random seed. It is reset before each baseline, so a baseline's results do not
        depend on which other baselines run.
    deterministic : bool
        Use only deterministic algorithms, so that a rerun with the same seed gives
        identical results. Without this, results on the GPU vary slightly between runs.
    batch_size : int
        Trials per training batch.
    pretrain_epochs, finetune_epochs : int
        Maximum number of epochs for ME pretraining and for MI training.
    patience_pretrain, patience_finetune : int
        A training stage stops early after this many epochs without a lower validation
        loss.
    pretrain_lr, finetune_lr : float
        AdamW learning rates for ME pretraining and for MI training.
    weight_decay : float
        AdamW weight decay.
    grad_clip_norm : float
        Gradients are scaled down when their total norm is larger than this.
    lr_decay_factor : float
        The learning rate is multiplied by this factor when the validation loss stops
        improving.
    lr_decay_patience : int
        Number of epochs without improvement that are allowed before the learning rate
        decays.
    latent_dim : int
        Number of feature channels inside the models.
    dropout : float
        Dropout rate used throughout the models.
    lag_radius : int
        Baseline 5: the lag-aware fusion compares the EEG with the IMU shifted by up to
        this many samples in either direction.
    cross_attention_heads : int
        Number of attention heads in the cross-modal fusion of Baseline 6.
    normal_mha_heads : int
        Number of attention heads in the self-attention fusion of Baseline 7.
    corr_loss_weight : float
        The regression loss is the Huber loss plus this weight times (1 - correlation).
        The correlation term rewards getting the shape of each velocity curve right.
    huber_beta : float
        Threshold of the Huber part of the regression loss.
    teacher_loss_weight, student_sup_loss_weight, kd_loss_weight : float
        Baselines 4 and 5 train a teacher and a student together. These weights set how
        much the teacher's error, the student's error and the gap between the student
        and the teacher count in the loss.
    kd_beta : float
        SmoothL1 threshold for the gap between the student and the teacher.
    proto_loss_weight : float
        Baseline 5: weight of the term that pulls MI features toward the ME prototype of
        the same target.
    baseline0_use_ridge : bool
        Baseline 0: use ridge regression. If False, use ordinary least squares.
    baseline0_ridge_alpha : float
        Baseline 0: strength of the ridge penalty.
    linear_lags_sec : list of float
        Baseline 0: lags in seconds. The velocity at time t is predicted from the EEG
        around t + lag for each lag, so -0.3 means 300 ms before t.
    linear_lag_window_sec : float
        Baseline 0: length in seconds of the EEG window averaged around each lag.
    """

    subject: str = "s026"
    data_dir: Path = PROJ_ROOT / "data" / "processed"
    results_dir: Path = PROJ_ROOT / "reports" / "baseline_results"

    baselines: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 6, 7], metadata={"choices": BASELINE_IDS}
    )
    target_stream: str = field(default="all_xyz", metadata={"choices": TARGET_STREAMS})

    use_filterbank: bool = True
    eeg_clip_value: float = 8.0
    test_fraction: float = 0.15
    val_fraction: float = 0.1765

    device: str = "auto"
    seed: int = 42
    deterministic: bool = True
    batch_size: int = 8
    pretrain_epochs: int = 80
    finetune_epochs: int = 60
    patience_pretrain: int = 12
    patience_finetune: int = 12
    pretrain_lr: float = 1e-3
    finetune_lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 4

    latent_dim: int = 64
    dropout: float = 0.20
    lag_radius: int = 3
    cross_attention_heads: int = 4
    normal_mha_heads: int = 4

    corr_loss_weight: float = 0.10
    huber_beta: float = 0.5
    teacher_loss_weight: float = 1.0
    student_sup_loss_weight: float = 1.0
    kd_loss_weight: float = 0.5
    kd_beta: float = 0.5
    proto_loss_weight: float = 0.2

    baseline0_use_ridge: bool = True
    baseline0_ridge_alpha: float = 10.0
    linear_lags_sec: list[float] = field(default_factory=lambda: [-0.300, -0.200, -0.100, 0.000])
    linear_lag_window_sec: float = 0.050

    def __post_init__(self):
        """Check the settings and turn the folder settings into `Path` objects."""
        if not isinstance(self.subject, str):
            raise TypeError(f"subject must be one ID such as 's026', got {self.subject!r}")
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
        self._check_choices()
        self._check_ranges()

    def _check_choices(self):
        for f in fields(self):
            choices = f.metadata.get("choices")
            if choices is None:
                continue
            value = getattr(self, f.name)
            for item in value if isinstance(value, list) else [value]:
                if item not in choices:
                    raise ValueError(f"{f.name}: {item!r} is not one of {list(choices)}")

        if not re.fullmatch(r"auto|cpu|cuda(:\d+)?", self.device):
            raise ValueError(
                f"device: {self.device!r} must be 'auto', 'cpu', 'cuda' or 'cuda:<index>'"
            )

    def _check_ranges(self):
        for name in ("test_fraction", "val_fraction"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be between 0 and 1, got {getattr(self, name)}")
        for name in (
            "batch_size",
            "pretrain_epochs",
            "finetune_epochs",
            "patience_pretrain",
            "patience_finetune",
            "latent_dim",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")

    @property
    def result_xlsx_path(self) -> Path:
        """Path of the Excel file with this run's results."""
        return self.results_dir / f"baseline_results_{self.subject}_multihead.xlsx"


def add_config_arguments(parser: argparse.ArgumentParser, skip=("subject",)):
    """Add a command-line option for each `Config` field.

    Field names become options with dashes, so ``pretrain_epochs`` becomes
    ``--pretrain-epochs``. Boolean fields get a pair of options, such as
    ``--use-filterbank`` and ``--no-use-filterbank``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the options to.
    skip : tuple of str
        Fields to leave out.
    """
    defaults = Config()
    for f in fields(Config):
        if f.name in skip:
            continue
        flag = "--" + f.name.replace("_", "-")
        help_text = f"default: {getattr(defaults, f.name)}"
        choices = f.metadata.get("choices")
        if f.type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, help=help_text)
        elif typing.get_origin(f.type) is list:
            (item_type,) = typing.get_args(f.type)
            parser.add_argument(flag, nargs="+", type=item_type, choices=choices, help=help_text)
        else:
            parser.add_argument(flag, type=f.type, choices=choices, help=help_text)


def config_from_args(args: argparse.Namespace, skip=("subject",), **overrides) -> Config:
    """Build a `Config` from parsed command-line arguments.

    Options that were not given keep their default values.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed with the options from `add_config_arguments`.
    skip : tuple of str
        Fields not to read from `args`. ``subject`` is skipped by default because the
        command line accepts several subjects; pass one in `overrides` instead.
    **overrides
        Field values that take precedence over `args`.

    Returns
    -------
    Config

    Raises
    ------
    ValueError
        If a setting has an invalid value.
    TypeError
        If `subject` is not a single string.
    """
    values = {
        f.name: getattr(args, f.name)
        for f in fields(Config)
        if f.name not in skip and getattr(args, f.name, None) is not None
    }
    return Config(**{**values, **overrides})
