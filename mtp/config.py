"""Project settings.

Module-level constants describe the recordings and the epoch timing; they are the same
for every run. `Config` holds the experiment settings for one run (one subject). Every
`Config` field can be overridden from the command line, e.g. `--pretrain-epochs 5`
(see `python -m mtp.modeling.train --help`).
"""
import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path
import re
import typing

PROJ_ROOT = Path(__file__).resolve().parents[1]

# ---- Recording and epoch timing ---------------------------------------------

EEG_FS = 512.0
TRAJ_FS = 30.0
EPOCH_SEC = 5.0
TARGET_LEN = int(TRAJ_FS * EPOCH_SEC)  # 150
IMU_DIM = 6  # elbow + wrist rotational velocity, x/y/z each

# EEG input epoch timing.
# Input EEG epoch = -2..7 s, model uses 0..5 s after filter-bank filtering.
EEG_EPOCH_TMIN = -2.0
EEG_CROP_TMIN = 0.0
EEG_CROP_TMAX = 5.0
MODEL_EEG_SEC = EEG_CROP_TMAX - EEG_CROP_TMIN
MODEL_EEG_SAMPLES = int(round(EEG_FS * MODEL_EEG_SEC))  # 2560

BANDS = {
    "slow": (0.5, 1.0),
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
    # If final low-pass is exactly 40 Hz, change this to (30.0, 40.0).
    "low_gamma": (30.0, 45.0),
}

# ---- Targets and labels -----------------------------------------------------

# Trajectory dict keys written by step 3 ("wirst" is the spelling used in the files).
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

TARGET_STREAMS = ("wrist", "elbow", "shoulder", "imu", "all_xyz", "all_kin")
BASELINE_IDS = tuple(range(8))


# ---- Experiment settings ----------------------------------------------------

@dataclass
class Config:
    # Data and output
    subject: str = "s026"
    data_dir: Path = PROJ_ROOT / "data" / "processed"
    results_dir: Path = PROJ_ROOT / "reports" / "baseline_results"

    # Baselines to run (0-7, described in mtp/modeling/baselines.py)
    baselines: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 6, 7], metadata={"choices": BASELINE_IDS}
    )

    # Main output target:
    # "wrist" / "elbow" / "shoulder" -> Vx,Vy,Vz of that joint (3 dims); "imu" -> 6 dims
    # "all_xyz" -> shoulder+elbow+wrist (9 dims); "all_kin" -> all_xyz + IMU (15 dims)
    target_stream: str = field(default="all_xyz", metadata={"choices": TARGET_STREAMS})

    # Preprocessing and split
    use_filterbank: bool = True
    eeg_clip_value: float = 8.0  # z-scored EEG is clipped to +-this value
    test_fraction: float = 0.15  # of all trials
    val_fraction: float = 0.1765  # of the trials left after the test split (about 15% of all)

    # Training
    device: str = "auto"  # "auto" (GPU if available), "cpu", "cuda" or "cuda:<index>"
    seed: int = 42
    batch_size: int = 8
    pretrain_epochs: int = 80
    finetune_epochs: int = 60
    patience_pretrain: int = 12
    patience_finetune: int = 12
    pretrain_lr: float = 1e-3
    finetune_lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    lr_decay_factor: float = 0.5  # multiply the learning rate by this when validation loss plateaus
    lr_decay_patience: int = 4  # epochs without improvement before the learning rate decays

    # Deep model settings
    latent_dim: int = 64
    dropout: float = 0.20
    lag_radius: int = 3
    cross_attention_heads: int = 4  # Used by Baseline 6 bidirectional cross-modal MHA fusion.
    normal_mha_heads: int = 4  # Used by Baseline 7 normal MHA fusion.

    # Loss settings
    corr_loss_weight: float = 0.10  # regression loss = Huber + this * (1 - correlation)
    huber_beta: float = 0.5
    teacher_loss_weight: float = 1.0
    student_sup_loss_weight: float = 1.0
    kd_loss_weight: float = 0.5
    kd_beta: float = 0.5  # SmoothL1 beta of the distillation loss
    proto_loss_weight: float = 0.2

    # Baseline 0: time-lagged MLR.
    # Each velocity sample y(t) is predicted from EEG features around t + lag.
    # Negative lags mean EEG before the target velocity sample.
    # Example: -0.300 means EEG 300 ms before the current velocity sample.
    baseline0_use_ridge: bool = True
    baseline0_ridge_alpha: float = 10.0
    linear_lags_sec: list[float] = field(default_factory=lambda: [-0.300, -0.200, -0.100, 0.000])
    linear_lag_window_sec: float = 0.050  # average EEG within ±25 ms around each lag center

    def __post_init__(self):
        if not isinstance(self.subject, str):
            raise ValueError(f"subject must be one ID such as 's026', got {self.subject!r}")
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)

        for f in fields(self):
            choices = f.metadata.get("choices")
            if choices is None:
                continue
            value = getattr(self, f.name)
            for item in value if isinstance(value, list) else [value]:
                if item not in choices:
                    raise ValueError(f"{f.name}: {item!r} is not one of {list(choices)}")

        if not re.fullmatch(r"auto|cpu|cuda(:\d+)?", self.device):
            raise ValueError(f"device: {self.device!r} must be 'auto', 'cpu', 'cuda' or 'cuda:<index>'")
        for name in ("test_fraction", "val_fraction"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be between 0 and 1, got {getattr(self, name)}")
        for name in ("batch_size", "pretrain_epochs", "finetune_epochs", "patience_pretrain",
                     "patience_finetune", "latent_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")

    @property
    def result_xlsx_path(self) -> Path:
        return self.results_dir / f"baseline_results_{self.subject}_multihead.xlsx"


def add_config_arguments(parser: argparse.ArgumentParser, skip=("subject",)):
    """Add a `--field-name` option for every Config field (defaults come from Config)."""
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
    """
    Build a Config from parsed arguments; options left unset keep their defaults.
    Fields in `skip` are not read from `args` (pass them as overrides instead).
    Raises ValueError for invalid settings.
    """
    values = {
        f.name: getattr(args, f.name)
        for f in fields(Config)
        if f.name not in skip and getattr(args, f.name, None) is not None
    }
    return Config(**{**values, **overrides})
