"""Shared setup for one subject's run: data, loaders, training settings and results."""

import logging
import os
import random

import numpy as np
import torch

from mtp.config import Config
from mtp.dataset import (
    Loaders,
    fit_target_scaler,
    make_loaders,
    normalize_targets,
    prepare_dataset,
)
from mtp.modeling.metrics import SequenceRegressionLoss
from mtp.modeling.results import ResultsLog
from mtp.modeling.trainers import TrainConfig, fit, run_epoch

log = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Seed the Python, NumPy and PyTorch random number generators, on CPU and GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic(enabled: bool):
    """Turn PyTorch's deterministic mode on or off.

    Some GPU operations add up numbers in a different order on every run, so GPU
    training gives slightly different results each time, even with the same seed. In
    deterministic mode, PyTorch only uses operations that always give the same result,
    and raises an error if the model needs one that has no deterministic version.

    Parameters
    ----------
    enabled : bool
        Turn deterministic mode on (True) or off (False).
    """
    if enabled:
        # cuBLAS reads this when it starts, so it has to be set before the first GPU
        # operation. Without a fixed workspace, cuBLAS is not deterministic.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = enabled


def resolve_device(device: str) -> str:
    """Turn "auto" into "cuda" when a GPU is available and into "cpu" otherwise."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class Experiment:
    """Everything the baselines share for one subject.

    Building an Experiment loads and prepares the subject's ME and MI data, fits one
    target scaler on the ME and MI training trials together, and makes the loaders.

    Parameters
    ----------
    cfg : Config
        Run settings.
    device : str
        Device to train on, such as "cpu" or "cuda".

    Attributes
    ----------
    me, mi : PreparedCondition
        The prepared ME and MI data, with normalized targets.
    me_loaders : Loaders
        ME batches with EEG and IMU.
    me_eeg_loaders : Loaders
        ME batches with EEG only.
    mi_loaders : Loaders
        MI batches, which have EEG only.
    pretrain, finetune : TrainConfig
        Settings for ME pretraining and for MI training.
    results : ResultsLog
        Collects the result rows of all baselines.
    """

    def __init__(self, cfg: Config, device: str):
        self.cfg = cfg
        self.device = device

        self.me = prepare_dataset(cfg, "ME", with_imu=True)
        self.mi = prepare_dataset(cfg, "MI", with_imu=False)
        for name in ("n_bands", "n_channels", "out_dim"):
            me_value, mi_value = getattr(self.me, name), getattr(self.mi, name)
            if me_value != mi_value:
                raise ValueError(f"ME and MI have different {name}: {me_value} vs {mi_value}.")

        self.dim_names = self.me.dim_names
        if self.me.dim_names != self.mi.dim_names:
            log.warning(
                "ME and MI dim_names differ: ME %s, MI %s", self.me.dim_names, self.mi.dim_names
            )

        self.scaler = fit_target_scaler(self.me.train.y, self.mi.train.y)
        for condition in (self.me, self.mi):
            for split in condition.splits():
                split.y_norm = normalize_targets(split.y, self.scaler)

        self.me_loaders = make_loaders(self.me, cfg.batch_size, with_imu=True)
        self.me_eeg_loaders = make_loaders(self.me, cfg.batch_size)
        self.mi_loaders = make_loaders(self.mi, cfg.batch_size)

        schedule = {
            "device": device,
            "grad_clip_norm": cfg.grad_clip_norm,
            "lr_decay_factor": cfg.lr_decay_factor,
            "lr_decay_patience": cfg.lr_decay_patience,
        }
        self.pretrain = TrainConfig(
            cfg.pretrain_lr,
            cfg.weight_decay,
            cfg.pretrain_epochs,
            cfg.patience_pretrain,
            **schedule,
        )
        self.finetune = TrainConfig(
            cfg.finetune_lr,
            cfg.weight_decay,
            cfg.finetune_epochs,
            cfg.patience_finetune,
            **schedule,
        )
        self.results = ResultsLog(cfg, self.dim_names)

    def regression_loss(self):
        """Return a new regression loss with the weights from the run settings."""
        return SequenceRegressionLoss(self.cfg.corr_loss_weight, self.cfg.huber_beta)

    @property
    def model_dims(self) -> dict:
        """Numbers of filter bands, EEG channels and outputs, for the model classes."""
        return {
            "n_bands": self.me.n_bands,
            "n_channels": self.me.n_channels,
            "out_dim": self.me.out_dim,
        }

    def fit(self, stage_name, model, loaders: Loaders, objective, train_cfg: TrainConfig):
        """Train a model on one set of loaders and record the per-epoch rows.

        Parameters
        ----------
        stage_name : str
            Name of the stage in the logs and the "stages" sheet.
        model : nn.Module
            Model to train. It keeps the weights with the lowest validation loss.
        loaders : Loaders
            For example ``self.mi_loaders``: the model trains on ``loaders.train`` and
            stops early based on ``loaders.val``.
        objective : callable
            One of the objectives in `mtp.modeling.objectives`.
        train_cfg : TrainConfig
            Usually ``self.pretrain`` or ``self.finetune``.
        """
        rows = fit(
            stage_name,
            model,
            loaders.train,
            loaders.val,
            objective,
            self.scaler,
            self.dim_names,
            train_cfg,
        )
        self.results.stage_rows.extend(rows)

    def evaluate(self, model, loader, objective):
        """Score a model on one loader, such as ``self.mi_loaders.test``, without training it.

        Returns
        -------
        EpochResult
        """
        return run_epoch(model, loader, objective, self.scaler, self.dim_names, None, self.device)

    def record(self, baseline_id, baseline_name, metrics, notes, evaluation_task, branch):
        """Add one evaluation to the results; see `ResultsLog.add_evaluation`."""
        self.results.add_evaluation(
            baseline_id, baseline_name, metrics, notes, evaluation_task, branch
        )
