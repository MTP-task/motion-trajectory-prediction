"""Shared setup for one subject's run: data, loaders, training settings and results."""

import logging
import random

import numpy as np
import torch

from mtp.config import Config
from mtp.dataset import fit_target_scaler, make_loader, normalize_targets, prepare_dataset
from mtp.modeling.metrics import SequenceRegressionLoss
from mtp.modeling.results import ResultsLog
from mtp.modeling.trainers import TrainConfig, fit, run_epoch

log = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class Experiment:
    """
    Everything the baselines share for one subject.

    Loaders (all built from targets normalized with one scaler fitted on ME+MI train):
        me_train / me_val / me_test              ME EEG + IMU
        me_eeg_train / me_eeg_val / me_eeg_test  ME EEG only
        mi_train / mi_val / mi_test              MI EEG only
    """

    def __init__(self, cfg: Config, device: str):
        self.cfg = cfg
        self.device = device

        self.me = prepare_dataset(cfg, "ME", with_imu=True)
        self.mi = prepare_dataset(cfg, "MI", with_imu=False)
        for key in ("n_bands", "n_channels", "out_dim"):
            if self.me[key] != self.mi[key]:
                raise ValueError(
                    f"ME and MI have different {key}: {self.me[key]} vs {self.mi[key]}."
                )

        self.dim_names = self.me["dim_names"]
        if self.me["dim_names"] != self.mi["dim_names"]:
            log.warning(
                "ME and MI dim_names differ: ME %s, MI %s",
                self.me["dim_names"],
                self.mi["dim_names"],
            )

        self.scaler = fit_target_scaler(self.me["y_train"], self.mi["y_train"])
        for data in (self.me, self.mi):
            for split in ("train", "val", "test"):
                data[f"y_{split}_n"] = normalize_targets(data[f"y_{split}"], self.scaler)

        self.loaders = {}
        for split in ("train", "val", "test"):
            self.loaders[f"me_{split}"] = make_loader(
                self.me, split, cfg.batch_size, with_imu=True
            )
            self.loaders[f"me_eeg_{split}"] = make_loader(self.me, split, cfg.batch_size)
            self.loaders[f"mi_{split}"] = make_loader(self.mi, split, cfg.batch_size)

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
        """The regression loss used by every deep baseline (see Config loss settings)."""
        return SequenceRegressionLoss(self.cfg.corr_loss_weight, self.cfg.huber_beta)

    @property
    def model_dims(self) -> dict:
        """Input/output sizes shared by every model constructor."""
        return {
            "n_bands": self.me["n_bands"],
            "n_channels": self.me["n_channels"],
            "out_dim": self.me["out_dim"],
        }

    def fit(self, stage_name, model, loaders: str, objective, train_cfg: TrainConfig):
        """Train on `<loaders>_train`, early-stop on `<loaders>_val` (e.g. loaders="mi")."""
        rows = fit(
            stage_name,
            model,
            self.loaders[f"{loaders}_train"],
            self.loaders[f"{loaders}_val"],
            objective,
            self.scaler,
            self.dim_names,
            train_cfg,
        )
        self.results.stage_rows.extend(rows)

    def evaluate(self, model, loader: str, objective):
        return run_epoch(
            model, self.loaders[loader], objective, self.scaler, self.dim_names, None, self.device
        )

    def record(self, baseline_id, baseline_name, metrics, notes, evaluation_task, branch):
        self.results.add_evaluation(
            baseline_id, baseline_name, metrics, notes, evaluation_task, branch
        )
