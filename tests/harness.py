"""Helpers shared by the tests: mock data, running the pipeline, and comparing results.

The pipeline runs in a subprocess through its command line, as
``python -m mtp.modeling.train ...``.
"""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from mtp.dataset import DATA_FILES

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
SUBJECT = "mock"
# The "config" sheet contains temp-folder paths, so it is not compared.
SHEETS = ["summary", "axis_metrics", "stages"]


def mock_cli_args(data_dir):
    """Return command-line options that run every baseline for 3 epochs on mock data."""
    return [
        "--subject",
        SUBJECT,
        "--data-dir",
        str(data_dir),
        "--results-dir",
        str(result_path(data_dir).parent),
        "--baselines",
        *[str(i) for i in range(8)],
        "--pretrain-epochs",
        "3",
        "--finetune-epochs",
        "3",
        "--patience-pretrain",
        "1",
        "--patience-finetune",
        "1",
        "--device",
        "cpu",
    ]


def write_mock_data(
    data_dir, subject=SUBJECT, n_channels=8, n_samples=4609, seed=0, layout="flat"
):
    """Write a small random dataset in the format of step 3's output.

    The real data has 64 EEG channels and about 240 trials per condition. The mock data
    has `n_channels` channels and 48 trials per condition, with the same epoch length,
    labels and trajectory keys.

    Parameters
    ----------
    data_dir : str or Path
        Folder to write to.
    subject : str
        Subject ID used in the file names.
    n_channels, n_samples : int
        Size of each EEG epoch.
    seed : int
        Random seed, so the data is the same on every call.
    layout : {"flat", "subfolders"}
        All files in `data_dir`, as on the lab PC, or in the subfolders used in
        data/processed.
    """
    data_dir = Path(data_dir)
    rng = np.random.default_rng(seed)
    class_counts = {"ME": [12, 12, 11, 13], "MI": [12, 12, 12, 12]}
    for condition, counts in class_counts.items():
        labels = np.repeat(np.array([101, 102, 103, 104], dtype=np.int32), counts)
        rng.shuffle(labels)
        n = len(labels)
        eeg = rng.normal(scale=2e-5, size=(n, n_channels, n_samples))
        trajectory = {
            "shoulder": rng.normal(size=(n, 150, 3)),
            "elbow": rng.normal(size=(n, 150, 3)),
            "wirst": rng.normal(size=(n, 150, 3)),  # step 3 files spell the wrist key this way
            "IMU": rng.normal(size=(n, 150, 6)),
        }
        for kind, array in {"eeg": eeg, "trajectory": trajectory, "labels": labels}.items():
            subdir, pattern = DATA_FILES[kind]
            folder = data_dir / subdir if layout == "subfolders" else data_dir
            folder.mkdir(parents=True, exist_ok=True)
            np.save(
                folder / pattern.format(subject=subject, condition=condition),
                array,
                allow_pickle=True,
            )


def run_package(data_dir):
    """Run the current `mtp` package on the mock data in `data_dir`."""
    return run_cli(mock_cli_args(data_dir))


def run_cli(args, check=True, gpu=False):
    """Run ``python -m mtp.modeling.train`` with `args` and return the finished process.

    Parameters
    ----------
    args : list of str
        Command-line arguments.
    check : bool
        Raise RuntimeError, with the end of the output, if the run fails.
    gpu : bool
        Let the run see the GPU. By default it is hidden, because the golden results
        are CPU results.
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if not gpu:
        # Golden results are CPU results; GPU arithmetic differs slightly. "-1" is needed
        # on Windows, where an empty CUDA_VISIBLE_DEVICES still exposes the GPU.
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    proc = subprocess.run(
        [sys.executable, "-m", "mtp.modeling.train", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"pipeline failed (exit {proc.returncode})\n"
            f"--- stdout (tail) ---\n{proc.stdout[-3000:]}\n--- stderr (tail) ---\n{proc.stderr[-3000:]}"
        )
    return proc


def result_path(data_dir, subject=SUBJECT):
    """Return where a run with `mock_cli_args` writes the Excel results."""
    return Path(data_dir) / "baseline_results" / f"baseline_results_{subject}_multihead.xlsx"


def read_results(xlsx_path):
    """Read the compared sheets of a results file into DataFrames, keyed by sheet name."""
    return {sheet: pd.read_excel(xlsx_path, sheet_name=sheet) for sheet in SHEETS}


def save_golden(results, golden_dir=GOLDEN_DIR):
    """Save results from `read_results` as the expected results, one CSV file per sheet."""
    golden_dir.mkdir(exist_ok=True)
    for sheet, df in results.items():
        df.to_csv(golden_dir / f"{sheet}.csv", index=False)


def load_golden(golden_dir=GOLDEN_DIR):
    """Load the expected results saved by `save_golden`."""
    return {sheet: pd.read_csv(golden_dir / f"{sheet}.csv") for sheet in SHEETS}


def assert_same_results(actual, expected, rtol=1e-5, atol=1e-8):
    """Check that two sets of results have the same columns, rows and values.

    Numbers must match within `rtol` and `atol`; text must match exactly.
    """
    for sheet in SHEETS:
        a, e = actual[sheet], expected[sheet]
        assert list(a.columns) == list(e.columns), f"{sheet}: columns differ"
        assert len(a) == len(e), f"{sheet}: {len(a)} rows, expected {len(e)}"
        for col in e.columns:
            if pd.api.types.is_numeric_dtype(e[col]):
                np.testing.assert_allclose(
                    a[col].to_numpy(dtype=float),
                    e[col].to_numpy(dtype=float),
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                    err_msg=f"{sheet}.{col}",
                )
            else:
                assert a[col].astype(str).tolist() == e[col].astype(str).tolist(), f"{sheet}.{col}"
