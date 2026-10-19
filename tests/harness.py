"""Helpers for the golden-output regression test (test_golden.py, make_golden.py).

Both run the pipeline in a subprocess on CPU, on small mock data, with every baseline
for a few epochs:
- the current package through its command line (`python -m mtp.modeling.train ...`);
- an old single-file version of the script from git, whose settings are plain
  assignments such as `PRETRAIN_EPOCHS = 80` that we rewrite in a temp copy.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mtp.dataset import DATA_FILES

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
SUBJECT = "mock"
# The "config" sheet contains temp-folder paths, so it is not compared.
SHEETS = ["summary", "axis_metrics", "stages"]


def mock_cli_args(data_dir):
    """Command-line options that run every baseline for a few epochs on the mock data."""
    return [
        "--subject", SUBJECT,
        "--data-dir", str(data_dir),
        "--results-dir", str(result_path(data_dir).parent),
        "--baselines", *[str(i) for i in range(8)],
        "--pretrain-epochs", "3",
        "--finetune-epochs", "3",
        "--patience-pretrain", "1",
        "--patience-finetune", "1",
        "--device", "cpu",
    ]


def legacy_script_settings(data_dir):
    """The same settings for the original single-file script (see run_script)."""
    return {
        "subject": SUBJECT,
        "DATA_DIR": str(data_dir),
        **{f"RUN_BASELINE_{i}": True for i in range(8)},
        "PRETRAIN_EPOCHS": 3,
        "FINETUNE_EPOCHS": 3,
        "PATIENCE_PRETRAIN": 1,
        "PATIENCE_FINETUNE": 1,
    }


def write_mock_data(data_dir, subject=SUBJECT, n_channels=8, n_samples=4609, seed=0, layout="flat"):
    """Write a small random dataset in the same format as step 3's output.

    Real data: EEG (N, 64, 4609) in volts, labels 101-104, and a trajectory dict of
    (N, 150, 3|6) arrays. Here N is 48 per condition and there are 8 channels.
    layout: "flat" (all files in data_dir, as on the lab PC) or "subfolders" (as in
    data/processed).
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
            "wirst": rng.normal(size=(n, 150, 3)),  # key spelling used by the pipeline
            "IMU": rng.normal(size=(n, 150, 6)),
        }
        for kind, array in {"eeg": eeg, "trajectory": trajectory, "labels": labels}.items():
            subdir, pattern = DATA_FILES[kind]
            folder = data_dir / subdir if layout == "subfolders" else data_dir
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / pattern.format(subject=subject, condition=condition), array, allow_pickle=True)


def patch_settings(path, settings):
    """Rewrite each top-level `NAME = ...` line in `path`; each must appear exactly once."""
    text = path.read_text(encoding="utf-8")
    for name, value in settings.items():
        pattern = re.compile(rf"^{re.escape(name)}\s*=.*$", re.MULTILINE)
        text, count = pattern.subn(lambda _: f"{name} = {value!r}", text)
        if count != 1:
            raise RuntimeError(f"expected one `{name} = ...` line in {path}, found {count}")
    path.write_text(text, encoding="utf-8")


def run_package(data_dir):
    """Run the current `mtp` package on the mock data in `data_dir`."""
    return run_cli(mock_cli_args(data_dir))


def run_cli(args, check=True):
    """Run `python -m mtp.modeling.train <args>` on CPU and return the finished process."""
    return _run([sys.executable, "-m", "mtp.modeling.train", *args], REPO_ROOT, check=check)


def run_script(workdir, git_spec, settings):
    """Run a single-file version of the pipeline taken from git, e.g. 'REF:path/to/script.py'."""
    source = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", git_spec], capture_output=True, check=True
    ).stdout
    script = workdir / "pipeline.py"
    script.write_bytes(source)
    patch_settings(script, settings)
    return _run([sys.executable, str(script)], workdir)


def _run(cmd, cwd, check=True):
    # Hide the GPU: GPU training is not reproducible run to run. "-1" is needed on
    # Windows, where an empty CUDA_VISIBLE_DEVICES still exposes the GPU.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8")
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"pipeline failed (exit {proc.returncode})\n"
            f"--- stdout (tail) ---\n{proc.stdout[-3000:]}\n--- stderr (tail) ---\n{proc.stderr[-3000:]}"
        )
    return proc


def result_path(data_dir, subject=SUBJECT):
    return Path(data_dir) / "baseline_results" / f"baseline_results_{subject}_multihead.xlsx"


def read_results(xlsx_path):
    return {sheet: pd.read_excel(xlsx_path, sheet_name=sheet) for sheet in SHEETS}


def save_golden(results, golden_dir=GOLDEN_DIR):
    golden_dir.mkdir(exist_ok=True)
    for sheet, df in results.items():
        df.to_csv(golden_dir / f"{sheet}.csv", index=False)


def load_golden(golden_dir=GOLDEN_DIR):
    return {sheet: pd.read_csv(golden_dir / f"{sheet}.csv") for sheet in SHEETS}


def assert_same_results(actual, expected, rtol=1e-5, atol=1e-8):
    for sheet in SHEETS:
        a, e = actual[sheet], expected[sheet]
        assert list(a.columns) == list(e.columns), f"{sheet}: columns differ"
        assert len(a) == len(e), f"{sheet}: {len(a)} rows, expected {len(e)}"
        for col in e.columns:
            if pd.api.types.is_numeric_dtype(e[col]):
                np.testing.assert_allclose(
                    a[col].to_numpy(dtype=float), e[col].to_numpy(dtype=float),
                    rtol=rtol, atol=atol, equal_nan=True, err_msg=f"{sheet}.{col}",
                )
            else:
                assert a[col].astype(str).tolist() == e[col].astype(str).tolist(), f"{sheet}.{col}"
