"""End-to-end runs of the command line on mock data, 1 epoch per stage, on CPU."""

import pytest
import torch

import harness

pytestmark = pytest.mark.slow


def run(data_dir, *extra):
    args = [
        "--data-dir",
        str(data_dir),
        "--results-dir",
        str(data_dir / "out"),
        "--pretrain-epochs",
        "1",
        "--finetune-epochs",
        "1",
        "--device",
        "cpu",
        *extra,
    ]
    return harness.run_cli(args, check=False)


@pytest.mark.parametrize(
    "extra",
    [
        ["--target-stream", "wrist", "--baselines", "0", "1", "3"],
        ["--target-stream", "all_kin", "--baselines", "2", "5"],
        ["--target-stream", "imu", "--baselines", "1"],
        ["--no-use-filterbank", "--baselines", "0", "4", "6", "7"],
    ],
    ids=["wrist", "all_kin", "imu", "no-filterbank"],
)
def test_options_run_end_to_end(tmp_path, extra):
    harness.write_mock_data(tmp_path)
    proc = run(tmp_path, "--subject", harness.SUBJECT, *extra)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = tmp_path / "out"
    assert (out / "baseline_results_mock_multihead.xlsx").exists()
    assert "RESULT" in (out / "baseline_results_mock_multihead.log").read_text(encoding="utf-8")


def test_subfolder_layout_and_subject_all(tmp_path):
    harness.write_mock_data(tmp_path, subject="s001", layout="subfolders")
    proc = run(tmp_path, "--subject", "all", "--baselines", "0")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (tmp_path / "out" / "baseline_results_s001_multihead.xlsx").exists()


def test_failed_subject_does_not_stop_the_others(tmp_path):
    harness.write_mock_data(tmp_path, subject="s001")
    harness.write_mock_data(tmp_path, subject="s002")
    (tmp_path / "labels_clean_s001_MI.npy").unlink()

    proc = run(tmp_path, "--subject", "s001", "s002", "--baselines", "0")

    assert proc.returncode == 1
    assert "1 of 2 subjects failed: s001" in proc.stderr
    out = tmp_path / "out"
    assert (out / "baseline_results_s002_multihead.xlsx").exists()
    assert "FileNotFoundError" in (out / "baseline_results_s001_multihead.log").read_text(
        encoding="utf-8"
    )


def test_skip_existing(tmp_path):
    harness.write_mock_data(tmp_path, subject="s001")
    assert run(tmp_path, "--subject", "s001", "--baselines", "0").returncode == 0

    proc = run(tmp_path, "--subject", "s001", "--baselines", "0", "--skip-existing")
    assert proc.returncode == 0
    assert "Skipping s001" in proc.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_gpu_runs_are_repeatable(tmp_path):
    harness.write_mock_data(tmp_path)
    results = []
    for run_dir in ("first", "second"):
        out = tmp_path / run_dir
        args = [
            "--subject", harness.SUBJECT, "--data-dir", str(tmp_path), "--results-dir", str(out),
            "--baselines", "1", "3", "5", "6", "--pretrain-epochs", "2", "--finetune-epochs", "2",
            "--device", "cuda",
        ]  # fmt: skip
        harness.run_cli(args, gpu=True)
        results.append(harness.read_results(out / "baseline_results_mock_multihead.xlsx"))
    harness.assert_same_results(results[0], results[1], rtol=0, atol=0)
