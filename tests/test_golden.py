"""Golden-output regression test for the training pipeline.

Runs every baseline for a few epochs on small random data (CPU, fixed seed) and checks
that the result tables match tests/golden/. Those were produced by the original
single-file script (src/model/step4_model_training_MHA.py at commit 4fa608d).

A failure means the pipeline's behaviour changed. If the change is intended,
regenerate the expected results with `python tests/make_golden.py`.
"""
import harness
import pytest

pytestmark = pytest.mark.slow


def test_results_match_golden(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    harness.write_mock_data(data_dir)

    harness.run_package(data_dir)

    actual = harness.read_results(harness.result_path(data_dir))
    harness.assert_same_results(actual, harness.load_golden())
