"""Regenerate tests/golden/, the expected results used by test_golden.py.

Only do this when you intend to change results, and say why in the commit message.

    python tests/make_golden.py
"""

from pathlib import Path
import sys
import tempfile

# Lets this script import mtp from the repo even when the package is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harness


def main():
    """Run the pipeline on mock data and save its results as the new expected results."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        data_dir = workdir / "data"
        data_dir.mkdir()
        harness.write_mock_data(data_dir)
        harness.run_package(data_dir)
        results = harness.read_results(harness.result_path(data_dir))

    harness.save_golden(results)
    print(f"Wrote golden results to {harness.GOLDEN_DIR}")


if __name__ == "__main__":
    main()
