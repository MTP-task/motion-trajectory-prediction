"""Regenerate tests/golden/, the expected results used by test_golden.py.

Only do this when you intend to change results, and say why in the commit message.

    python tests/make_golden.py            # from the current mtp package
    python tests/make_golden.py --script 4fa608d:src/model/step4_model_training_MHA.py
"""

import argparse
from pathlib import Path
import sys
import tempfile

# Lets this script import mtp from the repo even when the package is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harness


def main():
    """Run the pipeline on mock data and save its results as the new expected results."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--script",
        metavar="REF:PATH",
        help="run a single-file pipeline from git instead of the current package",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        data_dir = workdir / "data"
        data_dir.mkdir()
        harness.write_mock_data(data_dir)
        if args.script:
            harness.run_script(workdir, args.script, harness.legacy_script_settings(data_dir))
        else:
            harness.run_package(data_dir)
        results = harness.read_results(harness.result_path(data_dir))

    harness.save_golden(results)
    print(f"Wrote golden results to {harness.GOLDEN_DIR}")


if __name__ == "__main__":
    main()
