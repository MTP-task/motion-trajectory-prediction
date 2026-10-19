"""
Train and evaluate baselines 0-7 (see mtp/modeling/baselines.py) and save the ME/MI
results of each subject to an Excel file.

Examples (from the repo root):
    python -m mtp.modeling.train
    python -m mtp.modeling.train --subject s008 s009 --baselines 3 6 7
    python -m mtp.modeling.train --subject all --data-dir D:/MTP_task/model

Input: for each subject and condition (ME, MI), the step 3 outputs
    EEG_data_<subject>_<cond>.npy       EEG epochs covering -2..7 s
    trajectory_<subject>_<cond>.npy     dict of (epoch, 150, 3|6) arrays aligned with EEG
    labels_clean_<subject>_<cond>.npy   target labels 101-104
in --data-dir, either all in one folder or in the data/processed subfolders.
Filter-bank filtering is applied to the full -2..7 s epoch first, then the model input
is cropped to 0..5 s.

Output: <results-dir>/baseline_results_<subject>_multihead.xlsx with sheets
summary, axis_metrics, stages and config.
"""
import argparse
from dataclasses import replace
import time

from mtp.config import Config, add_config_arguments, config_from_args
from mtp.dataset import list_subjects
from mtp.modeling.baselines import BASELINES
from mtp.modeling.experiment import Experiment, resolve_device, set_seed


def run_experiment(cfg: Config):
    """Run cfg.baselines for cfg.subject and save the Excel results."""
    start_time = time.time()
    device = resolve_device(cfg.device)
    print(f"Subject: {cfg.subject} | Device: {device}")

    unknown = sorted(set(cfg.baselines) - set(BASELINES))
    if unknown:
        raise ValueError(f"Unknown baselines {unknown}; choose from {sorted(BASELINES)}.")

    set_seed(cfg.seed)
    exp = Experiment(cfg, device)

    for baseline_id in sorted(set(cfg.baselines)):
        # Every baseline starts from the same seed, so its results do not depend on
        # which other baselines run.
        set_seed(cfg.seed)
        BASELINES[baseline_id](exp)

    exp.results.save(cfg.result_xlsx_path)
    print(f"\nAll requested baselines finished in {(time.time() - start_time) / 60:.2f} minutes.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--subject", nargs="+", default=[Config.subject],
        help=f"subject IDs, or 'all' for every subject in --data-dir (default: {Config.subject})",
    )
    add_config_arguments(parser)
    args = parser.parse_args(argv)

    base_cfg = config_from_args(args)
    subjects = list_subjects(base_cfg.data_dir) if args.subject == ["all"] else args.subject
    for subject in subjects:
        run_experiment(replace(base_cfg, subject=subject))


if __name__ == "__main__":
    main()
