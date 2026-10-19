"""
Train and evaluate baselines 0-7 (see mtp/modeling/baselines.py) and save the ME/MI
results of each subject to an Excel file.

Examples (from the repo root):
    python -m mtp.modeling.train
    python -m mtp.modeling.train --subject s008 s009 --baselines 3 6 7
    python -m mtp.modeling.train --subject all --data-dir D:/MTP_task/model --skip-existing

Input: for each subject and condition (ME, MI), the step 3 outputs
    EEG_data_<subject>_<cond>.npy       EEG epochs covering -2..7 s
    trajectory_<subject>_<cond>.npy     dict of (epoch, 150, 3|6) arrays aligned with EEG
    labels_clean_<subject>_<cond>.npy   target labels 101-104
in --data-dir, either all in one folder or in the data/processed subfolders.
Filter-bank filtering is applied to the full -2..7 s epoch first, then the model input
is cropped to 0..5 s.

Output: <results-dir>/baseline_results_<subject>_multihead.xlsx with sheets
summary, axis_metrics, stages and config, and a .log file with the same name.
With several subjects, a failing subject is logged and the run continues; the exit
code is 1 if any subject failed.
"""
import argparse
from dataclasses import replace
import logging
import sys
import time

from mtp.config import Config, add_config_arguments, config_from_args
from mtp.dataset import list_subjects
from mtp.modeling.baselines import BASELINES
from mtp.modeling.experiment import Experiment, resolve_device, set_seed

log = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATEFMT = "%H:%M:%S"


def run_experiment(cfg: Config):
    """Run cfg.baselines for cfg.subject; save the Excel results and a log file next to them."""
    log_path = cfg.result_xlsx_path.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, "%Y-%m-%d " + LOG_DATEFMT))
    logging.getLogger().addHandler(file_handler)
    try:
        start_time = time.time()
        device = resolve_device(cfg.device)
        log.info("Subject %s | device %s | baselines %s", cfg.subject, device, sorted(set(cfg.baselines)))

        set_seed(cfg.seed)
        exp = Experiment(cfg, device)

        for baseline_id in sorted(set(cfg.baselines)):
            # Every baseline starts from the same seed, so its results do not depend on
            # which other baselines run.
            set_seed(cfg.seed)
            BASELINES[baseline_id](exp)

        exp.results.save(cfg.result_xlsx_path)
        log.info("Subject %s finished in %.2f minutes.", cfg.subject, (time.time() - start_time) / 60)
    except Exception:
        log.exception("Subject %s failed", cfg.subject)
        raise
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--subject", nargs="+", default=[Config.subject],
        help=f"subject IDs, or 'all' for every subject in --data-dir (default: {Config.subject})",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="skip subjects whose results file already exists in --results-dir",
    )
    add_config_arguments(parser)
    args = parser.parse_args(argv)

    try:
        base_cfg = config_from_args(args)
    except ValueError as err:
        parser.error(str(err))

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATEFMT, stream=sys.stderr)

    subjects = list_subjects(base_cfg.data_dir) if args.subject == ["all"] else args.subject
    failed = []
    for subject in subjects:
        cfg = replace(base_cfg, subject=subject)
        if args.skip_existing and cfg.result_xlsx_path.exists():
            log.info("Skipping %s: %s already exists.", subject, cfg.result_xlsx_path)
            continue
        try:
            run_experiment(cfg)
        except Exception:  # already logged with its traceback; continue with the next subject
            failed.append(subject)

    if failed:
        log.error("%d of %d subjects failed: %s (see their .log files)", len(failed), len(subjects), ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
