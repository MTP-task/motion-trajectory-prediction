# motion-trajectory-prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predicting upper-limb motion trajectories from EEG and IMU during motor execution (ME) and motor imagery (MI).

## Setup

```bash
conda create -n motion-trajectory-prediction python=3.12 -y
conda activate motion-trajectory-prediction
pip install -r requirements.txt
```

`requirements.txt` also installs this repo's `mtp` package in editable mode (`-e .`).

## Pipeline

Data is never pushed to this repo (`/data/` is gitignored). The notebooks (steps 1-3) read and write absolute paths under `D:\MTP_task`. Step 4 reads the step 3 outputs from `data/processed/` by default (`EEG_data/`, `Kinematics_data/label/`, `Kinematics_data/trajectory/`), or from any folder given with `--data-dir`.

| Step | Code | Reads | Writes |
|---|---|---|---|
| 1 | `notebooks/step1_ICA_Epoch.ipynb` | `D:\MTP_task\MTP_eeg\<subject>\*.bdf` | `D:\MTP_task\MTP_eeg\cleaned_data\ICA_cleaned_Epoch_<subject>.fif` |
| 2 | `notebooks/step2_LabelNoiseRemoval.ipynb` | `D:\MTP_task\cam_data\<subject>\*.xlsx` | `D:\MTP_task\cam_data\trajectory_label\Trajectory_<subject>_<ME\|MI>.pkl`, `Reject_Labels_<subject>_ME.pkl` |
| 3 | `notebooks/step3_get_epoch_label.ipynb` | outputs of steps 1 and 2 | `D:\MTP_task\model\EEG_data_*.npy`, `trajectory_*.npy`, `labels_clean_*.npy` |
| 4 | `python -m mtp.modeling.train` | outputs of step 3 | `reports/baseline_results/baseline_results_<subject>_multihead.xlsx` |

Run step 4 from the repo root. Every setting in `mtp/config.py` (`Config`) is also a command-line option; see `--help`:

```bash
python -m mtp.modeling.train                                   # subject s026, baselines 0 1 2 3 6 7
python -m mtp.modeling.train --subject s008 s009 --baselines 3 6 7
python -m mtp.modeling.train --subject all --skip-existing     # every subject in data/processed; resume a batch
python -m mtp.modeling.train --data-dir D:/MTP_task/model      # lab PC: all files in one folder
python -m mtp.modeling.train --pretrain-epochs 5 --finetune-epochs 5 --device cpu
```

Each subject also gets a `.log` file next to its Excel file. With several subjects, a subject that fails is logged and the run continues; the exit code is 1 if any subject failed. Invalid settings (for example `--device gpu`) are rejected before any data is loaded.

## Tests

```bash
python -m pytest -m "not slow"   # unit tests, a few seconds
python -m pytest                 # everything, about 2 minutes on CPU
```

- `tests/test_config.py`, `test_data.py`, `test_modeling.py`: unit tests for settings, data loading and splitting, metrics, Baseline 0 features and model shapes.
- `tests/test_cli.py` (slow): runs the command line on mock data with less common options, a failing subject and `--skip-existing`.
- `tests/test_golden.py` (slow): runs every baseline for 3 epochs on small random data (CPU, fixed seed) and checks that the result tables match `tests/golden/`.

If the golden test fails, the pipeline's results changed. If you meant to change them, regenerate the expected results with `python tests/make_golden.py` and say why in the commit message.

## Project Organization

```
├── LICENSE
├── Makefile               <- `make requirements`, `make test`, `make test-fast`, `make lint` (needs GNU make)
├── README.md
├── pyproject.toml         <- Package metadata for `mtp`, ruff and pytest settings
├── requirements.txt       <- Dependencies; libraries that affect results are pinned
│
├── data                   <- Local data, gitignored
│   ├── external
│   ├── interim
│   ├── processed          <- Step 3 outputs: EEG_data/, Kinematics_data/{label,trajectory}/
│   └── raw
│
├── docs
├── models                 <- Trained and serialized models
├── notebooks              <- Preprocessing notebooks, steps 1-3
├── references             <- Protocols, electrode montage, manuals
├── reports                <- Generated results
│   ├── baseline_results   <- Step 4 output: one Excel file and one log per subject
│   └── figures
│
├── experimental_program   <- Data-collection code for the lab PC (not part of `mtp`)
│   ├── arduino_imu        <- MPU6050 IMU firmware
│   ├── experiment_task    <- pygame + Kinect experiment task
│   └── requirements.txt
│
├── tests                  <- Unit tests and end-to-end tests (see "Tests")
│   ├── golden             <- Expected result tables for test_golden.py
│   ├── harness.py         <- Mock data, running the CLI, comparing results
│   ├── make_golden.py     <- Regenerates tests/golden/
│   └── test_*.py
│
└── mtp                    <- Source code for this project
    ├── config.py          <- Recording constants and `Config` (all run settings, also CLI options)
    ├── dataset.py         <- Find and load step 3 outputs, build targets, split, scale, loaders
    ├── features.py        <- EEG filter bank, crop and z-score; IMU normalization
    └── modeling
        ├── train.py       <- Step 4 entry point (command line)
        ├── baselines.py   <- Baselines 0-7, one function each
        ├── experiment.py  <- Shared per-subject setup: data, loaders, fit/evaluate/record
        ├── objectives.py  <- Per-batch losses: supervised, teacher-student, prototype-aligned
        ├── trainers.py    <- The training loop shared by all deep baselines
        ├── models.py      <- Encoders, fusion blocks and models
        ├── metrics.py     <- Losses and evaluation metrics
        ├── linear.py      <- Baseline 0: time-lagged ridge regression
        └── results.py     <- Result rows and Excel export
```

--------
