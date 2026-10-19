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

Data is never pushed to this repo (`/data/` is gitignored). The code currently reads and writes absolute paths under `D:\MTP_task`. A local copy of the step 3 outputs can live in `data/processed/`, but the code doesn't read from there yet.

| Step | Code | Reads (under `D:\MTP_task`) | Writes (under `D:\MTP_task`) |
|---|---|---|---|
| 1 | `notebooks/step1_ICA_Epoch.ipynb` | `MTP_eeg/<subject>/*.bdf` | `MTP_eeg/cleaned_data/ICA_cleaned_Epoch_<subject>.fif` |
| 2 | `notebooks/step2_LabelNoiseRemoval.ipynb` | `cam_data/<subject>/*.xlsx` | `cam_data/trajectory_label/Trajectory_<subject>_<ME\|MI>.pkl`, `Reject_Labels_<subject>_ME.pkl` |
| 3 | `notebooks/step3_get_epoch_label.ipynb` | outputs of steps 1 and 2 | `model/EEG_data_*.npy`, `model/trajectory_*.npy`, `model/labels_clean_*.npy` |
| 4 | `mtp/modeling/train.py` | outputs of step 3 | `model/baseline_results/baseline_results_<subject>_multihead.xlsx` |

Run step 4 from the repo root (set `subject`, paths and the `RUN_BASELINE_*` flags in `mtp/config.py` first):

```bash
python -m mtp.modeling.train
```

## Tests

`tests/test_golden.py` runs every baseline for 3 epochs on small random data (CPU, fixed seed) and checks that the result tables match `tests/golden/`. Those expected results came from the original single-file script. The test takes about a minute:

```bash
python -m pytest
```

If it fails, the pipeline's results changed. If you meant to change them, regenerate the expected results with `python tests/make_golden.py` and say why in the commit message.

The test runs on CPU because GPU training is not bit-for-bit reproducible.

## Project Organization

```
├── LICENSE
├── Makefile               <- `make requirements`, `make test`, `make lint` (needs GNU make)
├── README.md
├── pyproject.toml         <- Package metadata for `mtp`, ruff and pytest settings
├── requirements.txt       <- Dependencies for preprocessing, training and tests
│
├── data                   <- Local data, gitignored (the code still uses D:\MTP_task)
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
│   └── figures
│
├── experimental_program   <- Data-collection code for the lab PC (not part of `mtp`)
│   ├── arduino_imu        <- MPU6050 IMU firmware
│   ├── experiment_task    <- pygame + Kinect experiment task
│   └── requirements.txt
│
├── tests                  <- Golden-output regression test (see "Tests")
│   ├── golden             <- Expected result tables from the original script
│   ├── harness.py         <- Mock data, settings patching, running and comparing
│   ├── make_golden.py     <- Regenerates tests/golden/
│   └── test_golden.py
│
└── mtp                    <- Source code for this project
    ├── config.py          <- User settings: subject, paths, baselines to run, hyperparameters
    ├── dataset.py         <- Load step 3 outputs, build targets, split, scale, PyTorch datasets
    ├── features.py        <- EEG filter bank, crop and z-score; IMU normalization
    └── modeling
        ├── models.py      <- Encoders, fusion blocks and baseline models
        ├── metrics.py     <- Losses and evaluation metrics
        ├── trainers.py    <- Training and evaluation loops for each model type
        ├── linear.py      <- Baseline 0: time-lagged ridge regression
        ├── results.py     <- Summary rows and Excel export
        └── train.py       <- Step 4 entry point: runs baselines 0-7
```

--------
