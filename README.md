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

Data is not stored in this repo. The code currently reads and writes absolute paths under `D:\MTP_task`.

| Step | Code | Reads (under `D:\MTP_task`) | Writes (under `D:\MTP_task`) |
|---|---|---|---|
| 1 | `notebooks/step1_ICA_Epoch.ipynb` | `MTP_eeg/<subject>/*.bdf` | `MTP_eeg/cleaned_data/ICA_cleaned_Epoch_<subject>.fif` |
| 2 | `notebooks/step2_LabelNoiseRemoval.ipynb` | `cam_data/<subject>/*.xlsx` | `cam_data/trajectory_label/Trajectory_<subject>_<ME\|MI>.pkl`, `Reject_Labels_<subject>_ME.pkl` |
| 3 | `notebooks/step3_get_epoch_label.ipynb` | outputs of steps 1 and 2 | `model/EEG_data_*.npy`, `model/trajectory_*.npy`, `model/labels_clean_*.npy` |
| 4 | `mtp/modeling/train.py` | outputs of step 3 | `model/baseline_results/baseline_results_<subject>_multihead.xlsx` |

Run step 4 from the repo root (set `subject` and the `RUN_BASELINE_*` flags at the top of the file first):

```bash
python -m mtp.modeling.train
```

## Project Organization

```
├── LICENSE
├── Makefile               <- `make requirements`, `make lint`, `make format` (needs GNU make)
├── README.md
├── pyproject.toml         <- Package metadata for `mtp` and ruff settings
├── requirements.txt       <- Dependencies for preprocessing and training
│
├── data                   <- Local data folders (gitignored; the code still uses D:\MTP_task)
│   ├── external
│   ├── interim
│   ├── processed
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
└── mtp                    <- Source code for this project
    ├── __init__.py
    └── modeling
        ├── __init__.py
        └── train.py       <- Step 4: baselines 0-7, saves results to Excel
```

--------
