# motion-trajectory-prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project tries to predict how a person's arm moves from their brain activity.

In the experiment, participants sat at a table with four numbered ping-pong balls in front of them. When a ball's number appeared on screen, they either reached out and picked it up (motor execution, ME) or kept still and only imagined doing so (motor imagery, MI). Throughout, 64-channel EEG was recorded. During real reaches, a Kinect camera tracked the shoulder, elbow and wrist, and two IMUs on the arm measured its rotation. Each subject did about 240 real and 240 imagined reaches.

The models take five seconds of EEG, starting at movement onset, and predict the velocity of the shoulder, elbow and wrist in x, y and z, 30 times a second. The main question is whether real reaches, and the IMU signals recorded during them, help a model decode imagined ones.

## Method

### Preparing the data

The raw recordings go through four steps. The first three are Jupyter notebooks, run one subject at a time; the fourth is the `mtp` package in this repo.

| Step | Code | What it does |
|---|---|---|
| 1 | `notebooks/step1_ICA_Epoch.ipynb` | Filters the EEG (0.1 to 50 Hz, 50 Hz notch), re-references it to two external electrodes, cuts epochs around movement onset and removes artifacts such as eye movements and muscle activity with ICA |
| 2 | `notebooks/step2_LabelNoiseRemoval.ipynb` | Turns the camera positions into joint velocities, filters them and drops real reaches with jerky movement |
| 3 | `notebooks/step3_get_epoch_label.ipynb` | Lines up the EEG epochs with the velocities trial by trial and saves both, with the target labels |
| 4 | `python -m mtp.modeling.train` | Builds the model input, trains and tests the baselines, and saves the results |

In step 4, the EEG of each trial (−2 to 7 s around movement onset) is split into six frequency bands: 0.5–1, 1–4, 4–8, 8–13, 13–30 and 30–45 Hz. Each band is filtered over the whole epoch before the 0–5 s window is cut out, so the filter's edge effects stay outside the part the model sees. Each band is then normalized per trial and channel.

**A note on the imagined trials.** An imagined reach has no movement to measure, so step 2 gives every imagined trial the same target: the average of that subject's real reaches to the same ball. A model can therefore score well on MI just by recognizing which ball was imagined. Keep this in mind when you compare ME and MI results.

### Models

All deep models share an EEG encoder. It runs temporal convolutions on each frequency band, combines the 64 channels with a spatial convolution, and models the time course with dilated temporal convolutions. Models that also use the IMU have an IMU encoder and a block that fuses the two. During MI there is no IMU signal, so these models get zeros in its place.

Each baseline tests one idea:

| # | What it trains | Runs by default |
|---|---|---|
| 0 | Linear regression from time-lagged EEG | yes |
| 1 | An EEG model on MI only | yes |
| 2 | An EEG model pretrained on ME, then fine-tuned on MI | yes |
| 3 | An EEG and IMU model pretrained on ME, then fine-tuned on MI with the IMU set to zero | yes |
| 4 | A teacher (EEG and IMU) and a student (EEG only) on ME, then the student on MI | no |
| 5 | Baseline 4 with lag-aware fusion, per-subject scaling and alignment to ME prototypes | no |
| 6 | Baseline 3 with cross-attention between EEG and IMU | yes |
| 7 | Baseline 3 with self-attention over the fused features | yes |

`mtp/modeling/baselines.py` describes each one in more detail.

### Training and scoring

Every subject is trained and tested separately. Each subject's trials are split into about 70% for training, 15% for validation and 15% for testing, with the same mix of balls in each part. Training stops when the validation loss stops improving, and the best weights are kept. The loss combines the error in each velocity with how well the shape of the predicted curve follows the real one.

The models are scored on the test trials with the Pearson correlation (r), the RMSE and the MAE between predicted and real velocity, for each joint and axis. The main score of each baseline is on the MI test trials.

## Running it

### Setup

You need Python 3.12. If the machine has an NVIDIA GPU, install the CUDA build of PyTorch first ([pytorch.org](https://pytorch.org/get-started/locally/)); otherwise pip installs the CPU-only build. Then:

```bash
conda create -n motion-trajectory-prediction python=3.12 -y
conda activate motion-trajectory-prediction
pip install -r requirements.txt
```

### Data

The recordings are not in this repository. Put the output of step 3 in `data/processed/`, in the subfolders `EEG_data/`, `Kinematics_data/label/` and `Kinematics_data/trajectory/`. If all the files are in one folder instead, as on the lab PC, pass that folder with `--data-dir`.

### Training

From the repository root:

```bash
python -m mtp.modeling.train                                 # subject s026, the default baselines
python -m mtp.modeling.train --subject s008 s009 --baselines 3 6 7
python -m mtp.modeling.train --subject all --skip-existing   # every subject; skips finished ones
python -m mtp.modeling.train --data-dir D:/MTP_task/model    # all files in one folder
```

Every setting, from the number of epochs to the loss weights, can be changed on the command line; `--help` lists them with their defaults.

On the same machine, a run with the same settings and seed gives exactly the same results every time. Results on a CPU and on a GPU differ slightly from each other.

### Results

Each subject gets `reports/baseline_results/baseline_results_<subject>_multihead.xlsx`, with a log of the run next to it. The Excel file has four sheets:

- `summary`: one row per evaluation, with the mean r, RMSE and MAE
- `axis_metrics`: the same scores for each joint and axis
- `stages`: the training and validation loss of every epoch
- `config`: every setting used for the run

### Tests

```bash
python -m pytest
```

The tests check data loading, the metrics and the models, and run every baseline on small random data to make sure the results stay the same. They take about three minutes on a CPU.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the code style and how to work with the tests.

## Project layout

```
├── LICENSE
├── Makefile               <- Shortcuts: make requirements, test, lint, format (needs GNU make)
├── README.md
├── CONTRIBUTING.md        <- Code style and how to work with the tests
├── pyproject.toml         <- Package metadata and the ruff and pytest settings
├── requirements.txt       <- Dependencies; the libraries that affect results are pinned
│
├── data                   <- Local data, not in git
│   ├── external
│   ├── interim
│   ├── processed          <- Step 3 output: EEG_data/, Kinematics_data/{label,trajectory}/
│   └── raw
│
├── docs
├── models                 <- Trained models
├── notebooks              <- Steps 1 to 3
├── references             <- Protocols, electrode montage, manuals
├── reports                <- Generated results
│   ├── baseline_results   <- Excel results and logs from step 4
│   └── figures
│
├── experimental_program   <- Code for the lab PC that runs the experiment
│   ├── arduino_imu        <- Firmware for the MPU6050 IMUs
│   ├── experiment_task    <- The pygame and Kinect task
│   └── requirements.txt
│
├── tests
│   ├── golden             <- Expected results for the golden test
│   ├── harness.py         <- Mock data, running the command line, comparing results
│   ├── make_golden.py     <- Regenerates tests/golden/
│   └── test_*.py
│
└── mtp                    <- The Python package
    ├── config.py          <- Recording constants and Config, the settings of a run
    ├── dataset.py         <- Loads the step 3 output, builds targets, splits, scales
    ├── features.py        <- EEG filter bank, cropping and normalization
    └── modeling
        ├── train.py       <- Command-line entry point
        ├── baselines.py   <- One function per baseline
        ├── experiment.py  <- Data, loaders and results shared by the baselines of a subject
        ├── objectives.py  <- What each training stage computes per batch
        ├── trainers.py    <- The training loop
        ├── models.py      <- Encoders, fusion blocks and models
        ├── metrics.py     <- Losses and scores
        ├── linear.py      <- Baseline 0
        └── results.py     <- Result tables and the Excel export
```
