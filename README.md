# Motion Trajectory Prediction (MTP)

This repository contains the experimental programs, preprocessing notebooks, and model-training code for an EEG-based **3D movement trajectory prediction** project. It focuses on decoding continuous movement information from **motor execution (ME)** and **motor imagery (MI)**, with inertial measurement unit (IMU) data used in selected multimodal training methods.

## Repository structure

```text
motion-trajectory-prediction/
├── README.md
├── LICENSE
├── .gitignore
└── src/
    ├── experimental_program/
    │   ├── arduino_imu/
    │   │   └── MPU6050_calAngle.ino
    │   └── experiment_task/
    │       └── MI_pygame_kinect_main.py
    ├── preprocess/
    │   ├── step1_ICA_Epoch.ipynb
    │   ├── step2_LabelNoiseRemoval.ipynb
    │   └── step3_get_epoch_label.ipynb
    └── model/
        └── step4_model_training_MHA.py
```
## 0. Collected EEG and label Data (I will upload in NX server later)
Raw data (Subject 8-28, exclude S12,S20,S27):  `Neuromaze\EEG\pipo\MTP_Project\EEG`, `Neuromaze\EEG\pipo\MTP_Project\Camera`

Preprocessed data (using step1-3): `Neuromaze\EEG\pipo\MTP_Project\Preprocessed`

## 1. Experimental program (`src/experimental_program/`)

Detail of experiment design please refer here: [Participants and experimental paradigm](https://docs.google.com/document/d/1y3-T7zNQfSTVXOaqpaMZzKOZvc6PtGJ4gtaEkH0EEes/edit?usp=sharing)

Code for the movement experiment and IMU acquisition.

| File | Description |
| --- | --- |
| `arduino_imu/MPU6050_calAngle.ino` | Arduino firmware for two MPU6050 sensors (I²C addresses `0x68` and `0x69`). Calibrates gyroscope offsets and streams accelerometer and angular-velocity measurements over serial at approximately 100 Hz. |
| `experiment_task/MI_pygame_kinect_main.py` | Pygame-based ME/MI task with visual cues, serial triggers, camera/Kinect tracking, and synchronized IMU logging. Odd-numbered blocks run **ME** trials using Kinect tracking; even-numbered blocks run **MI** trials using webcam capture. |

The experiment script reads settings from a local `config.json` (such as subject, block, stimulus timing, number of trials, and serial ports). **This configuration file is not included in the current repository.** Recorded data are written to `./data/<subject>_block_<block>_MTPTask.xlsx`, with `frames` and `imu_100hz` sheets.


## 2. Preprocessing (`src/preprocess/`)

The preprocessing notebooks are organized into three sequential steps.

| Step | File | Description |
| --- | --- | --- |
| 1 | `step1_ICA_Epoch.ipynb` | EEG preprocessing and epoch preparation, including independent component analysis (ICA) for artifact handling. |
| 2 | `step2_LabelNoiseRemoval.ipynb` | Review/cleanup of noisy movement-trajectory labels and preparation of trial-rejection information for subsequent alignment. |
| 3 | `step3_get_epoch_label.ipynb` | Separates ME and MI EEG epochs, removes auxiliary EEG channels, applies recorded ME trial exclusions, and aligns trajectory samples to EEG event-label order. Exports arrays for model training. |

**Step 3 output files** (for each subject and condition, `ME` or `MI`):

```text
EEG_data_<subject>_<condition>.npy
trajectory_<subject>_<condition>.npy
labels_clean_<subject>_<condition>.npy
```

The step 3 notebook reads ICA-cleaned EEG epochs (`.fif`), trajectory data (`.pkl`), and, for ME, rejected-trial labels (`.pkl`). Its current implementation crops EEG epochs to **−2 to 7 seconds** and organizes trajectory samples into **150 time points per trial (5 seconds at 30 Hz)**. File paths and subject/condition selections are configured inside the notebook.

## 3. Model training (`src/model/`)

| File | Description |
| --- | --- |
| `step4_model_training_MHA.py` | Trains and evaluates baseline methods for EEG-to-trajectory regression, including EEG-only models, ME-to-MI transfer, EEG+IMU fusion, knowledge distillation, and multi-head attention (MHA) fusion. |

The training script includes the following baselines:

| Baseline | Method |
| --- | --- |
| B0 | Time-lagged multiple linear regression (MLR; optional Ridge regularization) |
| B1 | MI-only EEG → CNN-TCN |
| B2 | ME EEG pretraining → MI fine-tuning (EEG only) |
| B3 | ME EEG+IMU fusion pretraining → MI fine-tuning (zero-filled IMU) |
| B4 | ME multimodal teacher → EEG-only student knowledge distillation → MI fine-tuning |
| B5 | Full model: lag-aware fusion, distillation, subject adaptation, and prototype alignment |
| B6 | Bidirectional cross-modal multi-head attention fusion |
| B7 | Standard multi-head self-attention fusion |

The script uses the `.npy` outputs from step 3. It supports **shoulder**, **elbow**, and **wrist** velocity prediction (`Vx`, `Vy`, `Vz`), either separately or together; `all_kin` additionally includes IMU angular-velocity targets. EEG filter-bank features are computed before cropping the model input to **0–5 seconds**.

### Running the model

1. Set `subject`, `DATA_DIR`, `TARGET_STREAM`, and the `RUN_BASELINE_*` switches in `step4_model_training_MHA.py`.
2. Place the matching ME/MI EEG, trajectory, and clean-label `.npy` files in `DATA_DIR`.
3. Run the script:

   ```bash
   python src/model/step4_model_training_MHA.py
   ```

Results are saved under `DATA_DIR/baseline_results/` as `baseline_results_<subject>_multihead.xlsx` with the sheets `summary`, `axis_metrics`, `stages`, and `config`. The script reports trajectory-regression performance using correlation, RMSE, and MAE.

## Overall workflow

```text
ME/MI experiment + EEG, Kinect/camera, and IMU recordings
                          │
                          ▼
Step 1: EEG artifact processing / ICA / epoching
                          │
                          ▼
Step 2: Movement-label cleanup / trial rejection
                          │
                          ▼
Step 3: Match EEG epochs, labels, and trajectory sequences
                          │
                          ▼
Step 4: Train baselines and evaluate 3D trajectory decoding
```

> **Setup note:** The scripts contain local file paths and hardware-dependent settings. Update these for your environment before running. The first two preprocessing notebooks are described at a high level; consult their cells for the complete processing parameters.
