import os


# ============================================================
# 0) USER SETTINGS
# ============================================================

subject = "s026"

# Set False if you want to quickly test only one or two baselines.
RUN_BASELINE_0 = True
RUN_BASELINE_1 = True
RUN_BASELINE_2 = True
RUN_BASELINE_3 = True
RUN_BASELINE_4 = False
RUN_BASELINE_5 = False
RUN_BASELINE_6 = True
RUN_BASELINE_7 = True

DATA_DIR = r"D:\MTP_task\model"

RESULTS_DIR = os.path.join(DATA_DIR, "baseline_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RESULT_XLSX_PATH = os.path.join(RESULTS_DIR, f"baseline_results_{subject}_multihead.xlsx")

ME_EEG_PATH = os.path.join(DATA_DIR, f"EEG_data_{subject}_ME.npy")
ME_TRAJ_PATH = os.path.join(DATA_DIR, f"trajectory_{subject}_ME.npy")
ME_LABELS_PATH = os.path.join(DATA_DIR, f"labels_clean_{subject}_ME.npy")

MI_EEG_PATH = os.path.join(DATA_DIR, f"EEG_data_{subject}_MI.npy")
MI_TRAJ_PATH = os.path.join(DATA_DIR, f"trajectory_{subject}_MI.npy")
MI_LABELS_PATH = os.path.join(DATA_DIR, f"labels_clean_{subject}_MI.npy")


EEG_FS = 512.0
TRAJ_FS = 30.0
EPOCH_SEC = 5.0
TARGET_LEN = int(TRAJ_FS * EPOCH_SEC)  # 150

# EEG input epoch timing.
# Input EEG epoch = -2..7 s, model uses 0..5 s after filter-bank filtering.
EEG_EPOCH_TMIN = -2.0
EEG_CROP_TMIN = 0.0
EEG_CROP_TMAX = 5.0
MODEL_EEG_SEC = EEG_CROP_TMAX - EEG_CROP_TMIN
MODEL_EEG_SAMPLES = int(round(EEG_FS * MODEL_EEG_SEC))  # 2560

# Main output target:
# "wrist"    -> wrist Vx,Vy,Vz
# "elbow"    -> elbow Vx,Vy,Vz
# "shoulder" -> shoulder Vx,Vy,Vz
# "all_xyz"  -> shoulder+elbow+wrist, output_dim=9
# "all_kin"  -> shoulder+elbow+wrist+IMU, output_dim=15
TARGET_STREAM = "all_xyz"

TRAJ_KEY_ALIAS = {
    "wrist": "wirst",
    "wirst": "wirst",
    "elbow": "elbow",
    "shoulder": "shoulder",
    "imu": "IMU",
    "IMU": "IMU",
}

TARGET_ORDER = ["Number 1", "Number 2", "Number 3", "Number 4"]

LABEL_TO_KEY = {
    101: "Number 1",
    102: "Number 2",
    103: "Number 3",
    104: "Number 4",
}
KEY_TO_TARGET_ID = {key: i for i, key in enumerate(TARGET_ORDER)}

USE_FILTERBANK = True
BANDS = {
    "slow": (0.5, 1.0),
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
    # If your final low-pass is exactly 40 Hz, you may change this to (30.0, 40.0).
    "low_gamma": (30.0, 45.0),
}

# Deep model settings
LATENT_DIM = 64
LAG_RADIUS = 3
CROSS_ATTENTION_HEADS = 4  # Used by Baseline 6 bidirectional cross-modal MHA fusion.
NORMAL_MHA_HEADS = 4        # Used by Baseline 7 normal MHA fusion.

BATCH_SIZE = 8
PRETRAIN_EPOCHS = 80
FINETUNE_EPOCHS = 60

PATIENCE_PRETRAIN = 12
PATIENCE_FINETUNE = 12

PRETRAIN_LR = 1e-3
FINETUNE_LR = 2e-4
WEIGHT_DECAY = 1e-4
SEED = 42

# Loss weights
TEACHER_LOSS_WEIGHT = 1.0
STUDENT_SUP_LOSS_WEIGHT = 1.0
KD_LOSS_WEIGHT = 0.5
PROTO_LOSS_WEIGHT = 0.2

# Baseline 0 settings: time-lagged MLR
BASELINE0_USE_RIDGE = True
BASELINE0_RIDGE_ALPHA = 10.0

# Time-lagged MLR setting:
# Each velocity sample y(t) is predicted from EEG features around t + lag.
# Negative lags mean EEG before the target velocity sample.
# Example: -0.300 means EEG 300 ms before the current velocity sample.
LINEAR_LAGS_SEC = [-0.300, -0.200, -0.100, 0.000]
LINEAR_LAG_WINDOW_SEC = 0.050  # average EEG within ±25 ms around each lag center
