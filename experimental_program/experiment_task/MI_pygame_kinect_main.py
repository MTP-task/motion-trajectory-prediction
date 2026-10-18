# -*- coding: utf-8 -*-
"""
MTPTask (Pygame) + Kinect (libfreenect) + MediaPipe Pose + IMU (MPU6050 x2)

Changes in this version:
- IMU is read in a dedicated thread (callback-style)
- IMU samples are logged in a separate 100 Hz table
- If IMU disconnects (port closed/unplug/serial error), the program:
  -> breaks loops, closes devices, quits pygame, exits
"""

##### Imports #########
import pygame
import random
import time
import json
import pandas as pd
import serial
import serial.tools.list_ports
import sys
import os
import threading
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List, Tuple

import cv2
import numpy as np
import mediapipe as mp

# =============== Config ===============
with open("config.json") as f:
    config = json.load(f)

SUBJECT_NO = config["subject_no"]
BLOCK_NO = config["block_no"]
NUM_TRIALS = config["num_trials"]
STIMULI = config["stimuli"]
STIMULUS_DURATION = config["stimulus_duration"]
VIDEO_DURATION = config["video_duration"]
REST_DURATION = config["rest_duration"]
CROSS_DURATION = config["cross_duration"]
SET_FULLSCREEN = config["full_screen"]
SERIAL_PORT = config["serial_port"]   # trigger port
IMU_PORT = config["IMU_port"]         # imu port (can be empty)
BAUD_RATE = 115200

trigger_label = {
    "Number 1": 101,
    "Number 2": 102,
    "Number 3": 103,
    "Number 4": 104,
}

# =============== Trigger Serial (optional) ===============
try:
    serTrig = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"[INFO] Trigger serial {SERIAL_PORT} opened.")
except serial.SerialException:
    serTrig = None
    print(f"[WARNING] Trigger serial {SERIAL_PORT} not available.")

def send_trigger(number: int):
    if serTrig:
        try:
            serTrig.write(chr(number).encode())
        except Exception as e:
            print(f"[TRIGGER SERIAL ERROR] {e}")

# =============== IMU Reader (Thread + Callback) ===============

@dataclass
class ImuSample:
    t_perf: float
    t_wall: float
    sample_id: int
    # A accel (g)
    Aax_g: float; Aay_g: float; Aaz_g: float
    # B accel (g)
    Bax_g: float; Bay_g: float; Baz_g: float
    # A gyro (dps)
    Agx_dps: float; Agy_dps: float; Agz_dps: float
    # B gyro (dps)
    Bgx_dps: float; Bgy_dps: float; Bgz_dps: float
    raw_line: str = ""

class IMUReader:
    """
    Reads CSV lines from Arduino at ~100 Hz.
    Keeps a separate log (imu_log) and the latest sample snapshot (get_latest()).
    If the device disconnects, sets disconnected=True and disconnect_reason.
    """

    def __init__(
        self,
        port: Optional[str],
        baud: int = 115200,
        encoding: str = "utf-8",
        on_sample: Optional[Callable[[ImuSample], None]] = None,
        drop_until_calibration_done: bool = True,
    ):
        self.port = port
        self.baud = baud
        self.encoding = encoding
        self.on_sample = on_sample
        self.drop_until_calibration_done = drop_until_calibration_done

        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._latest: Optional[ImuSample] = None
        self._sample_id = 0
        self._calibration_done = not drop_until_calibration_done

        self.imu_log: List[Dict[str, Any]] = []  # separate IMU table

        # disconnect detection
        self.disconnected = False
        self.disconnect_reason = ""
        self._fail_count = 0
        self._max_fail = 30  # consecutive failures -> treat as disconnected

    def is_disconnected(self) -> bool:
        return self.disconnected

    def find_port(self) -> Optional[str]:
        if self.port:
            return self.port
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if any(k in p.description for k in ["Arduino", "CH340", "USB-SERIAL", "CP210", "Silicon Labs"]):
                return p.device
        return ports[0].device if ports else None

    def open(self):
        dev = self.find_port()
        if not dev:
            raise RuntimeError("No IMU serial ports found.")
        print(f"[IMU] Opening {dev} @ {self.baud}")
        self._ser = serial.Serial(dev, self.baud, timeout=1)
        time.sleep(2)  # allow Arduino reset
        self._ser.reset_input_buffer()

    def start(self):
        if self._ser is None:
            self.open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except:
                pass

    def get_latest(self) -> Optional[ImuSample]:
        with self._lock:
            return self._latest

    def _parse_csv_12(self, line: str) -> Optional[Tuple[float, ...]]:
        """
        Expected 12 float CSV:
        Aax_g,Aay_g,Aaz_g,Bax_g,Bay_g,Baz_g,Agx_dps,Agy_dps,Agz_dps,Bgx_dps,Bgy_dps,Bgz_dps
        """
        parts = line.split(",")
        if len(parts) != 12:
            return None
        try:
            vals = tuple(float(x) for x in parts)
            return vals
        except:
            return None

    def _run(self):
        assert self._ser is not None

        while not self._stop.is_set():
            try:
                # If OS closed port / unplugged
                if (self._ser is None) or (not self._ser.is_open):
                    self.disconnected = True
                    self.disconnect_reason = "Serial port closed"
                    break

                b = self._ser.readline()

                # timeout returns b"" -> not a failure
                if not b:
                    continue

                line = b.decode(self.encoding, errors="ignore").strip()
                if not line:
                    continue

                # reset fail counter on any real line
                self._fail_count = 0

                # Calibration gate
                if self.drop_until_calibration_done:
                    if "Finish Calibration" in line:
                        self._calibration_done = True
                        print("[IMU] Calibration done.")
                        continue
                    if not self._calibration_done:
                        continue

                # Ignore header / non-data lines
                if (
                    line.startswith("Aax_g")
                    or line.startswith("Ax,Ay")
                    or "Start Calibration" in line
                ):
                    continue

                vals = self._parse_csv_12(line)
                if vals is None:
                    continue

                self._sample_id += 1
                sample = ImuSample(
                    t_perf=time.perf_counter(),
                    t_wall=time.time(),
                    sample_id=self._sample_id,
                    Aax_g=vals[0], Aay_g=vals[1], Aaz_g=vals[2],
                    Bax_g=vals[3], Bay_g=vals[4], Baz_g=vals[5],
                    Agx_dps=vals[6], Agy_dps=vals[7], Agz_dps=vals[8],
                    Bgx_dps=vals[9], Bgy_dps=vals[10], Bgz_dps=vals[11],
                    raw_line=line
                )

                # Save to IMU table (100 Hz-ish)
                self.imu_log.append({
                    "imu_sample_id": sample.sample_id,
                    "t_perf": sample.t_perf,
                    "t_wall": sample.t_wall,
                    "Aax_g": sample.Aax_g, "Aay_g": sample.Aay_g, "Aaz_g": sample.Aaz_g,
                    "Bax_g": sample.Bax_g, "Bay_g": sample.Bay_g, "Baz_g": sample.Baz_g,
                    "Agx_dps": sample.Agx_dps, "Agy_dps": sample.Agy_dps, "Agz_dps": sample.Agz_dps,
                    "Bgx_dps": sample.Bgx_dps, "Bgy_dps": sample.Bgy_dps, "Bgz_dps": sample.Bgz_dps,
                    "raw": sample.raw_line
                })

                # Update latest
                with self._lock:
                    self._latest = sample

                # Callback
                if self.on_sample:
                    self.on_sample(sample)

            except (serial.SerialException, OSError) as e:
                self._fail_count += 1
                if self._fail_count >= self._max_fail:
                    self.disconnected = True
                    self.disconnect_reason = f"IMU disconnected: {repr(e)}"
                    break
                continue
            except Exception as e:
                self._fail_count += 1
                if self._fail_count >= self._max_fail:
                    self.disconnected = True
                    self.disconnect_reason = f"IMU error: {repr(e)}"
                    break
                continue

        # Ensure port closed on exit
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
        except:
            pass


# =============== Pygame Setup ===============
pygame.init()
pygame.mixer.init()
tick_sound = pygame.mixer.Sound("Tick - Sound Effect (HD).mp3")

if SET_FULLSCREEN:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    WIDTH, HEIGHT = screen.get_size()
else:
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Motor Imagery Experiment")
font = pygame.font.SysFont("Arial", 60)
clock = pygame.time.Clock()

# Load Kinect DLLs if needed (optional; if you already handle via add_dll_directory, keep yours)
import freenect

# MediaPipe / camera source decision
if BLOCK_NO % 2 == 1:
    pose = mp.solutions.pose.Pose(model_complexity=1, enable_segmentation=True, min_detection_confidence=0.7)
    cap = None
else:
    pose = None
    cap = cv2.VideoCapture(0)

# Depth intrinsics (your values)
FXd, FYd, CXd, CYd = 572.882768, 542.739980, 314.649173, 240.160459
FXc, FYc, CXc, CYc = 521.179233, 493.033034, 322.515987, 259.055966

BASELINE_X_M = 0.05
SIGN_X = +1
Y_UP = True

KEEP = {
    12: ("RShoulder", (255, 0, 0)),
    14: ("RElbow",    (0, 255, 255)),
    16: ("RWrist",    (0, 0, 255)),
}

def rgb_to_depth_uv(u_rgb, v_rgb, z_mm, USE_du=True):
    Z = z_mm / 1000.0
    du = SIGN_X * (FXd * BASELINE_X_M / max(Z, 1e-6)) if USE_du else 0.0
    u_depth = u_rgb + (CXd - CXc) + du
    v_depth = v_rgb + (CYd - CYc)
    return int(round(u_depth)), int(round(v_depth))

def sample_depth_mm_big(depth_mm, u, v, k=5):
    h, w = depth_mm.shape
    x0, x1 = max(0, u-k), min(w, u+k+1)
    y0, y1 = max(0, v-k), min(h, v+k+1)
    roi = depth_mm[y0:y1, x0:x1]
    nz = roi[roi > 0]
    if nz.size == 0:
        return 0
    return int(np.median(nz))

def sample_depth_with_baseline(depth_mm, u_rgb, v_rgb, k_seed=20, k_final=20):
    u_d, v_d = rgb_to_depth_uv(u_rgb, v_rgb, 0.0, USE_du=False)
    z_seed = sample_depth_mm_big(depth_mm, u_d, v_d, k=k_seed)
    u_t, v_t = rgb_to_depth_uv(u_rgb, v_rgb, z_seed, USE_du=True)
    z_final = sample_depth_mm_big(depth_mm, u_t, v_t, k=k_final)
    return z_final, (u_t, v_t)

def depth_to_xyz_cm(u, v, z_mm):
    Z = z_mm / 10.0  # mm -> cm
    X = (u - CXd) * Z / FXd
    Y = (v - CYd) * Z / FYd
    if Y_UP:
        Y = -Y
    return X, Y, Z

def draw_text(text, color=(255,255,255)):
    screen.fill((0, 0, 0))
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(WIDTH//2, HEIGHT//2))
    screen.blit(text_surface, text_rect)
    pygame.display.flip()

def draw_cross(img, center, size=40, color=(0, 255, 0), thickness=2):
    x, y = center
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)

# =============== Logs ===============
frame_logs: List[Dict[str, Any]] = []  # camera frame logs

# Optional: IMU callback
def imu_callback(sample: ImuSample):
    pass

# =============== Panic exit on IMU disconnect ===============
def panic_exit(imu_reader: Optional[IMUReader], message: str = "IMU disconnected"):
    print("[FATAL]", message)
    try:
        # draw_text(message, color=(255, 0, 0))
        time.sleep(1.0)
    except:
        pass

    try:
        if imu_reader:
            imu_reader.stop()
    except:
        pass

    try:
        if cap:
            cap.release()
    except:
        pass

    try:
        freenect.sync_stop()
    except:
        pass

    pygame.quit()
    sys.exit(1)

def wait(seconds, imu_reader: IMUReader):
    start_time = time.time()
    while time.time() - start_time < seconds:
        if imu_reader and imu_reader.is_disconnected():
            panic_exit(imu_reader, imu_reader.disconnect_reason or "IMU disconnected")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                panic_exit(imu_reader, "User closed window")
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    panic_exit(imu_reader, "User aborted (ESC/Q)")
        clock.tick(60)

# =============== Trial Functions ===============

def run_normal_video_capture(duration=3, stimulus_name="Unknown", trial_index=1, imu_reader: IMUReader=None):
    start_time = time.time()
    sampling_count = 0
    target_sample = int(30 * duration)

    send_trigger(trigger_label.get(stimulus_name, 0))

    while sampling_count < target_sample:
        if imu_reader and imu_reader.is_disconnected():
            panic_exit(imu_reader, imu_reader.disconnect_reason or "IMU disconnected")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                panic_exit(imu_reader, "User closed window")
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    panic_exit(imu_reader, "User aborted (ESC/Q)")

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        resized = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        draw_cross(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), size=8, color=(255,0,0))

        curr_time = time.time() - start_time
        latest = imu_reader.get_latest() if imu_reader else None

        frame_logs.append({
            "time_s": round(curr_time, 3),
            "stimulus": stimulus_name,
            "trial": trial_index,
            "source": "webcam",
            "Should_cx": 0, "Should_cy": 0, "Should_cz": 0,
            "Elbow_cx": 0, "Elbow_cy": 0, "Elbow_cz": 0,
            "Wrist_cx": 0, "Wrist_cy": 0, "Wrist_cz": 0,
            "Elbow_u": 0, "Elbow_v": 0,
            "Wrist_u": 0, "Wrist_v": 0,

            "imu_sample_id": latest.sample_id if latest else None,
            "imu_t_perf": latest.t_perf if latest else None,
            # "Aax_g": latest.Aax_g if latest else None,
            # "Aay_g": latest.Aay_g if latest else None,
            # "Aaz_g": latest.Aaz_g if latest else None,
            # "Bax_g": latest.Bax_g if latest else None,
            # "Bay_g": latest.Bay_g if latest else None,
            # "Baz_g": latest.Baz_g if latest else None,
            # "Agx_dps": latest.Agx_dps if latest else None,
            # "Agy_dps": latest.Agy_dps if latest else None,
            # "Agz_dps": latest.Agz_dps if latest else None,
            # "Bgx_dps": latest.Bgx_dps if latest else None,
            # "Bgy_dps": latest.Bgy_dps if latest else None,
            # "Bgz_dps": latest.Bgz_dps if latest else None,
        })

        rgb_show = cv2.flip(rgb, 1)
        rgb_show = np.rot90(rgb_show)
        frame_surface = pygame.surfarray.make_surface(rgb_show)
        screen.fill((0, 0, 0))
        screen.blit(pygame.transform.scale(frame_surface, screen.get_size()), (0, 0))
        pygame.display.flip()

        sampling_count += 1
        if sampling_count in (5, 40, 75, 110):
            tick_sound.play()

        clock.tick(60)

def run_hand_tracking_kinect(duration=3, stimulus_name="Unknown", trial_index=1, imu_reader: IMUReader=None):
    start_time = time.time()
    sampling_count = 0
    target_sample = int(30 * duration)

    send_trigger(trigger_label.get(stimulus_name, 0))

    while sampling_count < target_sample:
        if imu_reader and imu_reader.is_disconnected():
            panic_exit(imu_reader, imu_reader.disconnect_reason or "IMU disconnected")

        rgb, _ = freenect.sync_get_video(format=freenect.VIDEO_RGB)
        depth, _ = freenect.sync_get_depth(format=freenect.DEPTH_MM)
        if rgb is None or depth is None:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                panic_exit(imu_reader, "User closed window")
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    panic_exit(imu_reader, "User aborted (ESC/Q)")

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        res = pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        curr_time = time.time() - start_time
        Should_cx = Should_cy = Should_cz = 0.0
        Elbow_cx = Elbow_cy = Elbow_cz = 0.0
        Wrist_cx = Wrist_cy = Wrist_cz = 0.0
        Elbow_u = Elbow_v = 0
        Wrist_u = Wrist_v = 0

        draw_cross(bgr, (w//2, h//2), size=8, color=(0,0,255))

        if res.pose_landmarks:
            for idx, lm in enumerate(res.pose_landmarks.landmark):
                if idx not in KEEP:
                    continue
                u, v = int(lm.x * w), int(lm.y * h)
                if not (0 <= u < w and 0 <= v < h):
                    continue

                name, color = KEEP[idx]
                z_mm, _ = sample_depth_with_baseline(depth, u, v, k_seed=25, k_final=25)

                cv2.circle(bgr, (u, v), 5, color, -1)
                X, Y, Z = depth_to_xyz_cm(u, v, z_mm)

                if name == "RShoulder":
                    Should_cx, Should_cy, Should_cz = X, Y, Z
                elif name == "RElbow":
                    Elbow_cx, Elbow_cy, Elbow_cz = X, Y, Z
                    Elbow_u, Elbow_v = u, v
                elif name == "RWrist":
                    Wrist_cx, Wrist_cy, Wrist_cz = X, Y, Z
                    Wrist_u, Wrist_v = u, v

        latest = imu_reader.get_latest() if imu_reader else None

        frame_logs.append({
            "time_s": round(curr_time, 3),
            "stimulus": stimulus_name,
            "trial": trial_index,
            "source": "kinect",
            "Should_cx": -Should_cx,
            "Should_cy": Should_cy,
            "Should_cz": Should_cz,
            "Elbow_cx": -Elbow_cx,
            "Elbow_cy": Elbow_cy,
            "Elbow_cz": Elbow_cz,
            "Wrist_cx": -Wrist_cx,
            "Wrist_cy": Wrist_cy,
            "Wrist_cz": Wrist_cz,
            "Elbow_u": Elbow_u,
            "Elbow_v": Elbow_v,
            "Wrist_u": Wrist_u,
            "Wrist_v": Wrist_v,

            "imu_sample_id": latest.sample_id if latest else None,
            "imu_t_perf": latest.t_perf if latest else None,
            # "Aax_g": latest.Aax_g if latest else None,
            # "Aay_g": latest.Aay_g if latest else None,
            # "Aaz_g": latest.Aaz_g if latest else None,
            # "Bax_g": latest.Bax_g if latest else None,
            # "Bay_g": latest.Bay_g if latest else None,
            # "Baz_g": latest.Baz_g if latest else None,
            # "Agx_dps": latest.Agx_dps if latest else None,
            # "Agy_dps": latest.Agy_dps if latest else None,
            # "Agz_dps": latest.Agz_dps if latest else None,
            # "Bgx_dps": latest.Bgx_dps if latest else None,
            # "Bgy_dps": latest.Bgy_dps if latest else None,
            # "Bgz_dps": latest.Bgz_dps if latest else None,
        })

        rgb_show = np.rot90(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        frame_surface = pygame.surfarray.make_surface(rgb_show)
        screen.fill((0, 0, 0))
        screen.blit(pygame.transform.scale(frame_surface, screen.get_size()), (0, 0))
        pygame.display.flip()

        sampling_count += 1
        if sampling_count in (5, 40, 75, 110):
            tick_sound.play()

        clock.tick(60)

def run_trial(num_trial, stimulus, imu_reader: IMUReader):
    draw_text("+")
    wait(CROSS_DURATION, imu_reader)

    draw_text(stimulus)
    wait(STIMULUS_DURATION, imu_reader)

    if BLOCK_NO % 2 == 1:
        run_hand_tracking_kinect(duration=VIDEO_DURATION, stimulus_name=stimulus, trial_index=num_trial, imu_reader=imu_reader)
    else:
        run_normal_video_capture(duration=VIDEO_DURATION, stimulus_name=stimulus, trial_index=num_trial, imu_reader=imu_reader)

    send_trigger(10)
    draw_text("Rest", color=(100, 100, 100))
    wait(REST_DURATION, imu_reader)

# =============== Main Experiment ===============

def run_experiment():
    imu_reader = IMUReader(
        port=IMU_PORT,
        baud=115200,
        on_sample=imu_callback,
        drop_until_calibration_done=True
    )

    if BLOCK_NO % 2 == 1:
        draw_text("Motor Execution: Press SPACE to start", color=(0, 255, 0))
    else:
        draw_text("Motor Imagery: Press SPACE to start", color=(0, 255, 0))

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                panic_exit(imu_reader, "User closed window")
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
        clock.tick(60)

    imu_reader.start()

    # Wait for first IMU sample (after calibration)
    draw_text("IMU Calibration... keep still", color=(255, 255, 255))
    t0 = time.time()
    while time.time() - t0 < 10.0:
        if imu_reader.is_disconnected():
            panic_exit(imu_reader, imu_reader.disconnect_reason or "IMU disconnected")
        if imu_reader.get_latest() is not None:
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                panic_exit(imu_reader, "User closed window")
        clock.tick(30)

    if imu_reader._calibration_done:
        pass
    else:
        print("status", imu_reader._calibration_done)
        panic_exit(imu_reader, "User closed window")

    # Balanced trial list
    if NUM_TRIALS % len(STIMULI) != 0:
        panic_exit(imu_reader, "NUM_TRIALS must be divisible by number of stimuli classes.")

    repeats = NUM_TRIALS // len(STIMULI)
    trial_list = STIMULI * repeats
    random.shuffle(trial_list)

    for i, stimulus in enumerate(trial_list):
        if imu_reader.is_disconnected():
            panic_exit(imu_reader, imu_reader.disconnect_reason or "IMU disconnected")
        print(f"Trial {i+1}/{NUM_TRIALS}: {stimulus}")
        send_trigger(i+1)
        run_trial(i+1, stimulus, imu_reader)

    # Save logs
    os.makedirs("./data", exist_ok=True)
    filename = f"./data/{SUBJECT_NO}_block_{BLOCK_NO}_MTPTask.xlsx"
    
    imu_reader.stop()

    df_frames = pd.DataFrame(frame_logs)
    df_imu = pd.DataFrame(imu_reader.imu_log)

    try:
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df_frames.to_excel(writer, index=False, sheet_name="frames")
            df_imu.to_excel(writer, index=False, sheet_name="imu_100hz")
        print(f"[INFO] Saved: {filename}")
    except ModuleNotFoundError:
        print("[ERROR] openpyxl not installed. Run: pip install openpyxl")
        raise

    draw_text("Experiment Done!", color=(0, 255, 0))
    wait(3, imu_reader)
    send_trigger(40)


    if BLOCK_NO % 2 == 1:
        try:
            freenect.sync_stop()
        except:
            pass
    else:
        if cap:
            cap.release()

    pygame.quit()

if __name__ == "__main__":
    run_experiment()
