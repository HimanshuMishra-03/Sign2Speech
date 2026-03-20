#!/usr/bin/env python3
"""
ASL Real-time Inference Pipeline — Raspberry Pi 4B (Headless + TTS)
=====================================================================
MediaPipe runs in main process.
TFLite runs in isolated subprocess (inference_worker.py) to avoid XNNPACK conflict.
Output: spoken prediction via espeak → Pi 3.5mm audio jack.
Press Ctrl+C to stop.
"""

import os
import cv2
import sys
import json
import time
import subprocess
import threading
import collections
import numpy as np
import mediapipe as mp

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH     = "transformer_int8_static.tflite"
WORKER_SCRIPT  = "inference_worker.py"
LABELS_PATH    = "index_to_sign.json"

CAMERA_INDEX   = 0
CAP_WIDTH      = 640
CAP_HEIGHT     = 480
CAP_FPS        = 30

SEQUENCE_LEN   = 32
NUM_LANDMARKS  = 75
COORDS         = 3
TOP_K          = 3
CONF_THRESHOLD = 0.40
FLIP_FRAME     = True
INFER_EVERY_N  = 2

TTS_COOLDOWN   = 2.0
TTS_DEVICE     = "hw:0,0"
TTS_VOLUME     = 200

TEMP_WARN      = 75.0
TEMP_THROTTLE  = 80.0

LH_RANGE           = (0,  21)
POSE_RANGE         = (21, 54)
RH_RANGE           = (54, 75)
ARR_LEFT_SHOULDER  = 32
ARR_RIGHT_SHOULDER = 33


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def load_labels(path: str) -> list:
    with open(path, "r") as f:
        d = json.load(f)
    return [d[str(i)] for i in range(len(d))]


def read_cpu_temp() -> float:
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True)
        return float(out.strip().split("=")[1].replace("'C", ""))
    except Exception:
        return -1.0


def speak(text: str) -> None:
    def _run():
        try:
            p1 = subprocess.Popen(
                ["espeak", "-a", str(TTS_VOLUME), text, "--stdout"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            subprocess.run(
                ["aplay", "-D", TTS_DEVICE],
                stdin=p1.stdout, stderr=subprocess.DEVNULL
            )
            p1.stdout.close()
        except Exception as e:
            print(f"[tts] Error: {e}")
    threading.Thread(target=_run, daemon=True).start()


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def build_landmark_vector(hand_res, pose_res, w, h):
    frame = np.zeros((NUM_LANDMARKS, COORDS), dtype=np.float32)
    if pose_res.pose_landmarks is None:
        return None
    for i, lm in enumerate(pose_res.pose_landmarks.landmark):
        frame[POSE_RANGE[0] + i] = [lm.x * w, lm.y * h, lm.z]
    if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
        for hand_lms, handedness in zip(
            hand_res.multi_hand_landmarks, hand_res.multi_handedness
        ):
            label  = handedness.classification[0].label
            offset = LH_RANGE[0] if label == "Left" else RH_RANGE[0]
            for i, lm in enumerate(hand_lms.landmark):
                frame[offset + i] = [lm.x * w, lm.y * h, lm.z]
    return frame


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    out  = seq.copy()
    ls   = seq[:, ARR_LEFT_SHOULDER,  :]
    rs   = seq[:, ARR_RIGHT_SHOULDER, :]
    mid  = (ls + rs) / 2.0
    dist = np.linalg.norm(rs[:, :2] - ls[:, :2], axis=1, keepdims=True)
    valid = dist[:, 0] > 1e-6
    dist[~valid] = 1.0
    out[valid] = (seq[valid] - mid[valid, np.newaxis, :]) / dist[valid, np.newaxis, :]
    # Restore missing landmarks (all-zero before norm) back to 0
    # After centering, missing landmarks become non-zero — undo that
    missing = (seq == 0).all(axis=-1)  # (T, 75) bool
    out[missing] = 0.0
    return out


# ─────────────────────────────────────────────
# INFERENCE WORKER (subprocess)
# ─────────────────────────────────────────────
class InferenceWorker:
    """Runs TFLite in a subprocess to isolate from MediaPipe's XNNPACK."""

    def __init__(self, model_path: str, worker_script: str):
        self.proc = subprocess.Popen(
            [sys.executable, worker_script, model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for READY signal
        line = self.proc.stdout.readline().decode().strip()
        if line != "READY":
            raise RuntimeError(f"Worker failed to start: {line}")
        print("[worker] Inference subprocess ready.")

    def infer(self, seq: np.ndarray) -> np.ndarray:
        """Send (1,32,75,3) float32 array, get (250,) float32 back."""
        data = seq.astype(np.float32).tobytes()
        size = len(data).to_bytes(4, "little")
        self.proc.stdin.write(size + data)
        self.proc.stdin.flush()

        # Read response
        size_bytes = self.proc.stdout.read(4)
        if len(size_bytes) < 4:
            raise RuntimeError("Worker died")
        out_size = int.from_bytes(size_bytes, "little")
        out_data = self.proc.stdout.read(out_size)
        return np.frombuffer(out_data, dtype=np.float32)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            pass


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("[main] Loading labels...")
    labels = load_labels(LABELS_PATH)
    assert len(labels) == 250, f"Expected 250 labels, got {len(labels)}"
    print(f"[main] {len(labels)} labels loaded.")

    # ── MediaPipe (main process) ──────────────────────────────────────────────
    print("[main] Loading MediaPipe...")
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print("[main] MediaPipe loaded.")

    # ── Camera ────────────────────────────────────────────────────────────────
    print("[main] Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAP_FPS)
    if not cap.isOpened():
        print("[main] ERROR: Cannot open camera.")
        return
    print("[main] Camera opened.")

    # ── TFLite worker subprocess ──────────────────────────────────────────────
    print("[main] Starting inference worker...")
    worker = InferenceWorker(MODEL_PATH, WORKER_SCRIPT)

    # ── State ─────────────────────────────────────────────────────────────────
    frame_buf   = collections.deque(maxlen=SEQUENCE_LEN)
    last_spoken = 0.0
    last_label  = ""
    frame_count = 0
    temp_tick   = 0

    print(f"\n[main] Running — Conf≥{CONF_THRESHOLD*100:.0f}% | "
          f"Cooldown {TTS_COOLDOWN}s | Press Ctrl+C to stop\n")

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            if FLIP_FRAME:
                bgr = cv2.flip(bgr, 1)

            frame_count += 1
            h, w = bgr.shape[:2]

            # ── MediaPipe every N frames ──────────────────────────────────────
            if frame_count % INFER_EVERY_N == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                hand_res = hands.process(rgb)
                pose_res = pose.process(rgb)
                rgb.flags.writeable = True

                lm = build_landmark_vector(hand_res, pose_res, w, h)
                if lm is not None:
                    frame_buf.append(lm)

            # ── TFLite inference when buffer full ─────────────────────────────
            if len(frame_buf) == SEQUENCE_LEN:
                seq = np.array(frame_buf, dtype=np.float32)
                seq = normalize_sequence(seq)
                inp = seq[np.newaxis]       # (1, 32, 75, 3)

                t0 = time.perf_counter()
                try:
                    raw = worker.infer(inp)
                except RuntimeError as e:
                    print(f"[main] Worker error: {e}")
                    break
                infer_ms = (time.perf_counter() - t0) * 1000

                probs   = softmax(raw.astype(np.float32))
                top_idx = np.argsort(probs)[::-1][:TOP_K]
                top_k   = [(labels[i], float(probs[i])) for i in top_idx]

                best_label, best_conf = top_k[0]
                now = time.time()

                # Print top-3
                print(f"[{infer_ms:.0f}ms] ", end="")
                for label, conf in top_k:
                    print(f"{label}({conf*100:.0f}%)", end="  ")
                print()

                # Speak if confident, new word, cooldown passed
                if (best_conf >= CONF_THRESHOLD and
                        best_label != last_label and
                        now - last_spoken >= TTS_COOLDOWN):
                    print(f"  >>> SPEAKING: {best_label}")
                    speak(best_label)
                    last_spoken = now
                    last_label  = best_label

            # ── Thermal check every 300 frames ────────────────────────────────
            temp_tick += 1
            if temp_tick % 300 == 0:
                temp = read_cpu_temp()
                if temp >= 0:
                    status = ("OK" if temp < TEMP_WARN
                              else "WARNING" if temp < TEMP_THROTTLE
                              else "CRITICAL")
                    print(f"[temp] {temp:.1f}C {status}")

            time.sleep(0.002)

    except KeyboardInterrupt:
        print("\n[main] Stopping...")
    finally:
        worker.close()
        cap.release()
        hands.close()
        pose.close()
        print("[main] Stopped cleanly.")


if __name__ == "__main__":
    main()
