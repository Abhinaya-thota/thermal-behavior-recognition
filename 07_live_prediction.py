import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import os

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/thermal_cnn_lstm.h5"   # or .keras if you saved that way
LABELS_PATH = "data/sequences/train/labels.npy"
SEQ_LEN = 16
IMG_SIZE = (112, 112)

# -----------------------------
# LOAD MODEL + LABELS
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True)

print("Loaded model:", MODEL_PATH)
print("Model input shape:", model.input_shape)
print("Loaded labels:", list(labels))

# -----------------------------
# THERMAL CONVERSION (same style as training)
# -----------------------------
def rgb_to_thermal(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return gray

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing VideoCapture(0) to 1.")

# Frame buffer
buffer = deque(maxlen=SEQ_LEN)

# FPS calculation
prev_time = time.time()
fps = 0.0

# Simple smoothing to avoid flicker
pred_history = deque(maxlen=8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to model size
    frame = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # Convert to thermal-like
    thermal = rgb_to_thermal(frame)
    x = thermal.astype("float32") / 255.0
    buffer.append(x)

    # FPS
    now = time.time()
    dt = now - prev_time
    prev_time = now
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

    # Default text until buffer is full
    label_text = f"Collecting frames: {len(buffer)}/{SEQ_LEN}"
    conf_text = ""

    if len(buffer) == SEQ_LEN:
        seq = np.array(buffer, dtype=np.float32)[None, ..., None]  # (1,16,112,112,1)

        # sanity check for shape mismatch
        # model.input_shape is often (None, 16, 112, 112, 1)
        pred = model.predict(seq, verbose=0)[0]
        idx = int(np.argmax(pred))
        conf = float(np.max(pred))

        pred_history.append(idx)
        # majority vote smoothing
        smooth_idx = max(set(pred_history), key=pred_history.count)

        label_text = f"Behavior: {labels[smooth_idx]}"
        conf_text = f"Confidence: {conf:.2f}"

    # Display thermal image
    display = cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)

    display = cv2.resize(display, (400, 400), interpolation=cv2.INTER_NEAREST)

    # Overlay text (big + visible)
    cv2.putText(display, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if conf_text:
        cv2.putText(display, conf_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(display, f"FPS: {fps:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Thermal Live Behavior Analysis", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()