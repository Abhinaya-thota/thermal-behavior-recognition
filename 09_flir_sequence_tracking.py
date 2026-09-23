import os
import cv2
import numpy as np
import time
import argparse
from collections import defaultdict, deque
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
# Path to the trained YOLO weights (not included in this repo — see README)
MODEL_PATH = "models/flir_yolo/best.pt"

# Folder of thermal frame images to run detection + tracking over
FRAME_DIR = "data/flir_frames/thermal_8_bit"

OUT_PATH = "outputs/flir_sequence_demo.mp4"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=MODEL_PATH, help="Path to YOLO weights (.pt)")
parser.add_argument("--frames", default=FRAME_DIR, help="Folder of thermal frame images")
parser.add_argument("--out", default=OUT_PATH, help="Output video path")
args = parser.parse_args()

MODEL_PATH = args.model
FRAME_DIR = args.frames
OUT_PATH = args.out

CONF_THRES = 0.35
DISPLAY_FPS = 10

# behavior rules
SPEED_MOVING_PX = 4.0
LOITER_SECONDS = 8.0
HISTORY = 30

DISPLACEMENT_FRAMES = 8
MOVING_DISTANCE_PX = 14.0

# -----------------------------
# SETUP
# -----------------------------
os.makedirs("outputs", exist_ok=True)

model = YOLO(MODEL_PATH, task="detect")

centroid_hist = defaultdict(lambda: deque(maxlen=HISTORY))
stationary_start = defaultdict(lambda: None)

# anti-flicker behavior smoothing
behavior_hist = defaultdict(lambda: deque(maxlen=3))
stable_behavior = defaultdict(lambda: "Stationary")

def centroid_xy(xyxy):
    x1, y1, x2, y2 = xyxy
    # use bottom-center of box; more stable for walking people
    return (0.5 * (x1 + x2), y2)

# -----------------------------
# LOAD IMAGE FILES
# -----------------------------
files = sorted([
    f for f in os.listdir(FRAME_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if len(files) == 0:
    raise RuntimeError(f"No image files found in: {FRAME_DIR}")

first_frame = cv2.imread(os.path.join(FRAME_DIR, files[0]))
if first_frame is None:
    raise RuntimeError("Could not read the first frame.")

if len(first_frame.shape) == 2:
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)

h, w = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUT_PATH, fourcc, DISPLAY_FPS, (w, h))

if not out.isOpened():
    print("ERROR: VideoWriter not working")
else:
    print("Saving video to:", OUT_PATH)

prev = time.time()
fps = 0.0

# -----------------------------
# MAIN LOOP
# -----------------------------
try:
    for fname in files:
        frame_path = os.path.join(FRAME_DIR, fname)
        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        results = model.track(
            frame,
            persist=True,
            conf=CONF_THRES,
            classes=[0],   # person class only
            verbose=False
        )

        now = time.time()
        dt = now - prev
        prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)

        annotated = frame.copy()

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int)
            xyxys = boxes.xyxy.cpu().numpy()

            t_now = time.time()

            for track_id, xyxy in zip(ids, xyxys):
                cx, cy = centroid_xy(xyxy)
                centroid_hist[track_id].append((cx, cy))

                speed = 0.0
                hist = centroid_hist[track_id]

                if len(hist) >= 2:
                    x_prev, y_prev = hist[-2]
                    speed = float(np.hypot(cx - x_prev, cy - y_prev))

                # use displacement across a wider window, better for pedestrians
                if len(hist) >= DISPLACEMENT_FRAMES:
                    pts = list(hist)
                    x_old, y_old = pts[-DISPLACEMENT_FRAMES]
                    x_new, y_new = pts[-1]
                    displacement = float(np.hypot(x_new - x_old, y_new - y_old))
                else:
                    displacement = speed
                    # raw behavior
                if displacement >= MOVING_DISTANCE_PX:
                    raw_behavior = "Moving"
                    stationary_start[track_id] = None
                else:
                    if stationary_start[track_id] is None:
                        stationary_start[track_id] = t_now
                        stationary_time = t_now - stationary_start[track_id]
                        raw_behavior = "Loitering" if stationary_time >= LOITER_SECONDS else "Stationary"

                # smooth behavior with recent vote history
                behavior_hist[track_id].append(raw_behavior)
                behavior = max(set(behavior_hist[track_id]), key=behavior_hist[track_id].count)
                stable_behavior[track_id] = behavior

                if behavior == "Moving":
                    color = (0, 255, 0)
                elif behavior == "Stationary":
                    color = (0, 0, 255)
                else:
                    color = (0, 165, 255)

                x1, y1, x2, y2 = xyxy.astype(int)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    annotated,
                    f"ID {track_id} | {behavior}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("FLIR Thermal Detection + Behavior", annotated)
        out.write(annotated)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            print("Stopping...")
            break

except KeyboardInterrupt:
    print("Interrupted! Saving video...")

finally:
    out.release()
    cv2.destroyAllWindows()
    print("Saved:", OUT_PATH)