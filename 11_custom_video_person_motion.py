import os
import cv2
import numpy as np
import time
from collections import defaultdict, deque
from ultralytics import YOLO
import argparse

# -----------------------------
# ARGUMENTS
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Path to input video")
parser.add_argument("--out", default="outputs/custom_video_motion.mp4", help="Path to save output video")
parser.add_argument("--model", default="models/flir_yolo/best.pt", help="Path to YOLO weights (.pt)")
args = parser.parse_args()

# -----------------------------
# CONFIG
# -----------------------------
# Path to the trained YOLO weights (not included in this repo — see README)
MODEL_PATH = args.model

CONF_THRES = 0.35

# movement rules
SPEED_MOVING_PX = 4.0
LOITER_SECONDS = 8.0
HISTORY = 30

# smoothing
BEHAVIOR_WIN = 5

# -----------------------------
# SETUP
# -----------------------------
os.makedirs("outputs", exist_ok=True)

model = YOLO(MODEL_PATH, task="detect")

centroid_hist = defaultdict(lambda: deque(maxlen=HISTORY))
stationary_start = defaultdict(lambda: None)
behavior_hist = defaultdict(lambda: deque(maxlen=BEHAVIOR_WIN))

def centroid_xy(xyxy):
    x1, y1, x2, y2 = xyxy
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {args.video}")

fps_in = cap.get(cv2.CAP_PROP_FPS)
if fps_in <= 0:
    fps_in = 25

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(args.out, fourcc, fps_in, (w, h))

if not out.isOpened():
    raise RuntimeError("Could not create output video writer.")

prev = time.time()
fps = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # if grayscale, convert for drawing
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # detect + track people
        results = model.track(
            frame,
            persist=True,
            conf=CONF_THRES,
            classes=[0],   # person only
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

                hist = centroid_hist[track_id]
                speed = 0.0

                if len(hist) >= 2:
                    x_prev, y_prev = hist[-2]
                    speed = float(np.hypot(cx - x_prev, cy - y_prev))

                # average speed over recent points
                avg_speed = speed
                if len(hist) >= 3:
                    dists = []
                    pts = list(hist)
                    for i in range(1, len(pts)):
                        x_prev, y_prev = pts[i - 1]
                        x_curr, y_curr = pts[i]
                        dists.append(float(np.hypot(x_curr - x_prev, y_curr - y_prev)))
                    avg_speed = float(np.mean(dists))

                # raw behavior
                if avg_speed >= SPEED_MOVING_PX:
                    raw_behavior = "Moving"
                    stationary_start[track_id] = None
                else:
                    if stationary_start[track_id] is None:
                        stationary_start[track_id] = t_now
                    stationary_time = t_now - stationary_start[track_id]
                    raw_behavior = "Loitering" if stationary_time >= LOITER_SECONDS else "Stationary"

                # smooth behavior
                behavior_hist[track_id].append(raw_behavior)
                behavior = max(set(behavior_hist[track_id]), key=behavior_hist[track_id].count)

                # colors
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
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Custom Video Person Motion Detection", annotated)
        out.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Saved:", args.out)