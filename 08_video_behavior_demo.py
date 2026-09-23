import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import argparse

MODEL_PATH = "models/thermal_cnn_lstm.h5"
LABELS_PATH = "data/sequences/train/labels.npy"

SEQ_LEN = 16
IMG_SIZE = (112, 112)

# match training
FRAME_EVERY_N = 2
CROP_RATIO = 0.90

TOPK = 3
UNKNOWN_THRESH = 0.60
SMOOTH_WIN = 8

def center_crop(frame_bgr, crop_ratio=0.9):
    h, w = frame_bgr.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    y1 = max((h - ch) // 2, 0)
    x1 = max((w - cw) // 2, 0)
    return frame_bgr[y1:y1+ch, x1:x1+cw]

def rgb_to_thermal_like(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return gray

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video (.avi/.mp4)")
    parser.add_argument("--out", default="outputs/demo_prediction.mp4", help="Output video path")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (1200, 600))

    buffer = deque(maxlen=SEQ_LEN)
    pred_hist = deque(maxlen=SMOOTH_WIN)

    frame_idx = 0
    last_label = "Collecting..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_EVERY_N != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        cropped = center_crop(frame, CROP_RATIO)
        small = cv2.resize(cropped, IMG_SIZE, interpolation=cv2.INTER_AREA)
        thermal = rgb_to_thermal_like(small)

        x = thermal.astype("float32") / 255.0
        buffer.append(x)

        label_text = f"Collecting {len(buffer)}/{SEQ_LEN}"
        topk_text = ""

        if len(buffer) == SEQ_LEN:
            seq = np.array(buffer, dtype=np.float32)[None, ..., None]
            pred = model.predict(seq, verbose=0)[0]

            top_idx = np.argsort(pred)[::-1][:TOPK]
            top = [(str(labels[i]), float(pred[i])) for i in top_idx]

            pred_hist.append(int(top_idx[0]))
            smooth_idx = max(set(pred_hist), key=pred_hist.count)

            smooth_label = str(labels[smooth_idx])
            smooth_conf = float(pred[smooth_idx])

            if smooth_conf < UNKNOWN_THRESH:
                label_text = "Behavior: Unknown"
            else:
                label_text = f"Behavior: {smooth_label} ({smooth_conf:.2f})"

            topk_text = "  ".join([f"{j+1}) {name} {conf:.2f}" for j, (name, conf) in enumerate(top)])
            last_label = label_text
        else:
            last_label = label_text

        rgb_disp = cv2.resize(frame, (600, 600), interpolation=cv2.INTER_AREA)
        thermal_color = cv2.applyColorMap(thermal, cv2.COLORMAP_JET)
        thermal_disp = cv2.resize(thermal_color, (600, 600), interpolation=cv2.INTER_NEAREST)

        canvas = np.hstack([rgb_disp, thermal_disp])

        cv2.putText(canvas, last_label, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if topk_text:
            cv2.putText(canvas, topk_text, (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow("Demo: RGB | Thermal + Prediction", canvas)
        out.write(canvas)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Saved:", args.out)

if __name__ == "__main__":
    main()