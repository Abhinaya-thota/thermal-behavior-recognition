import os
import cv2
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
SRC_ROOT = "data/raw_videos"
OUT_ROOT = "data/frames_thermal"

IMG_SIZE = (112, 112)       # final frame size
FRAME_EVERY_N = 2           # sample rate: keep 1 frame every N frames
CROP_RATIO = 0.90           # center crop ratio (0.9 keeps central 90%)

# Skip frames that are almost blank / constant (prevents corrupt or dark frames)
MIN_MEAN_INTENSITY = 5.0
MIN_STD_INTENSITY = 3.0

# Allowed video extensions
VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")


# -----------------------------
# HELPERS
# -----------------------------
def center_crop(frame_bgr: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
    """Center crop the image by crop_ratio (e.g., 0.9 keeps 90% of width/height)."""
    h, w = frame_bgr.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    y1 = max((h - ch) // 2, 0)
    x1 = max((w - cw) // 2, 0)
    return frame_bgr[y1:y1 + ch, x1:x1 + cw]


def rgb_to_thermal_like(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Convert an RGB/BGR frame into a thermal-like grayscale image.
    This is a synthetic approximation:
      - grayscale conversion
      - CLAHE contrast enhancement
      - Gaussian blur for diffusion look
      - normalization to 0..255
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement (helps separate "hot vs cold" regions)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Blur mimics thermal diffusion / sensor softness
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Normalize intensities
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return gray


def process_video(video_path: str, out_dir: str) -> int:
    """Extract and save thermal-like frames from a video. Returns #frames saved."""
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"WARNING: Cannot open video: {video_path}")
        return 0

    idx, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % FRAME_EVERY_N == 0:
            # Preprocessing: crop + resize
            frame = center_crop(frame, crop_ratio=CROP_RATIO)
            frame = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_AREA)

            thermal = rgb_to_thermal_like(frame)

            # Quality filter (skip blank / constant frames)
            m = float(np.mean(thermal))
            s = float(np.std(thermal))
            if m < MIN_MEAN_INTENSITY or s < MIN_STD_INTENSITY:
                idx += 1
                continue

            out_path = os.path.join(out_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(out_path, thermal)
            saved += 1

        idx += 1

    cap.release()
    return saved


# -----------------------------
# MAIN
# -----------------------------
def main():
    if not os.path.isdir(SRC_ROOT):
        print(f"ERROR: source folder not found: {os.path.abspath(SRC_ROOT)}")
        return

    # Collect the work first so progress can show "n / total".
    jobs = []
    for split in ["train", "val", "test"]:
        split_in = os.path.join(SRC_ROOT, split)
        if not os.path.isdir(split_in):
            print(f"Missing split folder: {split_in}")
            continue

        classes = sorted([d for d in os.listdir(split_in) if os.path.isdir(os.path.join(split_in, d))])
        for cls in classes:
            cls_in = os.path.join(split_in, cls)

            videos = sorted([v for v in os.listdir(cls_in) if v.lower().endswith(VIDEO_EXTS)])
            if len(videos) == 0:
                print(f"[{split}] {cls}: No videos found.")
                continue

            for vid in videos:
                jobs.append((split, cls, vid))

    print(f"Found {len(jobs)} videos under {os.path.abspath(SRC_ROOT)}")
    if not jobs:
        print("Nothing to do. Expected data/raw_videos/<train|val|test>/<Class>/<video>.avi")
        return
    print(f"Writing frames to {os.path.abspath(OUT_ROOT)}\n")

    total_frames, empty = 0, []
    for i, (split, cls, vid) in enumerate(jobs, 1):
        video_path = os.path.join(SRC_ROOT, split, cls, vid)
        video_id = os.path.splitext(vid)[0]
        out_dir = os.path.join(OUT_ROOT, split, cls, video_id)

        n = process_video(video_path, out_dir)
        total_frames += n
        if n == 0:
            empty.append(video_path)
        print(f"[{i:4d}/{len(jobs)}] {split:5s} {cls} / {vid} -> {n} frames", flush=True)

    print(f"\nDONE. {total_frames} thermal-like frames from {len(jobs)} videos written to: {OUT_ROOT}")
    if empty:
        print(f"WARNING: {len(empty)} videos produced 0 frames (could not be read, or all frames too dark), e.g.:")
        for p in empty[:10]:
            print("   ", p)


if __name__ == "__main__":
    main()
