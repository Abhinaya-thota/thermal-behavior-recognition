import os
import numpy as np
import cv2

# -----------------------------
# CONFIG
# -----------------------------
FRAME_ROOT = "data/frames_thermal"
SEQ_ROOT = "data/sequences"

SEQ_LEN = 16        # frames per clip (good for behavior learning)
STRIDE = 4          # overlap between sequences
IMG_SIZE = (112, 112)

# -----------------------------
# LOAD FRAMES FROM ONE VIDEO
# -----------------------------
def load_frames(folder):
    frames = []
    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for f in files:
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype("float32") / 255.0
        frames.append(img)

    return frames

# -----------------------------
# CREATE SEQUENCES
# -----------------------------
def make_sequences(frames, seq_len=16, stride=4):
    sequences = []

    for start in range(0, len(frames) - seq_len + 1, stride):
        clip = frames[start:start+seq_len]
        clip = np.array(clip)[..., np.newaxis]
        sequences.append(clip)

    return sequences

# -----------------------------
# PROCESS SPLIT (train/val/test)
# -----------------------------
def process_split(split):
    print(f"\nProcessing {split}...")

    split_path = os.path.join(FRAME_ROOT, split)
    classes = sorted(os.listdir(split_path))
    label_map = {cls: i for i, cls in enumerate(classes)}

    X = []
    y = []

    for cls in classes:
        class_path = os.path.join(split_path, cls)

        for video_id in os.listdir(class_path):
            video_path = os.path.join(class_path, video_id)

            frames = load_frames(video_path)
            sequences = make_sequences(frames, SEQ_LEN, STRIDE)

            for seq in sequences:
                X.append(seq)
                y.append(label_map[cls])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    save_dir = os.path.join(SEQ_ROOT, split)
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "X.npy"), X)
    np.save(os.path.join(save_dir, "y.npy"), y)
    np.save(os.path.join(save_dir, "labels.npy"), np.array(classes))

    print(f"{split} saved:", X.shape)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)

    print("\nDONE. Sequences ready.")
