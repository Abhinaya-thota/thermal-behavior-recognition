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

# Sequences are stored as uint8 (0..255) instead of float32 (0..1):
# 4x smaller on disk. The training/eval loaders divide by 255 per batch.
STORE_DTYPE = np.uint8

IMG_EXTS = (".jpg", ".jpeg", ".png")


# -----------------------------
# LOAD FRAMES FROM ONE VIDEO
# -----------------------------
def list_frames(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def num_sequences(n_frames, seq_len=SEQ_LEN, stride=STRIDE):
    if n_frames < seq_len:
        return 0
    return (n_frames - seq_len) // stride + 1


def load_frames(folder):
    """Load one video's frames as a (T, H, W) uint8 array."""
    frames = []
    for f in list_frames(folder):
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"WARNING: unreadable frame skipped: {os.path.join(folder, f)}")
            continue
        if img.shape[::-1] != IMG_SIZE:
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        frames.append(img)
    if not frames:
        return np.empty((0, IMG_SIZE[1], IMG_SIZE[0]), dtype=np.uint8)
    return np.stack(frames)


# -----------------------------
# PROCESS SPLIT (train/val/test)
# -----------------------------
def process_split(split):
    """
    Two passes so peak RAM is one video, not the whole split:
      1) count how many sequences each video yields (file listing only)
      2) open X.npy as a memory-mapped file of exactly that size and
         write each sequence straight to disk.
    """
    print(f"\nProcessing {split}...")

    split_path = os.path.join(FRAME_ROOT, split)
    if not os.path.isdir(split_path):
        print(f"Missing split folder: {split_path}")
        return

    classes = sorted(d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d)))
    label_map = {cls: i for i, cls in enumerate(classes)}

    # Pass 1: plan
    plan = []      # (video_path, label, n_seq)
    total = 0
    for cls in classes:
        class_path = os.path.join(split_path, cls)
        for video_id in sorted(os.listdir(class_path)):
            video_path = os.path.join(class_path, video_id)
            if not os.path.isdir(video_path):
                continue
            n = num_sequences(len(list_frames(video_path)))
            if n:
                plan.append((video_path, label_map[cls], n))
                total += n

    save_dir = os.path.join(SEQ_ROOT, split)
    os.makedirs(save_dir, exist_ok=True)

    h, w = IMG_SIZE[1], IMG_SIZE[0]
    shape = (total, SEQ_LEN, h, w, 1)
    gb = np.prod(shape) * np.dtype(STORE_DTYPE).itemsize / 1e9
    print(f"  {len(plan)} videos -> {total} sequences, X.npy {shape} {np.dtype(STORE_DTYPE).name} ({gb:.2f} GB)")

    # Pass 2: fill memory-mapped output
    X = np.lib.format.open_memmap(os.path.join(save_dir, "X.npy"), mode="w+", dtype=STORE_DTYPE, shape=shape)
    y = np.empty(total, dtype=np.int32)
    video_idx = np.empty(total, dtype=np.int32)   # which video each sequence came from

    i = 0
    for v, (video_path, label, n_planned) in enumerate(plan):
        frames = load_frames(video_path)
        # Frames can be dropped if unreadable; recompute and never exceed the plan.
        n = min(num_sequences(len(frames)), n_planned)
        for k in range(n):
            start = k * STRIDE
            X[i, :, :, :, 0] = frames[start:start + SEQ_LEN]
            y[i] = label
            video_idx[i] = v
            i += 1

    if i < total:
        # Some frames were unreadable: shrink to what was actually written.
        print(f"  NOTE: {total - i} planned sequences were dropped (unreadable frames). Rewriting at final size.")
        X.flush()
        del X
        src = np.load(os.path.join(save_dir, "X.npy"), mmap_mode="r")
        tmp_path = os.path.join(save_dir, "X.tmp.npy")
        dst = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=STORE_DTYPE, shape=(i,) + shape[1:])
        for a in range(0, i, 256):
            dst[a:a + 256] = src[a:min(a + 256, i)]
        dst.flush()
        del dst, src
        os.replace(tmp_path, os.path.join(save_dir, "X.npy"))
    else:
        X.flush()
        del X

    np.save(os.path.join(save_dir, "y.npy"), y[:i])
    np.save(os.path.join(save_dir, "video_idx.npy"), video_idx[:i])
    np.save(os.path.join(save_dir, "videos.npy"), np.array([p for p, _, _ in plan]))
    np.save(os.path.join(save_dir, "labels.npy"), np.array(classes))

    print(f"{split} saved: {(i,) + shape[1:]}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)

    print("\nDONE. Sequences ready.")
