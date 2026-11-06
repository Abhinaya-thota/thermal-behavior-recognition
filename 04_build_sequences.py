import os
import numpy as np
import cv2

# -----------------------------
# CONFIG
# -----------------------------
FRAME_ROOT = "data/frames_thermal"
SEQ_ROOT = "data/sequences"

SEQ_LEN = 16        # frames per clip (good for behavior learning)
STRIDE = 8          # step between sequence starts. 8 = 50% overlap between neighbours.
                    # (4 = 75% overlap: 2x the data, mostly near-duplicates, 2x slower training)
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
def _write_npy_header(f, shape):
    np.lib.format.write_array_header_1_0(f, {
        "descr": np.lib.format.dtype_to_descr(np.dtype(STORE_DTYPE)),
        "fortran_order": False,
        "shape": tuple(int(d) for d in shape),
    })


def verify_sequences(x_path, expected_len, sample=300):
    """Re-open the saved X.npy and check it is the right size and not blank."""
    X = np.load(x_path, mmap_mode="r")
    if len(X) != expected_len:
        raise RuntimeError(f"{x_path}: has {len(X)} sequences, expected {expected_len}")
    idx = np.unique(np.concatenate([
        np.linspace(0, len(X) - 1, min(sample, len(X))).astype(int),   # spread over the whole file
        [len(X) - 1],                                                   # and the very end
    ]))
    stds = np.array([float(np.asarray(X[i], dtype=np.float32).std()) for i in idx])
    blank = float(np.mean(stds < 1.0))
    print(f"  check: {len(idx)} sequences sampled across the file, {blank:.0%} blank, "
          f"last sequence std={stds[-1]:.1f}")
    if blank > 0.05:
        raise RuntimeError(
            f"{x_path}: {blank:.0%} of sampled sequences are blank -- the file was not written correctly. "
            f"Make sure the disk has free space and is a local drive, then run this script again.")


def process_split(split):
    """
    Two passes so peak RAM is one video, not the whole split:
      1) count how many sequences each video yields (file listing only)
      2) write each sequence straight to X.npy with ordinary sequential file
         writes (no memory-mapping, which is unreliable on network drives),
         then re-open the file and verify it.
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
    x_path = os.path.join(save_dir, "X.npy")

    h, w = IMG_SIZE[1], IMG_SIZE[0]
    seq_shape = (SEQ_LEN, h, w, 1)
    seq_bytes = int(np.prod(seq_shape)) * np.dtype(STORE_DTYPE).itemsize
    print(f"  {len(plan)} videos -> {total} sequences, X.npy {(total,) + seq_shape} "
          f"{np.dtype(STORE_DTYPE).name} ({total * seq_bytes / 1e9:.2f} GB)")

    # Pass 2: stream sequences to disk
    y = np.empty(total, dtype=np.int32)
    video_idx = np.empty(total, dtype=np.int32)   # which video each sequence came from

    i = 0
    with open(x_path, "wb") as f:
        _write_npy_header(f, (total,) + seq_shape)
        for v, (video_path, label, n_planned) in enumerate(plan):
            frames = load_frames(video_path)
            # Frames can be dropped if unreadable; recompute and never exceed the plan.
            n = min(num_sequences(len(frames)), n_planned)
            for k in range(n):
                start = k * STRIDE
                f.write(np.ascontiguousarray(frames[start:start + SEQ_LEN], dtype=STORE_DTYPE).tobytes())
                y[i] = label
                video_idx[i] = v
                i += 1
            if (v + 1) % 100 == 0 or v + 1 == len(plan):
                print(f"  [{v + 1}/{len(plan)} videos] {i} sequences written", flush=True)
        f.flush()
        os.fsync(f.fileno())

    if i < total:
        # Some frames were unreadable: rewrite with the correct sequence count in the header.
        print(f"  NOTE: {total - i} planned sequences were dropped (unreadable frames). Rewriting at final size.")
        tmp_path = os.path.join(save_dir, "X.tmp.npy")
        with open(x_path, "rb") as src, open(tmp_path, "wb") as f:
            np.lib.format.read_magic(src)
            np.lib.format.read_array_header_1_0(src)      # skip the old header (it has the planned count)
            _write_npy_header(f, (i,) + seq_shape)
            remaining = i * seq_bytes
            while remaining:
                chunk = src.read(min(remaining, 64 * seq_bytes))
                if not chunk:
                    raise RuntimeError(f"{x_path} ended early while rewriting")
                f.write(chunk)
                remaining -= len(chunk)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, x_path)

    np.save(os.path.join(save_dir, "y.npy"), y[:i])
    np.save(os.path.join(save_dir, "video_idx.npy"), video_idx[:i])
    np.save(os.path.join(save_dir, "videos.npy"), np.array([p for p, _, _ in plan]))
    np.save(os.path.join(save_dir, "labels.npy"), np.array(classes))

    verify_sequences(x_path, i)
    print(f"{split} saved: {(i,) + seq_shape}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)

    print("\nDONE. Sequences ready.")
