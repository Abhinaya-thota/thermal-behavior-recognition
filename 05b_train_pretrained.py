"""
Pretrained-backbone CNN-LSTM with clip-level augmentation.

Same input as 05_train_cnn_lstm.py (16 x 112 x 112 x 1 thermal-like clips from
04_build_sequences.py), but:

  * each frame goes through MobileNetV2 pretrained on ImageNet instead of a
    small CNN trained from scratch -- with only ~1,000 training videos, the
    from-scratch CNN memorises backgrounds instead of learning motion;
  * clips are augmented while training (the SAME random flip / crop / brightness /
    contrast / noise for all 16 frames of a clip, so motion stays consistent);
  * training runs in two phases: (1) backbone frozen, train the LSTM head;
    (2) unfreeze the top of the backbone and fine-tune with a small learning rate;
  * optional class weights so the large JumpRope class does not dominate.

Saves the best model (lowest val_loss over both phases) to
models/thermal_mobilenet_lstm.keras. Evaluate with:
    python 06_evaluate_test.py --model models/thermal_mobilenet_lstm.keras

Run on a GPU (see colab_train.ipynb); on a CPU this is very slow.
"""
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEQ_ROOT = "data/sequences"
MODEL_DIR = "models"
OUT_DIR = "outputs"
MODEL_PATH = os.path.join(MODEL_DIR, "thermal_mobilenet_lstm.keras")

BATCH_SIZE = 16
LOADER_WORKERS = 4

# Phase 1: frozen backbone
HEAD_EPOCHS = 10
HEAD_LR = 1e-3
# Phase 2: fine-tune the top of the backbone
FINETUNE_EPOCHS = 15
FINETUNE_LR = 1e-4
FINETUNE_LAYERS = 40          # how many of MobileNetV2's last layers to unfreeze (BatchNorm stays frozen)

USE_CLASS_WEIGHTS = True
AUGMENT = True
SEED = 42


# -----------------------------
# DATA
# -----------------------------
def augment_clip(clip, rng):
    """clip: (T, H, W) float32 in [0, 1]. One random transform for the whole clip."""
    t, h, w = clip.shape
    if rng.random() < 0.5:                                   # horizontal flip
        clip = clip[:, :, ::-1]
    scale = rng.uniform(0.80, 1.0)                           # random crop, resized back
    ch, cw = int(h * scale), int(w * scale)
    y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    if scale < 0.999:
        crop = clip[:, y0:y0 + ch, x0:x0 + cw]
        clip = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in crop])
    contrast = rng.uniform(0.8, 1.2)                         # contrast / brightness jitter
    brightness = rng.uniform(-0.1, 0.1)
    mean = clip.mean()
    clip = (clip - mean) * contrast + mean + brightness
    clip = clip + rng.normal(0.0, 0.02, size=clip.shape).astype(np.float32)   # sensor-like noise
    return np.clip(clip, 0.0, 1.0).astype(np.float32)


class ClipSequence(keras.utils.PyDataset):
    """Memory-mapped batch loader for (X.npy, y.npy) with optional augmentation."""

    def __init__(self, x_path, y_path, batch_size, shuffle=True, augment=False, workers=1, seed=0):
        super().__init__(workers=workers, use_multiprocessing=False, max_queue_size=workers * 2)
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.indexes = np.arange(len(self.y))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def __getitem__(self, idx):
        batch_idx = np.sort(self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size])
        Xb = np.array(self.X[batch_idx], dtype=np.float32)
        if self.X.dtype == np.uint8:
            Xb /= 255.0
        yb = np.array(self.y[batch_idx], dtype=np.int32)
        if self.augment:
            rng = np.random.default_rng((self.seed, self.epoch, idx))   # thread-safe, reproducible
            Xb[..., 0] = np.stack([augment_clip(c, rng) for c in Xb[..., 0]])
        return Xb, yb

    def on_epoch_end(self):
        self.epoch += 1
        if self.shuffle:
            np.random.default_rng((self.seed, self.epoch)).shuffle(self.indexes)


def check_not_blank(name, path):
    X = np.load(path, mmap_mode="r")
    idx = np.unique(np.linspace(0, len(X) - 1, min(200, len(X))).astype(int))
    blank = float(np.mean([np.asarray(X[i], dtype=np.float32).std() < 0.25 for i in idx]))
    print(f"Data check ({name}): {blank:.0%} of {len(idx)} sampled sequences are blank")
    if blank > 0.05:
        raise SystemExit(f"STOP: {path} looks broken ({blank:.0%} blank). Rebuild it with 04_build_sequences.py.")


# -----------------------------
# MODEL
# -----------------------------
def build_model(seq_len, h, w, num_classes, weights="imagenet"):
    backbone = keras.applications.MobileNetV2(
        input_shape=(h, w, 3), include_top=False, weights=weights, pooling="avg")
    backbone.trainable = False

    inp = keras.Input(shape=(seq_len, h, w, 1))
    x = layers.Rescaling(2.0, offset=-1.0, name="to_minus1_1")(inp)      # [0,1] -> [-1,1], MobileNetV2's range
    x = layers.Concatenate(axis=-1, name="gray_to_rgb")([x, x, x])        # 1 channel -> 3
    x = layers.TimeDistributed(backbone, name="frame_features")(x)         # (T, 1280)
    x = layers.Dropout(0.5)(x)
    x = layers.LSTM(128, dropout=0.2)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="mobilenet_lstm"), backbone


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def unfreeze_top(backbone, n_layers):
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False          # keep ImageNet BN statistics (small batches would wreck them)
    n = sum(1 for l in backbone.layers if l.trainable and l.weights)
    print(f"Fine-tuning {n} backbone layers with weights (last {n_layers} layers, BatchNorm frozen)")


def plot_history(histories, out_path, phase_split):
    acc = sum((h.history.get("accuracy", []) for h in histories), [])
    val = sum((h.history.get("val_accuracy", []) for h in histories), [])
    plt.figure()
    plt.plot(acc, label="train_acc")
    plt.plot(val, label="val_acc")
    if 0 < phase_split < len(acc):
        plt.axvline(phase_split - 0.5, color="gray", linestyle="--", label="start fine-tuning")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    keras.utils.set_random_seed(SEED)

    paths = {s: (os.path.join(SEQ_ROOT, s, "X.npy"), os.path.join(SEQ_ROOT, s, "y.npy")) for s in ("train", "val")}
    labels = np.load(os.path.join(SEQ_ROOT, "train", "labels.npy"), allow_pickle=True)
    if list(labels) != list(np.load(os.path.join(SEQ_ROOT, "val", "labels.npy"), allow_pickle=True)):
        raise ValueError("Train/Val labels do not match.")
    num_classes = len(labels)

    X_mm = np.load(paths["train"][0], mmap_mode="r")
    seq_len, h, w, _ = X_mm.shape[1:]
    print("Train samples:", len(X_mm), "Input:", X_mm.shape[1:], "Classes:", num_classes,
          "| GPUs:", tf.config.list_physical_devices("GPU"))
    for s in ("train", "val"):
        check_not_blank(s, paths[s][0])

    train_gen = ClipSequence(*paths["train"], BATCH_SIZE, shuffle=True, augment=AUGMENT,
                             workers=LOADER_WORKERS, seed=SEED)
    val_gen = ClipSequence(*paths["val"], BATCH_SIZE, shuffle=False, workers=LOADER_WORKERS)

    class_weight = None
    if USE_CLASS_WEIGHTS:
        y = np.load(paths["train"][1])
        counts = np.bincount(y, minlength=num_classes)
        class_weight = {i: float(len(y) / (num_classes * max(c, 1))) for i, c in enumerate(counts)}
        print("Class weights:", {str(labels[i]): round(wt, 2) for i, wt in class_weight.items()})

    model, backbone = build_model(seq_len, h, w, num_classes)
    model.summary()

    # Phase 1: train the head on frozen ImageNet features
    print("\n=== Phase 1: frozen backbone ===")
    compile_model(model, HEAD_LR)
    ckpt1 = keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1)
    h1 = model.fit(train_gen, validation_data=val_gen, epochs=HEAD_EPOCHS, class_weight=class_weight,
                   callbacks=[ckpt1,
                              keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                              keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)])
    best_val_loss = float(np.min(h1.history["val_loss"]))

    # Phase 2: fine-tune the top of the backbone, keep the phase-1 best unless it improves
    print("\n=== Phase 2: fine-tuning ===")
    unfreeze_top(backbone, FINETUNE_LAYERS)
    compile_model(model, FINETUNE_LR)
    ckpt2 = keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1,
                                            initial_value_threshold=best_val_loss)
    h2 = model.fit(train_gen, validation_data=val_gen, epochs=FINETUNE_EPOCHS, class_weight=class_weight,
                   callbacks=[ckpt2,
                              keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
                              keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)])

    plot_history([h1, h2], os.path.join(OUT_DIR, "training_accuracy_pretrained.png"), len(h1.history["accuracy"]))
    best = min(best_val_loss, float(np.min(h2.history["val_loss"])))
    print(f"\nBest val_loss {best:.4f}. Best model saved to: {MODEL_PATH}")
    print("Saved training plot to outputs/training_accuracy_pretrained.png")
    print(f"Evaluate with: python 06_evaluate_test.py --model {MODEL_PATH}")


if __name__ == "__main__":
    main()
