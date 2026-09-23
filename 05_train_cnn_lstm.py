import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv2D, MaxPooling2D,
    BatchNormalization, Flatten, Dropout, LSTM, Dense
)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GlobalAveragePooling2D

SEQ_ROOT = "data/sequences"
MODEL_DIR = "models"
OUT_DIR = "outputs"

EPOCHS = 20
BATCH_SIZE = 8          # start small on RAM
LEARNING_RATE = 1e-3

class NpySequence(tf.keras.utils.Sequence):
    """Memory-mapped batch loader for (X.npy, y.npy)."""
    def __init__(self, x_path, y_path, batch_size, shuffle=True):
        self.X = np.load(x_path, mmap_mode="r")     # does NOT load into RAM
        self.y = np.load(y_path, mmap_mode="r")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.y))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        # Force a real in-RAM copy for the batch only (small)
        Xb = np.array(self.X[batch_idx], dtype=np.float32)
        yb = np.array(self.y[batch_idx], dtype=np.int32)
        return Xb, yb

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def build_model(seq_len, h, w, num_classes):
    inp = Input(shape=(seq_len, h, w, 1))

    x = TimeDistributed(Conv2D(32, (3,3), activation="relu", padding="same"))(inp)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Conv2D(64, (3,3), activation="relu", padding="same"))(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Conv2D(128, (3,3), activation="relu", padding="same"))(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = Dropout(0.3)(x)

    x = LSTM(128)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def plot_history(history, out_path):
    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_dir = os.path.join(SEQ_ROOT, "train")
    val_dir = os.path.join(SEQ_ROOT, "val")

    x_train_path = os.path.join(train_dir, "X.npy")
    y_train_path = os.path.join(train_dir, "y.npy")
    x_val_path = os.path.join(val_dir, "X.npy")
    y_val_path = os.path.join(val_dir, "y.npy")

    labels_train = np.load(os.path.join(train_dir, "labels.npy"), allow_pickle=True)
    labels_val = np.load(os.path.join(val_dir, "labels.npy"), allow_pickle=True)

    if list(labels_train) != list(labels_val):
        raise ValueError("Train/Val labels do not match. Fix class consistency.")

    num_classes = len(labels_train)

    # Peek shape without loading fully
    X_train_mm = np.load(x_train_path, mmap_mode="r")
    seq_len, h, w, c = X_train_mm.shape[1:]
    print("Train samples:", X_train_mm.shape[0], "Input:", (seq_len, h, w, c), "Classes:", num_classes)

    train_gen = NpySequence(x_train_path, y_train_path, batch_size=BATCH_SIZE, shuffle=True)
    val_gen = NpySequence(x_val_path, y_val_path, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(seq_len, h, w, num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "thermal_cnn_lstm.h5"),
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
    ]
    history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
    )

    model.save(os.path.join(MODEL_DIR, "thermal_cnn_lstm_final.h5"))
    plot_history(history, os.path.join(OUT_DIR, "training_accuracy.png"))
    print("Saved model(s) to:", MODEL_DIR)
    print("Saved training plot to outputs/training_accuracy.png")

if __name__ == "__main__":
    main()
