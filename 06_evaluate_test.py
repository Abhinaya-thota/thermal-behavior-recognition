import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

SEQ_ROOT = "data/sequences"
OUT_DIR = "outputs"

# Use the best checkpoint saved during training
# (change to .keras if you switched formats)
MODEL_PATH = "models/thermal_cnn_lstm.h5"

BATCH_SIZE = 4  # safe on RAM; if memory error set to 2

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    test_dir = os.path.join(SEQ_ROOT, "test")
    X_test = np.load(os.path.join(test_dir, "X.npy"), mmap_mode="r")
    y_test = np.load(os.path.join(test_dir, "y.npy"), mmap_mode="r")
    labels = np.load(os.path.join(test_dir, "labels.npy"), allow_pickle=True)

    print("Test samples:", len(y_test), "Classes:", len(labels))

    model = tf.keras.models.load_model(MODEL_PATH)

    # Predict in batches (works for uint8 or float32 sequences, never loads the full test set)
    probs = []
    for a in range(0, len(y_test), BATCH_SIZE):
        Xb = np.array(X_test[a:a + BATCH_SIZE], dtype=np.float32)
        if X_test.dtype == np.uint8:
            Xb /= 255.0
        probs.append(model.predict_on_batch(Xb))
    preds = np.concatenate(probs, axis=0)
    y_pred = np.argmax(preds, axis=1)
    y_test = np.asarray(y_test)

    eps = 1e-7
    test_loss = float(-np.mean(np.log(preds[np.arange(len(y_test)), y_test] + eps)))
    test_acc = float(np.mean(y_pred == y_test))
    print("\nTEST accuracy (per sequence):", test_acc)
    print("TEST loss:", test_loss)

    # Per-video accuracy: average sequence probabilities within each video.
    # Overlapping sequences from one video are not independent samples, so this
    # is the more honest headline number.
    video_line = ""
    vid_path = os.path.join(test_dir, "video_idx.npy")
    if os.path.exists(vid_path):
        video_idx = np.load(vid_path)
        vids = np.unique(video_idx)
        v_true = np.array([y_test[video_idx == v][0] for v in vids])
        v_pred = np.array([np.argmax(preds[video_idx == v].mean(axis=0)) for v in vids])
        video_acc = float(np.mean(v_true == v_pred))
        video_line = f"Test accuracy (per video, {len(vids)} videos): {video_acc:.6f}\n"
        print("TEST accuracy (per video):", video_acc, f"over {len(vids)} videos")
    else:
        print("(video_idx.npy not found -- rebuild sequences with 04_build_sequences.py for per-video accuracy)")

    # Report
    report = classification_report(y_test, y_pred, target_names=labels, digits=4)
    print("\n=== Classification Report (TEST) ===\n")
    print(report)

    with open(os.path.join(OUT_DIR, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test accuracy (per sequence): {test_acc:.6f}\n")
        f.write(video_line)
        f.write(f"Test loss: {test_loss:.6f}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(9, 7))
    plt.imshow(cm)
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_test.png"))
    plt.close()

    print("\nSaved to outputs/:")
    print(" - test_report.txt")
    print(" - confusion_matrix_test.png")

if __name__ == "__main__":
    main()