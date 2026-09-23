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

    # Overall test metrics
    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
    print("\nTEST accuracy:", float(test_acc))
    print("TEST loss:", float(test_loss))

    # Predictions
    preds = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    y_pred = np.argmax(preds, axis=1)

    # Report
    report = classification_report(y_test, y_pred, target_names=labels, digits=4)
    print("\n=== Classification Report (TEST) ===\n")
    print(report)

    with open(os.path.join(OUT_DIR, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test accuracy: {test_acc:.6f}\n")
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