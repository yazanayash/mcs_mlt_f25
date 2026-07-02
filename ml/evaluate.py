"""Evaluate the trained model on the MNIST test set.

Loads the saved model, reports overall and per-digit accuracy, and saves
a confusion matrix plus a few misclassified examples as figures for the
report.

    python -m ml.evaluate
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from ml.data import load_data, NUM_CLASSES
from ml.train import MODEL_PATH

# Where to save the figures used in the report.
FIGURES_DIR = os.path.join("report", "figures")


def confusion_matrix(y_true, y_pred):
    """Build a confusion matrix using numpy (rows = true, cols = predicted)."""
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    return matrix


def plot_confusion_matrix(matrix, save_path):
    """Save the confusion matrix as a heatmap image."""
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted digit")
    plt.ylabel("True digit")
    plt.xticks(range(NUM_CLASSES))
    plt.yticks(range(NUM_CLASSES))

    # Write the count in each cell so the numbers are readable.
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            plt.text(j, i, matrix[i, j], ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_misclassified(x_test, y_true, y_pred, save_path, num_examples=10):
    """Save a grid of examples the model got wrong."""
    wrong = np.where(y_true != y_pred)[0][:num_examples]

    plt.figure(figsize=(10, 4))
    for i, index in enumerate(wrong):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[index].reshape(28, 28), cmap="gray")
        plt.title("true %d / pred %d" % (y_true[index], y_pred[index]))
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate():
    """Run the full evaluation and print a short report."""
    (_, _), (x_test, y_test) = load_data()

    model = load_model(MODEL_PATH)

    # Predicted probabilities -> predicted digit.
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    overall_acc = np.mean(y_pred == y_test)
    print("Overall test accuracy: %.4f" % overall_acc)

    print("\nAccuracy per digit:")
    for digit in range(NUM_CLASSES):
        mask = y_test == digit
        digit_acc = np.mean(y_pred[mask] == y_test[mask])
        print("  Digit %d: %.4f" % (digit, digit_acc))

    os.makedirs(FIGURES_DIR, exist_ok=True)

    matrix = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plot_confusion_matrix(matrix, cm_path)
    print("\nSaved confusion matrix to", cm_path)

    mis_path = os.path.join(FIGURES_DIR, "misclassified.png")
    plot_misclassified(x_test, y_test, y_pred, mis_path)
    print("Saved misclassified examples to", mis_path)


if __name__ == "__main__":
    evaluate()
