"""Train the CNN on MNIST and save the trained model.

Run this file to train the model from scratch:

    python -m ml.train

The trained model is written to models/mnist_cnn.keras so the backend
can load it later.
"""

import os

from ml.data import load_data
from ml.model import build_model

# Where to save the trained model.
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mnist_cnn.keras")

# Training settings.
EPOCHS = 10
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1


def train():
    """Load the data, train the model and save it to disk."""
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()

    history = model.fit(
        x_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # Check performance on the test set the model never saw during training.
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("\nTest accuracy: %.4f" % test_acc)
    print("Test loss:     %.4f" % test_loss)

    # Save the trained model for the backend to use.
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print("Saved model to", MODEL_PATH)

    return history


if __name__ == "__main__":
    train()
