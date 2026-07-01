"""MNIST data loading and preprocessing.

Keeps all dataset handling in one place so the model, training and
evaluation code can stay focused on their own jobs.
"""

import numpy as np
from tensorflow.keras.datasets import mnist

# MNIST images are 28x28 grayscale, digits 0-9.
IMG_SIZE = 28
NUM_CLASSES = 10


def load_data():
    """Load MNIST and return preprocessed train/test splits.

    Returns:
        (x_train, y_train), (x_test, y_test)
        Images are float32 in [0, 1] with shape (n, 28, 28, 1).
        Labels are the original integer labels (0-9).
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)

    return (x_train, y_train), (x_test, y_test)


def preprocess_images(images):
    """Scale pixels to [0, 1] and add the channel dimension.

    The model is a CNN, so it expects a channel axis at the end:
    (28, 28) -> (28, 28, 1).
    """
    images = images.astype("float32") / 255.0
    images = np.expand_dims(images, axis=-1)
    return images


if __name__ == "__main__":
    # Quick sanity check when running this file directly.
    (x_train, y_train), (x_test, y_test) = load_data()
    print("Train images:", x_train.shape)
    print("Train labels:", y_train.shape)
    print("Test images: ", x_test.shape)
    print("Test labels: ", y_test.shape)
    print("Pixel range: ", x_train.min(), "to", x_train.max())
