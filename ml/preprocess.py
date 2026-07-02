"""Image preprocessing for digit recognition.

Turns an arbitrary input image (an uploaded file or a drawing from the
frontend canvas) into the exact array the trained model expects:

    shape (1, 28, 28, 1), float32, pixel values in [0, 1],
    white digit on a black background (same style as MNIST).

The backend imports ``prepare_image`` so the frontend, the backend and
the model always agree on the input format.
"""

import numpy as np
from PIL import Image, ImageOps


def prepare_image(image, invert=True):
    """Convert a PIL image into the model's input tensor.

    Args:
        image: a ``PIL.Image`` of any size or mode.
        invert: set ``True`` when the digit is dark on a light background
            (the usual case for uploads and canvas drawings). MNIST digits
            are white on black, so inverting makes the input match what the
            model was trained on.

    Returns:
        A numpy array of shape (1, 28, 28, 1), dtype float32, values in [0, 1].
    """
    # Work in grayscale.
    image = image.convert("L")

    # Make the digit white on a black background to match MNIST.
    if invert:
        image = ImageOps.invert(image)

    # Resize to the 28x28 the model was trained on.
    image = image.resize((28, 28))

    # Scale to [0, 1] and add the batch and channel dimensions.
    array = np.asarray(image, dtype="float32") / 255.0
    array = array.reshape(1, 28, 28, 1)
    return array
