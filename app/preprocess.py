import cv2
import numpy as np
from PIL import Image


def preprocess_image(image_bytes, target_size=(28, 28)):
    """Convert uploaded image bytes to MNIST-compatible format."""
    img = Image.open(image_bytes).convert('L')
    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = 255.0 - img_array
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array
