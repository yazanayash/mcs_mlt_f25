# Model Input/Output Contract

This document describes how the backend should use the trained model. It
is meant to keep the ML side (Student 1) and the backend/API side
(Student 2) in agreement so neither blocks the other.

## Files

- `models/mnist_cnn.keras` — the trained Keras model (committed to the repo).
- `ml/preprocess.py` — the shared `prepare_image()` helper. The backend
  should use this so the input format always matches what the model was
  trained on.

## Model input

The model expects a numpy array with:

- **shape:** `(1, 28, 28, 1)` (batch of one, 28x28 pixels, single channel)
- **dtype:** `float32`
- **pixel range:** `[0, 1]`
- **style:** white digit on a black background (same as MNIST)

`prepare_image()` produces exactly this from any `PIL.Image`. Uploads and
canvas drawings are usually a dark digit on a light background, so keep
`invert=True` (the default) for them.

## Model output

`model.predict(tensor)` returns an array of shape `(1, 10)` containing the
softmax probabilities for digits 0-9.

- **predicted digit:** `int(np.argmax(probabilities))`
- **confidence:** `float(np.max(probabilities))`

## Example (backend usage)

```python
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from ml.preprocess import prepare_image

model = load_model("models/mnist_cnn.keras")

def predict_digit(image: Image.Image):
    tensor = prepare_image(image)                 # (1, 28, 28, 1)
    probabilities = model.predict(tensor)[0]      # shape (10,)
    digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    return digit, confidence
```

Suggested JSON response from the `/predict` endpoint:

```json
{
  "digit": 7,
  "confidence": 0.994
}
```
