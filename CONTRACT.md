# Model Input/Output Contract

This document describes how the backend should use the trained model. It
is meant to keep the ML side (Student 1) and the backend/API side
(Student 2) in agreement so neither blocks the other.

## Files

- `models/mnist_cnn.keras` — the trained Keras model (committed to the repo).
- `ml/preprocess.py` — the shared `prepare_image()` helper. The backend
  should use this so the input format always matches what the model was
  trained on.
- `app/main.py` — the Flask web application implementing the backend server and prediction API.

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

## API Contract (Flask Implementation)

The backend exposes a prediction API on the `/predict` endpoint.

- **URL:** `/predict`
- **Method:** `POST`
- **Consumes:** `multipart/form-data`
- **Request Parameters:**
  - `file`: The image file to classify.

### Success Response (200 OK)

Returns the predicted digit and confidence score.

```json
{
  "digit": 7,
  "confidence": 0.994
}
```

### Error Responses (400 Bad Request)

- If no file is uploaded:
  ```json
  {
    "error": "No file uploaded"
  }
  ```
- If an empty file/filename is selected:
  ```json
  {
    "error": "No file selected"
  }
  ```
- If the file is not a valid image:
  ```json
  {
    "error": "Invalid image file"
  }
  ```
