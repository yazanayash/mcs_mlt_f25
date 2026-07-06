# Backend: Web Application and API

This section describes the backend part of the project: the Flask web
server and the prediction API.

## Role in the project

The backend connects the trained MNIST model to end users. It loads
`models/mnist_cnn.keras`, accepts images from the browser, converts them
to the format the model expects using the shared `ml/preprocess.py`
helper, runs inference, and returns the predicted digit and confidence
score as JSON.


## Technology stack

| Component | Choice | Purpose |
|-----------|--------|---------|
| Web framework | Flask | Lightweight Python server for routes and templates |
| Model runtime | TensorFlow/Keras | Load and run the saved CNN |
| Image handling | Pillow (PIL) | Read uploaded files in the API |

## Project structure

All backend code lives under `app/`:

```
app/
└── main.py   # Flask app, model loading, /predict endpoint
```

The backend does **not** duplicate preprocessing logic. It imports
`prepare_image()` from `ml/preprocess.py` so training and inference always
use the same input format.

## API design

### `GET /`

Serves the main page with the drawing canvas and image upload form.

### `POST /predict`

Accepts a multipart form upload with a single field named `file`
(PNG/JPEG/GIF/BMP/WebP).

**Success response (200):**

```json
{
  "digit": 7,
  "confidence": 0.994
}
```

**Error responses (400):**

- No file in the request
- Empty filename
- Invalid or unreadable image

### Prediction pipeline

1. Receive the uploaded image bytes from Flask.
2. Open the image with Pillow.
3. Call `prepare_image(image)` → tensor of shape `(1, 28, 28, 1)`.
4. Run `model.predict(tensor)`.
5. Take `argmax` for the digit and `max` for the confidence.
6. Return JSON to the client.

`prepare_image()` inverts dark-on-light drawings and uploads to match
MNIST (white digit on black background), resizes to 28×28, and normalizes
pixels to `[0, 1]`.

## Model loading

At startup, the server loads the model once from:

```
models/mnist_cnn.keras
```

The path is resolved relative to the project root so the app works when
started with:

```bash
python app/main.py
```

The project root is added to `sys.path` at the top of `main.py` so the
`ml` package can be imported when running the file directly.

## Error handling

The API validates input before calling the model:

- Missing or empty uploads return a clear JSON error message with HTTP 400.
- Corrupt image files are caught when opening with Pillow.

This avoids unnecessary model calls and gives the frontend readable
feedback.

## How to run

From the project root:

```bash
pip install -r requirements.txt
python app/main.py
```

Open `http://127.0.0.1:5000` in a browser. Draw a digit or upload an
image, then click the predict button for that input method.

## Integration summary

| Contract item | Backend implementation |
|---------------|------------------------|
| Model file | `models/mnist_cnn.keras` loaded in `app/main.py` |
| Preprocessing | `from ml.preprocess import prepare_image` |
| Input shape | `(1, 28, 28, 1)` float32, values in `[0, 1]` |
| Output | JSON with `digit` (int) and `confidence` (float) |

Following `CONTRACT.md` ensures the web app produces the same tensor
format the model was trained on, which is essential for accurate
predictions on user drawings and uploads.
