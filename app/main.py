import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow.keras.models import load_model

from ml.preprocess import prepare_image

app = Flask(__name__)

MODEL_PATH = ROOT / "models" / "mnist_cnn.keras"
model = load_model(str(MODEL_PATH))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    tensor = prepare_image(image)
    probabilities = model.predict(tensor, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    return jsonify({"digit": digit, "confidence": round(confidence, 4)})


if __name__ == "__main__":
    app.run(debug=True)