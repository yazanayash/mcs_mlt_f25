import os
import sys
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import keras
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image

from ml.preprocess import prepare_image

app = Flask(__name__)

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

MODEL_PATH = ROOT / "models" / "mnist_cnn.keras"
model = keras.models.load_model(str(MODEL_PATH))


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
    probabilities = model(tensor, training=False).numpy()[0]
    digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    return jsonify({"digit": digit, "confidence": round(confidence, 4)})


if __name__ == "__main__":
    app.run(debug=True)
