from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from app.preprocess import preprocess_image

app = Flask(__name__)
model = tf.keras.models.load_model('models/digit_model.keras')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img_array = preprocess_image(file)
    predictions = model.predict(img_array, verbose=0)
    digit = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    return jsonify({'digit': digit, 'confidence': round(confidence, 4)})


if __name__ == '__main__':
    app.run(debug=True)
