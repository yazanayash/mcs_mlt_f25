"""CNN model definition for MNIST digit recognition.

A small convolutional network is more than enough for MNIST and trains
quickly even on a CPU. The architecture is kept simple on purpose so it
is easy to explain in the report.
"""

from tensorflow.keras import layers, models

# MNIST images are 28x28 grayscale and there are 10 possible digits (0-9).
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10


def build_model():
    """Build and compile the CNN used to classify handwritten digits.

    Architecture:
        Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Flatten
        -> Dense(128) -> Dropout -> Dense(10, softmax)

    Returns a compiled Keras model ready for training.
    """
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),

        # First convolution block: learns basic edges and strokes.
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Second convolution block: learns more complex shapes.
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Classifier head.
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),  # helps reduce overfitting
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # labels are integers 0-9
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # Print the model summary so we can check the layer shapes.
    model = build_model()
    model.summary()
