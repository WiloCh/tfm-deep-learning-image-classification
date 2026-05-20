import tensorflow as tf

from src.config import INPUT_SHAPE, NUM_CLASSES, DEFAULT_LEARNING_RATE


def build_baseline_cnn() -> tf.keras.Model:
    """
    Construye una CNN base para clasificación de imágenes CIFAR-10.

    Este modelo servirá como línea base experimental del TFM.
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=INPUT_SHAPE),

            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="baseline_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model