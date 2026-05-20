import tensorflow as tf

from src.config import INPUT_SHAPE, NUM_CLASSES, DEFAULT_LEARNING_RATE
from src.data.augmentation import build_data_augmentation


def build_improved_cnn() -> tf.keras.Model:
    """
    Construye una CNN mejorada para CIFAR-10.

    Mejoras aplicadas respecto al modelo base:
    - Data augmentation
    - Batch Normalization
    - Dropout
    - Mayor capacidad de extracción de características
    """

    data_augmentation = build_data_augmentation()

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=INPUT_SHAPE),

            data_augmentation,

            # Bloque convolucional 1
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Bloque convolucional 2
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.30),

            # Bloque convolucional 3
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.40),

            # Clasificación
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.50),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="improved_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model