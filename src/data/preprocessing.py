import numpy as np
from tensorflow.keras.utils import to_categorical

from src.config import NUM_CLASSES


def normalize_images(x: np.ndarray) -> np.ndarray:
    """
    Normaliza las imágenes al rango [0, 1].
    CIFAR-10 viene con valores de píxeles entre 0 y 255.
    """
    return x.astype("float32") / 255.0


def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Convierte etiquetas enteras a formato one-hot encoding.
    Ejemplo: 3 -> [0, 0, 0, 1, 0, ...]
    """
    return to_categorical(y, NUM_CLASSES)


def preprocess_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
):
    """
    Aplica el preprocesamiento completo:
    - Normalización de imágenes
    - One-hot encoding de etiquetas

    Retorna:
        x_train_p, y_train_p,
        x_val_p, y_val_p,
        x_test_p, y_test_p
    """

    x_train_p = normalize_images(x_train)
    x_val_p = normalize_images(x_val)
    x_test_p = normalize_images(x_test)

    y_train_p = encode_labels(y_train)
    y_val_p = encode_labels(y_val)
    y_test_p = encode_labels(y_test)

    return x_train_p, y_train_p, x_val_p, y_val_p, x_test_p, y_test_p