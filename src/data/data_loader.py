import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split

from src.config import CLASS_NAMES, RANDOM_SEED, VALIDATION_SPLIT


def load_cifar10_data():
    """
    Carga el dataset CIFAR-10 y lo divide en entrenamiento, validación y prueba.

    Retorna:
        x_train, y_train: datos de entrenamiento
        x_val, y_val: datos de validación
        x_test, y_test: datos de prueba
        class_names: nombres de las clases
    """

    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_train_full
    )

    return x_train, y_train, x_val, y_val, x_test, y_test, CLASS_NAMES


def get_dataset_shapes():
    """
    Devuelve las dimensiones principales del dataset CIFAR-10.
    """

    x_train, y_train, x_val, y_val, x_test, y_test, class_names = load_cifar10_data()

    shapes = {
        "x_train": x_train.shape,
        "y_train": y_train.shape,
        "x_val": x_val.shape,
        "y_val": y_val.shape,
        "x_test": x_test.shape,
        "y_test": y_test.shape,
        "num_classes": len(class_names),
    }

    return shapes