import tensorflow as tf


def build_data_augmentation() -> tf.keras.Sequential:
    """
    Define técnicas de aumento de datos para mejorar la generalización del modelo.

    Se aplican transformaciones suaves para no alterar demasiado las imágenes
    originales de CIFAR-10.
    """

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
        ],
        name="data_augmentation",
    )

    return data_augmentation