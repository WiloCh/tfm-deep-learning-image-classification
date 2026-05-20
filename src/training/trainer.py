import tensorflow as tf

from src.training.callbacks import build_callbacks
from src.config import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS


def train_model(
    model: tf.keras.Model,
    x_train,
    y_train,
    x_val,
    y_val,
    model_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
):
    """
    Entrena un modelo de clasificación de imágenes.

    Args:
        model: modelo Keras compilado.
        x_train, y_train: datos de entrenamiento.
        x_val, y_val: datos de validación.
        model_name: nombre del modelo para guardar resultados.
        batch_size: tamaño de lote.
        epochs: número máximo de épocas.

    Returns:
        history: historial de entrenamiento.
    """

    callbacks = build_callbacks(model_name)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return history