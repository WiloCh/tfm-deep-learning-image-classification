from pathlib import Path
import tensorflow as tf

from src.config import MODELS_DIR, LOGS_DIR


def build_callbacks(model_name: str):
    """
    Crea callbacks para entrenamiento:
    - Guarda el mejor modelo
    - Detiene entrenamiento si no mejora
    - Reduce learning rate si se estanca
    - Guarda logs en CSV
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{model_name}_best.keras"
    log_path = LOGS_DIR / f"{model_name}_training_log.csv"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(log_path),
            append=False,
        ),
    ]

    return callbacks