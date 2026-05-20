import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """
    Fija semillas para reproducibilidad.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass