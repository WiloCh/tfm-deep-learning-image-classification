import tensorflow as tf

from src.config import INPUT_SHAPE, NUM_CLASSES


def build_transfer_model() -> tf.keras.Model:
    """
    Construye un modelo de transferencia de aprendizaje usando MobileNetV2.

    Este modelo utiliza pesos preentrenados en ImageNet y adapta la salida
    para clasificar las 10 clases de CIFAR-10.
    """

    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)

    x = tf.keras.layers.Resizing(96, 96)(inputs)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = False

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="transfer_mobilenetv2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model