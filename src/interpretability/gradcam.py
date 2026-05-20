import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import CLASS_NAMES, FIGURES_DIR


def find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError("No se encontró una capa Conv2D en el modelo.")


def apply_layer(layer, x):
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def make_gradcam_heatmap(image, model, last_conv_layer_name: str, pred_index=None):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_index = model.layers.index(last_conv_layer)

    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output,
    )

    classifier_layers = model.layers[last_conv_index + 1:]

    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(image)
        tape.watch(conv_outputs)

        x = conv_outputs
        for layer in classifier_layers:
            x = apply_layer(layer, x)

        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise ValueError("No se pudieron calcular los gradientes para Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    max_value = tf.reduce_max(heatmap)
    if max_value == 0:
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap = tf.maximum(heatmap, 0) / max_value

    return heatmap.numpy(), predictions.numpy()


def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)

    jet = plt.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + image * 255
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

    return superimposed_img


def generate_gradcam_example(
    model,
    image,
    true_label,
    model_name: str,
    image_index: int,
):
    output_dir = FIGURES_DIR / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    last_conv_layer_name = find_last_conv_layer(model)

    input_image = np.expand_dims(image, axis=0)

    heatmap, predictions = make_gradcam_heatmap(
        input_image,
        model,
        last_conv_layer_name,
    )

    pred_label = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    overlay = overlay_heatmap(image, heatmap)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Original\nReal: {CLASS_NAMES[true_label]}")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(
        f"Grad-CAM\nPred: {CLASS_NAMES[pred_label]}\nConf: {confidence:.2f}"
    )

    plt.tight_layout()

    output_path = output_dir / f"{model_name}_gradcam_{image_index}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path