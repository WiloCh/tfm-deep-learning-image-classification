import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.config import CLASS_NAMES, FIGURES_DIR, MODELS_DIR
from src.data.data_loader import load_cifar10_data
from src.data.preprocessing import preprocess_data
from src.interpretability.gradcam import (
    find_last_conv_layer,
    generate_gradcam_example,
    get_gradcam_overlay,
)
from src.seeds import set_seed


DOCUMENTED_ERROR_PATTERNS = [
    ("dog", "cat"),
    ("cat", "dog"),
    ("bird", "deer"),
    ("ship", "airplane"),
    ("truck", "automobile"),
    ("automobile", "truck"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera visualizaciones Grad-CAM para modelos entrenados."
    )
    parser.add_argument(
        "--model",
        choices=["improved_cnn", "transfer_model"],
        default="improved_cnn",
        help="Modelo a interpretar.",
    )
    return parser.parse_args()


def select_transfer_examples(y_true_labels, y_pred_labels):
    correct_indices = np.where(y_true_labels == y_pred_labels)[0]
    incorrect_indices = np.where(y_true_labels != y_pred_labels)[0]

    selected_correct = [int(idx) for idx in correct_indices[:2]]
    selected_incorrect = []

    class_to_index = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    for true_name, pred_name in DOCUMENTED_ERROR_PATTERNS:
        true_idx = class_to_index[true_name]
        pred_idx = class_to_index[pred_name]
        matches = np.where(
            (y_true_labels == true_idx) & (y_pred_labels == pred_idx)
        )[0]
        for idx in matches:
            selected_incorrect.append(int(idx))
            break
        if len(selected_incorrect) == 2:
            break

    if len(selected_incorrect) < 2:
        for idx in incorrect_indices:
            int_idx = int(idx)
            if int_idx not in selected_incorrect:
                selected_incorrect.append(int_idx)
            if len(selected_incorrect) == 2:
                break

    return selected_correct + selected_incorrect


def generate_transfer_gradcam_grid(model, x_test, y_true_labels, y_prob):
    output_dir = FIGURES_DIR / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "transfer_model_gradcam_examples.png"

    y_pred_labels = np.argmax(y_prob, axis=1)
    selected_indices = select_transfer_examples(y_true_labels, y_pred_labels)

    nested_model_name, last_conv_layer_name = find_last_conv_layer(model)
    layer_label = (
        f"{nested_model_name}/{last_conv_layer_name}"
        if nested_model_name
        else last_conv_layer_name
    )

    fig, axes = plt.subplots(len(selected_indices), 2, figsize=(8, 12))
    if len(selected_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    example_rows = []
    for row, idx in enumerate(selected_indices):
        true_label = int(y_true_labels[idx])
        pred_label = int(y_pred_labels[idx])
        confidence = float(y_prob[idx, pred_label])

        _, overlay, grad_predictions, _, _ = get_gradcam_overlay(
            model=model,
            image=x_test[idx],
            pred_index=pred_label,
            nested_model_name=nested_model_name,
            last_conv_layer_name=last_conv_layer_name,
        )

        grad_pred_label = int(np.argmax(grad_predictions[0]))
        if grad_pred_label != pred_label:
            raise ValueError(
                "La prediccion mostrada no coincide con el checkpoint "
                f"en el indice {idx}."
            )

        axes[row, 0].imshow(x_test[idx])
        axes[row, 0].axis("off")
        axes[row, 0].set_title(
            f"Original\nReal: {CLASS_NAMES[true_label]}",
            fontsize=10,
        )

        axes[row, 1].imshow(overlay)
        axes[row, 1].axis("off")
        axes[row, 1].set_title(
            "Grad-CAM\n"
            f"Pred: {CLASS_NAMES[pred_label]} ({confidence:.2f})",
            fontsize=10,
        )

        example_rows.append(
            {
                "index": idx,
                "true": CLASS_NAMES[true_label],
                "pred": CLASS_NAMES[pred_label],
                "confidence": confidence,
                "correct": true_label == pred_label,
            }
        )

    fig.suptitle(f"MobileNetV2 Grad-CAM - capa {layer_label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path, layer_label, example_rows


def generate_improved_cnn_examples(model, x_test, y_true_labels, y_pred_labels):
    correct_indices = np.where(y_true_labels == y_pred_labels)[0]
    incorrect_indices = np.where(y_true_labels != y_pred_labels)[0]
    selected_indices = list(correct_indices[:3]) + list(incorrect_indices[:3])

    print("Generando Grad-CAM...")
    for idx in selected_indices:
        output_path = generate_gradcam_example(
            model=model,
            image=x_test[idx],
            true_label=y_true_labels[idx],
            model_name="improved_cnn",
            image_index=int(idx),
        )
        print(f"Imagen Grad-CAM guardada en: {output_path}")


def main():
    args = parse_args()
    set_seed(42)

    model_name = args.model
    model_path = MODELS_DIR / f"{model_name}_best.keras"

    print(f"Cargando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    print("Cargando datos CIFAR-10...")
    x_train, y_train, x_val, y_val, x_test, y_test, _ = load_cifar10_data()

    print("Preprocesando datos...")
    _, _, _, _, x_test_p, y_test_p = preprocess_data(
        x_train, y_train, x_val, y_val, x_test, y_test
    )

    print("Generando predicciones...")
    y_prob = model.predict(x_test_p, verbose=1)

    y_true_labels = np.argmax(y_test_p, axis=1)
    y_pred_labels = np.argmax(y_prob, axis=1)

    if model_name == "transfer_model":
        output_path, layer_label, examples = generate_transfer_gradcam_grid(
            model,
            x_test_p,
            y_true_labels,
            y_prob,
        )

        print(f"Capa Grad-CAM: {layer_label}")
        for item in examples:
            status = "correcta" if item["correct"] else "incorrecta"
            print(
                "Ejemplo "
                f"{item['index']}: real={item['true']}, "
                f"pred={item['pred']}, conf={item['confidence']:.6f}, "
                f"{status}"
            )
        print(f"Figura Grad-CAM guardada en: {output_path}")
    else:
        generate_improved_cnn_examples(
            model,
            x_test_p,
            y_true_labels,
            y_pred_labels,
        )

    print("Proceso Grad-CAM finalizado correctamente.")


if __name__ == "__main__":
    main()
