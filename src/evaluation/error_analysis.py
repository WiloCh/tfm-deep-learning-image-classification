import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import CLASS_NAMES, FIGURES_DIR, PREDICTIONS_DIR


def get_misclassified_samples(y_true, y_pred):
    """
    Identifica los índices de las imágenes mal clasificadas.
    """

    y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    y_pred_labels = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred

    misclassified_indices = np.where(y_true_labels != y_pred_labels)[0]

    return misclassified_indices, y_true_labels, y_pred_labels


def save_misclassified_report(y_true, y_pred, model_name: str):
    """
    Guarda un CSV con las predicciones incorrectas.
    """

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    misclassified_indices, y_true_labels, y_pred_labels = get_misclassified_samples(
        y_true, y_pred
    )

    df_errors = pd.DataFrame({
        "index": misclassified_indices,
        "true_label": y_true_labels[misclassified_indices],
        "true_class": [CLASS_NAMES[i] for i in y_true_labels[misclassified_indices]],
        "predicted_label": y_pred_labels[misclassified_indices],
        "predicted_class": [CLASS_NAMES[i] for i in y_pred_labels[misclassified_indices]],
    })

    output_path = PREDICTIONS_DIR / f"{model_name}_misclassified_samples.csv"
    df_errors.to_csv(output_path, index=False, encoding="utf-8-sig")

    return df_errors


def plot_misclassified_samples(x_data, y_true, y_pred, model_name: str, num_samples: int = 16):
    """
    Genera una figura con ejemplos de imágenes mal clasificadas.
    """

    output_dir = FIGURES_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    misclassified_indices, y_true_labels, y_pred_labels = get_misclassified_samples(
        y_true, y_pred
    )

    selected_indices = misclassified_indices[:num_samples]

    cols = 4
    rows = int(np.ceil(num_samples / cols))

    plt.figure(figsize=(12, 10))

    for i, idx in enumerate(selected_indices):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(x_data[idx])
        plt.axis("off")
        plt.title(
            f"Real: {CLASS_NAMES[y_true_labels[idx]]}\nPred: {CLASS_NAMES[y_pred_labels[idx]]}",
            fontsize=9,
        )

    plt.tight_layout()

    output_path = output_dir / f"{model_name}_misclassified_samples.png"
    plt.savefig(output_path, dpi=300)
    plt.close()