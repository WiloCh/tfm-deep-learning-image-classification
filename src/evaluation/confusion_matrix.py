import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from src.config import CLASS_NAMES, FIGURES_DIR


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    """
    Genera y guarda la matriz de confusión del modelo.
    """

    output_dir = FIGURES_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    y_pred_labels = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred

    cm = confusion_matrix(y_true_labels, y_pred_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title(f"Matriz de confusión - {model_name}")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = output_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return cm