import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import METRICS_DIR


def calculate_classification_metrics(y_true, y_pred, model_name: str):
    """
    Calcula métricas principales para clasificación multiclase:
    - Accuracy
    - Precision macro
    - Recall macro
    - F1 macro
    """

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    y_pred_labels = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true_labels, y_pred_labels)),
        "precision_macro": float(
            precision_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
        ),
    }

    output_path = METRICS_DIR / f"{model_name}_metrics.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    return metrics