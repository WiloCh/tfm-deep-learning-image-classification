import json
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report

from src.config import CLASS_NAMES, METRICS_DIR, REPORTS_TABLES_DIR


def generate_classification_report(y_true, y_pred, model_name: str):
    """
    Genera el reporte de clasificación por clase:
    - precision
    - recall
    - f1-score
    - support
    """

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    y_pred_labels = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred

    report = classification_report(
        y_true_labels,
        y_pred_labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    json_path = METRICS_DIR / f"{model_name}_classification_report.json"
    csv_path = REPORTS_TABLES_DIR / f"{model_name}_classification_report.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(csv_path, index=True, encoding="utf-8-sig")

    return df_report