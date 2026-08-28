import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    CLASS_NAMES,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    REPORTS_TABLES_DIR,
    LOGS_DIR,
)
from src.data.data_loader import load_cifar10_data
from src.data.preprocessing import normalize_images


ERROR_CAUSES = {
    ("cat", "dog"): "Rasgos visuales similares entre animales domesticos y posturas parecidas.",
    ("dog", "cat"): "Rasgos visuales similares entre animales domesticos y fondos poco distintivos.",
    ("bird", "airplane"): "Siluetas en el cielo y fondos azules pueden generar confusion.",
    ("airplane", "bird"): "Objetos pequenos sobre fondo uniforme pueden compartir forma global.",
    ("automobile", "truck"): "Vehiculos terrestres con ruedas y vistas frontales o laterales similares.",
    ("truck", "automobile"): "Vehiculos terrestres con colores, encuadres y fondos de carretera similares.",
    ("deer", "horse"): "Animales cuadrupedos con forma corporal y fondos naturales similares.",
    ("horse", "deer"): "Animales cuadrupedos con textura y contexto visual parecido.",
    ("ship", "airplane"): "Fondos amplios y objetos pequenos con bajo detalle en 32x32 pixeles.",
    ("airplane", "ship"): "Fondo azul y baja resolucion pueden reducir rasgos diferenciales.",
}


def get_possible_cause(true_class: str, predicted_class: str) -> str:
    return ERROR_CAUSES.get(
        (true_class, predicted_class),
        "Baja resolucion de CIFAR-10, variacion de pose, iluminacion o fondo poco discriminativo.",
    )


def get_checkpoint_epoch(
    log_path: Path = LOGS_DIR / "transfer_model_training_log.csv",
) -> dict:
    if not log_path.exists():
        return {
            "checkpoint_epoch_zero_based": None,
            "checkpoint_epoch_one_based": None,
            "checkpoint_val_accuracy": None,
            "checkpoint_monitor": "val_accuracy",
            "checkpoint_mode": "max",
        }

    try:
        df_log = pd.read_csv(log_path)
    except pd.errors.EmptyDataError:
        return {
            "checkpoint_epoch_zero_based": None,
            "checkpoint_epoch_one_based": None,
            "checkpoint_val_accuracy": None,
            "checkpoint_monitor": "val_accuracy",
            "checkpoint_mode": "max",
        }

    if df_log.empty:
        return {
            "checkpoint_epoch_zero_based": None,
            "checkpoint_epoch_one_based": None,
            "checkpoint_val_accuracy": None,
            "checkpoint_monitor": "val_accuracy",
            "checkpoint_mode": "max",
        }

    df_log["val_accuracy"] = pd.to_numeric(df_log["val_accuracy"], errors="coerce")
    df_log = df_log.dropna(subset=["val_accuracy"])

    if df_log.empty:
        return {
            "checkpoint_epoch_zero_based": None,
            "checkpoint_epoch_one_based": None,
            "checkpoint_val_accuracy": None,
            "checkpoint_monitor": "val_accuracy",
            "checkpoint_mode": "max",
        }

    best_index = int(df_log["val_accuracy"].idxmax())
    best_row = df_log.loc[best_index]

    return {
        "checkpoint_epoch_zero_based": int(best_row["epoch"]),
        "checkpoint_epoch_one_based": int(best_row["epoch"]) + 1,
        "checkpoint_val_accuracy": float(best_row["val_accuracy"]),
        "checkpoint_monitor": "val_accuracy",
        "checkpoint_mode": "max",
    }


def load_transfer_test_data():
    _, _, _, _, x_test, y_test, _ = load_cifar10_data()
    y_true = y_test.astype("int64")
    x_test_model = normalize_images(x_test)

    return x_test, x_test_model, y_true


def predict_once(model: tf.keras.Model, x_test_model: np.ndarray):
    y_prob = model.predict(x_test_model, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    return y_pred, y_prob


def build_misclassified_dataframe(y_true, y_pred):
    error_indices = np.where(y_true != y_pred)[0]

    return pd.DataFrame(
        {
            "index": error_indices,
            "true_label": y_true[error_indices],
            "true_class": [CLASS_NAMES[i] for i in y_true[error_indices]],
            "predicted_label": y_pred[error_indices],
            "predicted_class": [CLASS_NAMES[i] for i in y_pred[error_indices]],
        }
    )


def build_confused_classes_table(cm, top_n: int):
    rows = []

    for true_label, true_class in enumerate(CLASS_NAMES):
        for predicted_label, predicted_class in enumerate(CLASS_NAMES):
            if true_label == predicted_label:
                continue

            count = int(cm[true_label, predicted_label])
            if count == 0:
                continue

            rows.append(
                {
                    "clase_real": true_class,
                    "clase_predicha": predicted_class,
                    "errores": count,
                    "posible_causa": get_possible_cause(true_class, predicted_class),
                }
            )

    df_confusions = pd.DataFrame(rows)
    if df_confusions.empty:
        return df_confusions

    return df_confusions.sort_values(by="errores", ascending=False).head(top_n).reset_index(drop=True)


def save_confusion_matrix_figure(cm, output_path: Path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Matriz de confusion - transfer_model")
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_misclassified_figure(x_test_original, y_true, y_pred, output_path: Path, num_samples: int):
    error_indices = np.where(y_true != y_pred)[0]
    selected_indices = error_indices[:num_samples]

    if len(selected_indices) == 0:
        return

    cols = 4
    rows = int(np.ceil(len(selected_indices) / cols))

    plt.figure(figsize=(12, max(3, rows * 2.7)))

    for position, image_index in enumerate(selected_indices, start=1):
        plt.subplot(rows, cols, position)
        plt.imshow(x_test_original[image_index])
        plt.axis("off")
        plt.title(
            f"Real: {CLASS_NAMES[y_true[image_index]]}\n"
            f"Pred: {CLASS_NAMES[y_pred[image_index]]}",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_confused_classes_figure(df_confusions: pd.DataFrame, output_path: Path):
    if df_confusions.empty:
        return

    df_plot = df_confusions.copy()
    df_plot["confusion"] = df_plot["clase_real"] + " -> " + df_plot["clase_predicha"]

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=df_plot,
        x="errores",
        y="confusion",
        hue="confusion",
        palette="viridis",
        legend=False,
    )
    plt.title("Clases mas confundidas - transfer_model")
    plt.xlabel("Numero de errores")
    plt.ylabel("Clase real -> clase predicha")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def validate_transfer_artifacts(cm, y_true, y_pred, df_errors):
    test_samples = int(len(y_true))
    correct_predictions = int(np.sum(y_true == y_pred))
    incorrect_predictions = int(np.sum(y_true != y_pred))
    accuracy_from_predictions = correct_predictions / test_samples
    accuracy_from_matrix = int(np.trace(cm)) / int(np.sum(cm))

    checks = {
        "accuracy_matches_confusion_matrix": np.isclose(
            accuracy_from_predictions,
            accuracy_from_matrix,
            rtol=0,
            atol=1e-12,
        ),
        "confusion_matrix_total_matches_test_samples": int(np.sum(cm)) == test_samples,
        "correct_plus_errors_matches_test_samples": correct_predictions + incorrect_predictions == test_samples,
        "errors_match_csv_rows": incorrect_predictions == len(df_errors),
    }

    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise ValueError(
            "La evaluacion de transfer_model no es coherente. "
            f"Comprobaciones fallidas: {', '.join(failed_checks)}"
        )

    return {
        "test_samples": test_samples,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": incorrect_predictions,
        "accuracy_from_predictions": accuracy_from_predictions,
        "accuracy_from_confusion_matrix": accuracy_from_matrix,
        "confusion_matrix_total": int(np.sum(cm)),
        "misclassified_csv_rows": int(len(df_errors)),
        "checks": checks,
    }


def evaluate_transfer_checkpoint(
    checkpoint_path: Path = MODELS_DIR / "transfer_model_best.keras",
    num_examples: int = 16,
    top_n: int = 10,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No se encontro el checkpoint oficial: {checkpoint_path}. "
            "Ejecuta primero python main.py --model transfer_model."
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = FIGURES_DIR / "evaluation"
    figures_dir.mkdir(parents=True, exist_ok=True)

    x_test_original, x_test_model, y_true = load_transfer_test_data()
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    y_pred, y_prob = predict_once(model, x_test_model)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    df_errors = build_misclassified_dataframe(y_true, y_pred)
    validation = validate_transfer_artifacts(cm, y_true, y_pred, df_errors)
    checkpoint_info = get_checkpoint_epoch()

    metrics = {
        "model": "transfer_model",
        "model_name": "transfer_model",
        "checkpoint_used": str(checkpoint_path),
        "random_seed": RANDOM_SEED,
        "test_samples": validation["test_samples"],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "correct_predictions": validation["correct_predictions"],
        "incorrect_predictions": validation["incorrect_predictions"],
        **checkpoint_info,
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    df_report = pd.DataFrame(report).transpose()

    df_cm = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    df_confusions = build_confused_classes_table(cm, top_n=top_n)

    metrics_path = METRICS_DIR / "transfer_model_metrics.json"
    report_json_path = METRICS_DIR / "transfer_model_classification_report.json"
    report_csv_path = REPORTS_TABLES_DIR / "transfer_model_classification_report.csv"
    cm_csv_path = REPORTS_TABLES_DIR / "transfer_model_confusion_matrix.csv"
    errors_csv_path = PREDICTIONS_DIR / "transfer_model_misclassified_samples.csv"
    confused_classes_csv_path = REPORTS_TABLES_DIR / "transfer_model_confused_classes.csv"
    confusion_matrix_path = figures_dir / "transfer_model_confusion_matrix.png"
    misclassified_path = figures_dir / "transfer_model_misclassified_samples.png"
    confused_classes_path = figures_dir / "transfer_model_confused_classes.png"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    with open(report_json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    df_report.to_csv(report_csv_path, index=True, encoding="utf-8-sig")
    df_cm.to_csv(cm_csv_path, index=True, encoding="utf-8-sig")
    df_errors.to_csv(errors_csv_path, index=False, encoding="utf-8-sig")
    df_confusions.to_csv(confused_classes_csv_path, index=False, encoding="utf-8-sig")

    save_confusion_matrix_figure(cm, confusion_matrix_path)
    save_misclassified_figure(x_test_original, y_true, y_pred, misclassified_path, num_examples)
    save_confused_classes_figure(df_confusions, confused_classes_path)

    validation["accuracy_metrics_json"] = metrics["accuracy"]
    validation["precision_macro"] = metrics["precision_macro"]
    validation["recall_macro"] = metrics["recall_macro"]
    validation["f1_macro"] = metrics["f1_macro"]
    validation["checkpoint_used"] = str(checkpoint_path)
    validation.update(checkpoint_info)

    return {
        "metrics": metrics,
        "validation": validation,
        "paths": {
            "metrics": metrics_path,
            "classification_report_json": report_json_path,
            "classification_report_csv": report_csv_path,
            "confusion_matrix_csv": cm_csv_path,
            "misclassified_csv": errors_csv_path,
            "confused_classes_csv": confused_classes_csv_path,
            "confusion_matrix_figure": confusion_matrix_path,
            "misclassified_figure": misclassified_path,
            "confused_classes_figure": confused_classes_path,
        },
    }
