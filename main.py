import argparse

from src.seeds import set_seed
from src.data.data_loader import load_cifar10_data
from src.data.preprocessing import preprocess_data
from src.models.baseline_cnn import build_baseline_cnn
from src.models.improved_cnn import build_improved_cnn
from src.models.transfer_model import build_transfer_model
from src.training.trainer import train_model
from src.visualization.training_plots import plot_training_history
from src.evaluation.metrics import calculate_classification_metrics
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.reports import generate_classification_report
from src.evaluation.error_analysis import (
    save_misclassified_report,
    plot_misclassified_samples,
)
from src.evaluation.transfer_evidence import evaluate_transfer_checkpoint
from src.config import MODELS_DIR


def get_model(model_name: str):
    if model_name == "baseline_cnn":
        return build_baseline_cnn()

    if model_name == "improved_cnn":
        return build_improved_cnn()

    if model_name == "transfer_model":
        return build_transfer_model()

    raise ValueError(f"Modelo no válido: {model_name}")


def main(model_name: str):
    set_seed(42)

    print(f"Ejecutando experimento: {model_name}")

    print("Cargando dataset CIFAR-10...")
    x_train, y_train, x_val, y_val, x_test, y_test, class_names = load_cifar10_data()

    print("Preprocesando datos...")
    x_train_p, y_train_p, x_val_p, y_val_p, x_test_p, y_test_p = preprocess_data(
        x_train, y_train, x_val, y_val, x_test, y_test
    )

    print("Construyendo modelo...")
    model = get_model(model_name)
    model.summary()

    print("Entrenando modelo...")
    history = train_model(
        model=model,
        x_train=x_train_p,
        y_train=y_train_p,
        x_val=x_val_p,
        y_val=y_val_p,
        model_name=model_name,
    )

    print("Guardando gráficas de entrenamiento...")
    plot_training_history(history, model_name)

    if model_name == "transfer_model":
        checkpoint_path = MODELS_DIR / "transfer_model_best.keras"
        print("Evaluando checkpoint oficial de transfer_model...")
        result = evaluate_transfer_checkpoint(checkpoint_path=checkpoint_path)
        validation = result["validation"]

        print("Validacion final transfer_model:")
        print(f"Test samples: {validation['test_samples']}")
        print(f"Correct predictions: {validation['correct_predictions']}")
        print(f"Incorrect predictions: {validation['incorrect_predictions']}")
        print(f"Accuracy metrics.json: {validation['accuracy_metrics_json']:.4f}")
        print(f"Accuracy desde y_pred: {validation['accuracy_from_predictions']:.4f}")
        print(f"Accuracy desde matriz: {validation['accuracy_from_confusion_matrix']:.4f}")
        print(f"Precision macro: {validation['precision_macro']:.4f}")
        print(f"Recall macro: {validation['recall_macro']:.4f}")
        print(f"F1 macro: {validation['f1_macro']:.4f}")
        print(f"Filas CSV errores: {validation['misclassified_csv_rows']}")
        print(f"Suma matriz confusion: {validation['confusion_matrix_total']}")
        print(f"Checkpoint utilizado: {validation['checkpoint_used']}")
        print(
            "Epoca checkpoint: "
            f"{validation['checkpoint_epoch_one_based']} "
            f"(monitor={validation['checkpoint_monitor']}, "
            f"mode={validation['checkpoint_mode']})"
        )
        print("Proceso finalizado correctamente.")
        return

    print("Evaluando modelo en conjunto de prueba...")
    y_pred = model.predict(x_test_p)

    metrics = calculate_classification_metrics(
        y_true=y_test_p,
        y_pred=y_pred,
        model_name=model_name,
    )

    print("Métricas obtenidas:")
    print(metrics)

    print("Generando matriz de confusión...")
    plot_confusion_matrix(y_test_p, y_pred, model_name)

    print("Generando reporte de clasificación...")
    generate_classification_report(y_test_p, y_pred, model_name)

    print("Analizando errores...")
    save_misclassified_report(y_test_p, y_pred, model_name)
    plot_misclassified_samples(x_test_p, y_test_p, y_pred, model_name)

    print("Proceso finalizado correctamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="baseline_cnn",
        choices=["baseline_cnn", "improved_cnn", "transfer_model"],
        help="Modelo a entrenar: baseline_cnn o improved_cnn",
    )

    args = parser.parse_args()
    main(args.model)
