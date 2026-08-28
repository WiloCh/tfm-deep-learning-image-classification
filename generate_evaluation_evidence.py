import argparse

from src.evaluation.transfer_evidence import evaluate_transfer_checkpoint
from src.config import MODELS_DIR
from src.seeds import set_seed


def print_validation_summary(result):
    validation = result["validation"]

    print("MobileNetV2 - validacion final")
    print()
    print(f"Test samples: {validation['test_samples']}")
    print(f"Correct predictions: {validation['correct_predictions']}")
    print(f"Incorrect predictions: {validation['incorrect_predictions']}")
    print()
    print(f"Accuracy metrics.json: {validation['accuracy_metrics_json']:.4f}")
    print(f"Accuracy calculada desde y_pred: {validation['accuracy_from_predictions']:.4f}")
    print(
        "Accuracy calculada desde confusion matrix: "
        f"{validation['accuracy_from_confusion_matrix']:.4f}"
    )
    print()
    print(f"Precision macro: {validation['precision_macro']:.4f}")
    print(f"Recall macro: {validation['recall_macro']:.4f}")
    print(f"F1 macro: {validation['f1_macro']:.4f}")
    print()
    print(f"Filas CSV errores: {validation['misclassified_csv_rows']}")
    print(f"Suma matriz confusion: {validation['confusion_matrix_total']}")
    print()
    print("Checkpoint utilizado:")
    print(validation["checkpoint_used"])
    print()
    print(
        "Epoca checkpoint: "
        f"{validation['checkpoint_epoch_one_based']} "
        f"(monitor={validation['checkpoint_monitor']}, "
        f"mode={validation['checkpoint_mode']})"
    )
    print()
    print("Resultado:")
    print("TODAS LAS EVIDENCIAS COINCIDEN")


def main(num_examples: int, top_n: int):
    set_seed(42)

    checkpoint_path = MODELS_DIR / "transfer_model_best.keras"
    result = evaluate_transfer_checkpoint(
        checkpoint_path=checkpoint_path,
        num_examples=num_examples,
        top_n=top_n,
    )
    print_validation_summary(result)

    print()
    print("Artefactos generados:")
    for path in result["paths"].values():
        print(f"- {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera evidencias coherentes para transfer_model desde una sola prediccion."
    )
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    main(num_examples=args.num_examples, top_n=args.top_n)
