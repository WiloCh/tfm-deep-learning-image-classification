import matplotlib.pyplot as plt

from src.config import FIGURES_DIR


def plot_training_history(history, model_name: str):
    """
    Genera y guarda las curvas de entrenamiento:
    - Accuracy vs Val Accuracy
    - Loss vs Val Loss
    """

    output_dir = FIGURES_DIR / "training"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Entrenamiento")
    plt.plot(history.history["val_accuracy"], label="Validación")
    plt.title(f"Accuracy - {model_name}")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_accuracy.png", dpi=300)
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validación")
    plt.title(f"Loss - {model_name}")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_loss.png", dpi=300)
    plt.close()