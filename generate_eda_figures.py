import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.seeds import set_seed
from src.data.data_loader import load_cifar10_data
from src.config import CLASS_NAMES, FIGURES_DIR, REPORTS_TABLES_DIR


def plot_class_distribution(y_train, y_test):
    output_dir = FIGURES_DIR / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    train_counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
    test_counts = np.bincount(y_test, minlength=len(CLASS_NAMES))

    df = pd.DataFrame({
        "class": CLASS_NAMES,
        "train_count": train_counts,
        "test_count": test_counts,
        "total": train_counts + test_counts,
    })

    df.to_csv(REPORTS_TABLES_DIR / "cifar10_class_distribution.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 5))
    plt.bar(CLASS_NAMES, train_counts, label="Entrenamiento")
    plt.bar(CLASS_NAMES, test_counts, bottom=train_counts, label="Prueba")
    plt.title("Distribución de imágenes por clase - CIFAR-10")
    plt.xlabel("Clase")
    plt.ylabel("Número de imágenes")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cifar10_class_distribution.png", dpi=300)
    plt.close()


def plot_sample_images(x_train, y_train, samples_per_class=5):
    output_dir = FIGURES_DIR / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))

    plot_index = 1

    for class_id, class_name in enumerate(CLASS_NAMES):
        indices = np.where(y_train == class_id)[0][:samples_per_class]

        for idx in indices:
            plt.subplot(len(CLASS_NAMES), samples_per_class, plot_index)
            plt.imshow(x_train[idx])
            plt.axis("off")

            if plot_index <= samples_per_class:
                plt.title(f"Ejemplo {plot_index}", fontsize=8)

            if plot_index % samples_per_class == 1:
                plt.ylabel(class_name, fontsize=8)

            plot_index += 1

    plt.suptitle("Ejemplos de imágenes por clase - CIFAR-10", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "cifar10_sample_images.png", dpi=300)
    plt.close()


def main():
    set_seed(42)

    print("Cargando CIFAR-10...")
    x_train, y_train, x_val, y_val, x_test, y_test, class_names = load_cifar10_data()

    print("Generando distribución de clases...")
    plot_class_distribution(y_train, y_test)

    print("Generando ejemplos de imágenes...")
    plot_sample_images(x_train, y_train)

    print("Figuras EDA generadas correctamente.")


if __name__ == "__main__":
    main()