import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from src.config import CLASS_NAMES, FIGURES_DIR, MODELS_DIR, REPORTS_TABLES_DIR
from src.data.data_loader import load_cifar10_data
from src.data.preprocessing import normalize_images
from src.seeds import set_seed


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


def load_transfer_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo guardado: {model_path}. "
            "Ejecuta primero el entrenamiento de transfer_model."
        )

    return tf.keras.models.load_model(model_path, compile=False)


def predict_labels(model, x_test):
    y_pred_prob = model.predict(x_test, verbose=1)
    return np.argmax(y_pred_prob, axis=1)


def plot_transfer_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)

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

    return cm


def plot_misclassified_examples(x_test_original, y_true, y_pred, output_path, num_samples):
    error_indices = np.where(y_true != y_pred)[0]
    selected_indices = error_indices[:num_samples]

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


def build_confused_classes_table(cm, top_n):
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

    df_confusions = pd.DataFrame(rows).sort_values(
        by="errores", ascending=False
    )

    return df_confusions.head(top_n).reset_index(drop=True)


def plot_confused_classes(df_confusions, output_path):
    df_plot = df_confusions.copy()
    df_plot["confusion"] = (
        df_plot["clase_real"] + " -> " + df_plot["clase_predicha"]
    )

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


def main(num_examples: int, top_n: int):
    set_seed(42)

    model_path = MODELS_DIR / "transfer_model_best.keras"
    figures_dir = FIGURES_DIR / "evaluation"
    figures_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando CIFAR-10...")
    _, _, _, _, x_test, y_test, _ = load_cifar10_data()
    y_test = y_test.astype("int64")
    x_test_model = normalize_images(x_test)

    print(f"Cargando modelo guardado: {model_path}")
    model = load_transfer_model(model_path)

    print("Generando predicciones...")
    y_pred = predict_labels(model, x_test_model)

    confusion_matrix_path = figures_dir / "transfer_model_confusion_matrix.png"
    misclassified_path = figures_dir / "transfer_model_misclassified_samples.png"
    confused_classes_path = figures_dir / "transfer_model_confused_classes.png"
    confused_classes_table_path = (
        REPORTS_TABLES_DIR / "transfer_model_confused_classes.csv"
    )

    print("Guardando matriz de confusion...")
    cm = plot_transfer_confusion_matrix(y_test, y_pred, confusion_matrix_path)

    print("Guardando ejemplos mal clasificados...")
    plot_misclassified_examples(
        x_test_original=x_test,
        y_true=y_test,
        y_pred=y_pred,
        output_path=misclassified_path,
        num_samples=num_examples,
    )

    print("Guardando tabla y grafico de clases mas confundidas...")
    df_confusions = build_confused_classes_table(cm, top_n=top_n)
    df_confusions.to_csv(confused_classes_table_path, index=False, encoding="utf-8-sig")
    plot_confused_classes(df_confusions, confused_classes_path)

    print("Evidencias generadas:")
    print(f"- {confusion_matrix_path}")
    print(f"- {misclassified_path}")
    print(f"- {confused_classes_path}")
    print(f"- {confused_classes_table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera evidencias complementarias de evaluacion para transfer_model."
    )
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    main(num_examples=args.num_examples, top_n=args.top_n)
