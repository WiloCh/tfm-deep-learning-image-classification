import numpy as np
import tensorflow as tf

from src.seeds import set_seed
from src.data.data_loader import load_cifar10_data
from src.data.preprocessing import preprocess_data
from src.interpretability.gradcam import generate_gradcam_example
from src.config import MODELS_DIR


def main():
    set_seed(42)

    model_name = "improved_cnn"
    model_path = MODELS_DIR / f"{model_name}_best.keras"

    print(f"Cargando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("Cargando datos CIFAR-10...")
    x_train, y_train, x_val, y_val, x_test, y_test, class_names = load_cifar10_data()

    print("Preprocesando datos...")
    _, _, _, _, x_test_p, y_test_p = preprocess_data(
        x_train, y_train, x_val, y_val, x_test, y_test
    )

    print("Generando predicciones...")
    y_pred = model.predict(x_test_p)

    y_true_labels = np.argmax(y_test_p, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    correct_indices = np.where(y_true_labels == y_pred_labels)[0]
    incorrect_indices = np.where(y_true_labels != y_pred_labels)[0]

    selected_indices = list(correct_indices[:3]) + list(incorrect_indices[:3])

    print("Generando Grad-CAM...")
    for idx in selected_indices:
        output_path = generate_gradcam_example(
            model=model,
            image=x_test_p[idx],
            true_label=y_true_labels[idx],
            model_name=model_name,
            image_index=int(idx),
        )
        print(f"Imagen Grad-CAM guardada en: {output_path}")

    print("Proceso Grad-CAM finalizado correctamente.")


if __name__ == "__main__":
    main()