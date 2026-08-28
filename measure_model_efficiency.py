import csv
import gc
import os
import platform
import time
from ctypes import Structure, byref, c_ulonglong, c_uint, sizeof, windll

import numpy as np
import tensorflow as tf

from src.config import DEFAULT_BATCH_SIZE, MODELS_DIR, REPORTS_TABLES_DIR
from src.data.data_loader import load_cifar10_data
from src.data.preprocessing import preprocess_data
from src.seeds import set_seed


MODEL_NAMES = ["baseline_cnn", "improved_cnn", "transfer_model"]
REPEATS = 3
WARMUP_SAMPLES = DEFAULT_BATCH_SIZE


class MemoryStatusEx(Structure):
    _fields_ = [
        ("dwLength", c_uint),
        ("dwMemoryLoad", c_uint),
        ("ullTotalPhys", c_ulonglong),
        ("ullAvailPhys", c_ulonglong),
        ("ullTotalPageFile", c_ulonglong),
        ("ullAvailPageFile", c_ulonglong),
        ("ullTotalVirtual", c_ulonglong),
        ("ullAvailVirtual", c_ulonglong),
        ("ullAvailExtendedVirtual", c_ulonglong),
    ]


def get_total_ram_gib() -> float | None:
    if platform.system() != "Windows":
        return None

    status = MemoryStatusEx()
    status.dwLength = sizeof(status)
    if not windll.kernel32.GlobalMemoryStatusEx(byref(status)):
        return None

    return status.ullTotalPhys / (1024**3)


def get_hardware_info() -> dict[str, str]:
    gpus = tf.config.list_physical_devices("GPU")
    total_ram_gib = get_total_ram_gib()

    return {
        "system": platform.platform(),
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "ram": f"{total_ram_gib:.2f} GiB" if total_ram_gib else "unknown",
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "device": "GPU" if gpus else "CPU",
        "gpu_used": "SI" if gpus else "NO",
    }


def count_trainable_params(model: tf.keras.Model) -> tuple[int, int, int]:
    trainable = int(
        np.sum([tf.keras.backend.count_params(weight) for weight in model.trainable_weights])
    )
    non_trainable = int(
        np.sum(
            [tf.keras.backend.count_params(weight) for weight in model.non_trainable_weights]
        )
    )
    return int(model.count_params()), trainable, non_trainable


def load_test_data():
    x_train, y_train, x_val, y_val, x_test, y_test, _ = load_cifar10_data()
    _, _, _, _, x_test_p, y_test_p = preprocess_data(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )
    return x_test_p, y_test_p


def measure_inference(model: tf.keras.Model, x_test, batch_size: int) -> tuple[float, float]:
    model.predict(x_test[:WARMUP_SAMPLES], batch_size=batch_size, verbose=0)

    timings = []
    for _ in range(REPEATS):
        gc.collect()
        start = time.perf_counter()
        model.predict(x_test, batch_size=batch_size, verbose=0)
        end = time.perf_counter()
        timings.append(end - start)

    return float(np.mean(timings)), float(np.std(timings, ddof=1))


def measure_model(model_name: str, x_test, batch_size: int, device: str):
    model_path = MODELS_DIR / f"{model_name}_best.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {model_path}")

    model_size_bytes = model_path.stat().st_size
    model_size_mib = model_size_bytes / (1024**2)

    print(f"Evaluando eficiencia de inferencia: {model_name}")
    model = tf.keras.models.load_model(model_path, compile=False)
    total_params, trainable_params, non_trainable_params = count_trainable_params(model)

    mean_seconds, std_seconds = measure_inference(model, x_test, batch_size)
    mean_ms_per_image = (mean_seconds / len(x_test)) * 1000.0
    images_per_second = len(x_test) / mean_seconds

    tf.keras.backend.clear_session()
    del model
    gc.collect()

    return {
        "model_name": model_name,
        "model_size_bytes": model_size_bytes,
        "model_size_mib": round(model_size_mib, 4),
        "mean_inference_seconds": round(mean_seconds, 6),
        "std_inference_seconds": round(std_seconds, 6),
        "mean_ms_per_image": round(mean_ms_per_image, 6),
        "images_per_second": round(images_per_second, 4),
        "test_samples": len(x_test),
        "batch_size": batch_size,
        "device": device,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }


def write_results(rows):
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_TABLES_DIR / "model_inference_efficiency.csv"
    fieldnames = [
        "model_name",
        "model_size_bytes",
        "model_size_mib",
        "mean_inference_seconds",
        "std_inference_seconds",
        "mean_ms_per_image",
        "images_per_second",
        "test_samples",
        "batch_size",
        "device",
        "total_params",
        "trainable_params",
        "non_trainable_params",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main():
    set_seed(42)
    batch_size = DEFAULT_BATCH_SIZE
    hardware = get_hardware_info()

    print("Cargando datos oficiales de test CIFAR-10...")
    x_test, _ = load_test_data()
    if len(x_test) != 10000:
        raise ValueError(f"Se esperaban 10000 muestras de test, se obtuvieron {len(x_test)}.")

    rows = [
        measure_model(model_name, x_test, batch_size, hardware["device"])
        for model_name in MODEL_NAMES
    ]

    output_path = write_results(rows)

    print(f"Tabla guardada en: {output_path}")
    print("HARDWARE_UTILIZADO_PARA_LA_MEDICION_DE_INFERENCIA")
    for key, value in hardware.items():
        print(f"{key}: {value}")

    print("PARAMETROS")
    for row in rows:
        print(
            f"{row['model_name']}: total={row['total_params']}, "
            f"trainable={row['trainable_params']}, "
            f"non_trainable={row['non_trainable_params']}"
        )


if __name__ == "__main__":
    main()
