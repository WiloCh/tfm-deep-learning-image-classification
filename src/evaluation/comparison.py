import json
import pandas as pd
import matplotlib.pyplot as plt

from src.config import METRICS_DIR, REPORTS_TABLES_DIR, FIGURES_DIR


def load_metrics_files():
    """
    Carga todos los archivos *_metrics.json generados por los modelos.
    """

    metrics_files = list(METRICS_DIR.glob("*_metrics.json"))

    metrics_list = []

    for file_path in metrics_files:
        with open(file_path, "r", encoding="utf-8") as file:
            metrics_list.append(json.load(file))

    return metrics_list


def build_comparison_table():
    """
    Construye una tabla comparativa entre modelos.
    """

    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_list = load_metrics_files()

    df_comparison = pd.DataFrame(metrics_list)

    if not df_comparison.empty:
        metrics_columns = [
            "model_name",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]
        df_comparison = df_comparison[metrics_columns]
        df_comparison = df_comparison.sort_values(
            by="f1_macro",
            ascending=False
        )

    output_path = REPORTS_TABLES_DIR / "model_comparison.csv"
    df_comparison.to_csv(output_path, index=False, encoding="utf-8-sig")

    return df_comparison


def plot_model_comparison():
    """
    Genera una gráfica comparativa de métricas entre modelos.
    """

    output_dir = FIGURES_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_comparison = build_comparison_table()

    if df_comparison.empty:
        raise ValueError("No existen archivos de métricas para comparar.")

    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    df_plot = df_comparison.set_index("model_name")[metrics]

    ax = df_plot.plot(kind="bar", figsize=(10, 6))

    plt.title("Comparación de modelos")
    plt.xlabel("Modelo")
    plt.ylabel("Valor de la métrica")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(title="Métricas")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    output_path = output_dir / "model_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return df_comparison
