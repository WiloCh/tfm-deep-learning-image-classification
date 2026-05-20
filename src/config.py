from pathlib import Path

# =========================
# Rutas base del proyecto
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_TABLES_DIR = REPORTS_DIR / "tables"
REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"

CONFIGS_DIR = BASE_DIR / "configs"

# =========================
# Parámetros generales
# =========================
RANDOM_SEED = 42
NUM_CLASSES = 10
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# =========================
# Configuración por defecto
# =========================
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1