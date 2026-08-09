# 1. Informacion general del repositorio

| Elemento | Valor verificado | Evidencia |
|---|---|---|
| Nombre del proyecto | `tfm-image-classification` | Nombre de la carpeta raiz |
| Ruta local | `C:\Users\William Chugchilan\OneDrive\Documentos\TFM\Codigo\tfm-image-classification` | Contexto local del repositorio |
| URL GitHub | `https://github.com/WiloCh/tfm-deep-learning-image-classification.git` | `git remote -v`; README, lineas 136-137 |
| Rama actual | `dev_wilo` | `git branch --show-current` |
| Commit actual | `79d36ab94e8cd63c469ce986195f7fb037d1cd84` | `git rev-parse HEAD` |
| Estado Git | Hay cambios no confirmados previos: `README.md`, dos figuras de `transfer_model`, `generate_evaluation_evidence.py`, `reports/evidencias_experimentales.md`, figura y tabla de clases confundidas | `git status --short` |
| Finalidad general | Proyecto TFM para clasificacion automatica de imagenes con Deep Learning, CNN, transferencia y visual analytics | README, lineas 1-11 |
| Dataset | CIFAR-10 | README, lineas 15-24; `src/data/data_loader.py`, lineas 19-32 |

Archivos principales verificados: `main.py`, `README.md`, `requirements.txt`, `generate_eda_figures.py`, `generate_gradcam.py`, `generate_evaluation_evidence.py`, y los modulos de `src/`.

# 2. Arbol real del proyecto

```text
tfm-image-classification/
|-- .git/
|-- .vscode/
|-- configs/
|-- data/
|   |-- external/
|   |-- processed/
|   `-- raw/
|-- notebooks/
|-- outputs/
|   |-- figures/
|   |   |-- eda/
|   |   |-- evaluation/
|   |   |-- gradcam/
|   |   `-- training/
|   |-- logs/
|   |-- metrics/
|   |-- models/
|   `-- predictions/
|-- reports/
|   |-- figures/
|   |-- tables/
|   |-- evidencias_experimentales.md
|   `-- documentacion_codigo_datos.md
|-- src/
|   |-- config.py
|   |-- seeds.py
|   |-- data/
|   |   |-- augmentation.py
|   |   |-- data_loader.py
|   |   `-- preprocessing.py
|   |-- evaluation/
|   |   |-- comparison.py
|   |   |-- confusion_matrix.py
|   |   |-- error_analysis.py
|   |   |-- metrics.py
|   |   `-- reports.py
|   |-- interpretability/
|   |   `-- gradcam.py
|   |-- models/
|   |   |-- baseline_cnn.py
|   |   |-- improved_cnn.py
|   |   `-- transfer_model.py
|   |-- training/
|   |   |-- callbacks.py
|   |   `-- trainer.py
|   `-- visualization/
|       `-- training_plots.py
|-- tests/
|-- generate_evaluation_evidence.py
|-- generate_eda_figures.py
|-- generate_gradcam.py
|-- main.py
|-- README.md
`-- requirements.txt
```

Observacion verificable: `configs/`, `notebooks/`, `tests/`, `reports/figures/` y los subdirectorios `data/raw`, `data/processed`, `data/external` no muestran archivos en la inspeccion actual.

# 3. Responsabilidad de cada directorio

| Directorio | Responsabilidad | Entradas | Salidas |
|---|---|---|---|
| `configs/` | Carpeta prevista para configuraciones externas; actualmente sin archivos visibles | No encontrado | No encontrado |
| `data/` | Estructura reservada para datos locales crudos, procesados o externos; CIFAR-10 se carga desde Keras, no desde archivos locales | Dataset descargado por `tensorflow.keras.datasets.cifar10` | No se observaron archivos locales generados |
| `notebooks/` | Carpeta prevista para notebooks exploratorios; actualmente vacia | No encontrado | No encontrado |
| `src/` | Codigo fuente modular: datos, modelos, entrenamiento, evaluacion, visualizacion e interpretabilidad | Configuracion, CIFAR-10, modelos Keras | Funciones usadas por scripts y `main.py` |
| `outputs/` | Artefactos generados por ejecuciones: modelos, metricas, logs, predicciones y figuras | Resultados de entrenamiento/evaluacion | `.keras`, `.json`, `.csv`, `.png` |
| `reports/` | Insumos y tablas para documentacion del TFM | Reportes generados y tablas derivadas | CSV de tablas, informes Markdown |
| `tests/` | Carpeta prevista para pruebas automatizadas; actualmente sin archivos visibles | No encontrado | No encontrado |

# 4. Responsabilidad de cada modulo

| Archivo | Funcion principal | Funciones/clases relevantes | Utilizado por |
|---|---|---|---|
| `src/config.py` | Centraliza rutas, constantes, clases y parametros por defecto | `BASE_DIR`, `CLASS_NAMES`, `INPUT_SHAPE`, `DEFAULT_BATCH_SIZE`, `DEFAULT_EPOCHS` | Todo el proyecto |
| `src/seeds.py` | Fija semillas y activa determinismo cuando TensorFlow lo permite | `set_seed` | `main.py`, scripts auxiliares |
| `src/data/data_loader.py` | Carga CIFAR-10 y separa train/validacion/test | `load_cifar10_data`, `get_dataset_shapes` | `main.py`, scripts EDA/Grad-CAM/evidencias |
| `src/data/preprocessing.py` | Normaliza imagenes y codifica etiquetas | `normalize_images`, `encode_labels`, `preprocess_data` | `main.py`, `generate_gradcam.py`, `generate_evaluation_evidence.py` |
| `src/data/augmentation.py` | Define aumento de datos | `build_data_augmentation` | `src/models/improved_cnn.py` |
| `src/models/baseline_cnn.py` | Construye CNN base | `build_baseline_cnn` | `main.py` |
| `src/models/improved_cnn.py` | Construye CNN mejorada con augmentation, BN y dropout | `build_improved_cnn` | `main.py` |
| `src/models/transfer_model.py` | Construye modelo MobileNetV2 congelado con cabeza clasificadora | `build_transfer_model` | `main.py` |
| `src/training/trainer.py` | Ejecuta `model.fit` con callbacks | `train_model` | `main.py` |
| `src/training/callbacks.py` | Define checkpoint, early stopping, reduccion LR y CSVLogger | `build_callbacks` | `src/training/trainer.py` |
| `src/evaluation/metrics.py` | Calcula y guarda metricas globales | `calculate_classification_metrics` | `main.py` |
| `src/evaluation/confusion_matrix.py` | Genera matriz de confusion visual | `plot_confusion_matrix` | `main.py` |
| `src/evaluation/reports.py` | Genera classification report JSON y CSV | `generate_classification_report` | `main.py` |
| `src/evaluation/error_analysis.py` | Detecta errores y genera CSV/PNG de mal clasificados | `get_misclassified_samples`, `save_misclassified_report`, `plot_misclassified_samples` | `main.py` |
| `src/evaluation/comparison.py` | Agrega metricas de modelos y genera comparacion | `load_metrics_files`, `build_comparison_table`, `plot_model_comparison` | Uso manual o scripts externos; no llamado por `main.py` |
| `src/visualization/training_plots.py` | Guarda curvas de entrenamiento | `plot_training_history` | `main.py` |
| `src/interpretability/gradcam.py` | Implementa Grad-CAM sobre ultima capa Conv2D | `find_last_conv_layer`, `make_gradcam_heatmap`, `overlay_heatmap`, `generate_gradcam_example` | `generate_gradcam.py` |
| `main.py` | Punto de entrada del experimento principal por modelo | `get_model`, `main` | Usuario por CLI |
| `generate_eda_figures.py` | Genera figuras y tabla EDA de CIFAR-10 | `plot_class_distribution`, `plot_sample_images`, `main` | Usuario por CLI |
| `generate_gradcam.py` | Genera Grad-CAM para `improved_cnn_best.keras` | `main` | Usuario por CLI |
| `generate_evaluation_evidence.py` | Genera evidencias complementarias de evaluacion para `transfer_model` | `load_transfer_model`, `plot_transfer_confusion_matrix`, `build_confused_classes_table`, `main` | Usuario por CLI |

# 5. Tecnologias y dependencias

| Dependencia | Version exacta | Evidencia |
|---|---:|---|
| TensorFlow | 2.21.0 | `requirements.txt`, linea 1 |
| NumPy | 2.0.2 | `requirements.txt`, linea 2 |
| pandas | 2.2.3 | `requirements.txt`, linea 3 |
| matplotlib | 3.9.2 | `requirements.txt`, linea 4 |
| scikit-learn | 1.5.2 | `requirements.txt`, linea 5 |
| seaborn | 0.13.2 | `requirements.txt`, linea 6 |
| opencv-python | 4.10.0.84 | `requirements.txt`, linea 7 |
| Pillow | 11.0.0 | `requirements.txt`, linea 8 |
| PyYAML | 6.0.2 | `requirements.txt`, linea 9 |
| jupyter | 1.1.1 | `requirements.txt`, linea 10 |
| notebook | 7.3.2 | `requirements.txt`, linea 11 |
| ipykernel | 6.29.5 | `requirements.txt`, linea 12 |
| tqdm | 4.67.1 | `requirements.txt`, linea 13 |

| Elemento buscado | Estado |
|---|---|
| Version exacta de Python | No encontrado en el repositorio |
| Sistema operativo | No encontrado en el repositorio |
| GPU | No encontrado en el repositorio |
| CUDA | No encontrado en el repositorio |
| cuDNN | No encontrado en el repositorio |
| Entorno virtual/Docker | No encontrado en el repositorio |

# 6. Configuracion central

| Parametro | Valor | Archivo | Uso |
|---|---|---|---|
| `BASE_DIR` | Raiz del proyecto | `src/config.py`, linea 6 | Construccion de rutas |
| `DATA_DIR` | `data/` | `src/config.py`, linea 8 | Rutas de datos |
| `RAW_DATA_DIR` | `data/raw/` | `src/config.py`, linea 9 | Datos crudos locales, sin archivos actuales |
| `PROCESSED_DATA_DIR` | `data/processed/` | `src/config.py`, linea 10 | Datos procesados locales, sin archivos actuales |
| `EXTERNAL_DATA_DIR` | `data/external/` | `src/config.py`, linea 11 | Datos externos locales, sin archivos actuales |
| `OUTPUTS_DIR` | `outputs/` | `src/config.py`, linea 13 | Raiz de artefactos |
| `FIGURES_DIR` | `outputs/figures/` | `src/config.py`, linea 14 | Figuras |
| `METRICS_DIR` | `outputs/metrics/` | `src/config.py`, linea 15 | JSON de metricas/reportes |
| `MODELS_DIR` | `outputs/models/` | `src/config.py`, linea 16 | Checkpoints Keras |
| `LOGS_DIR` | `outputs/logs/` | `src/config.py`, linea 17 | Logs CSV |
| `PREDICTIONS_DIR` | `outputs/predictions/` | `src/config.py`, linea 18 | Errores CSV |
| `REPORTS_TABLES_DIR` | `reports/tables/` | `src/config.py`, linea 21 | Tablas para TFM |
| `RANDOM_SEED` | 42 | `src/config.py`, linea 29 | Reproducibilidad |
| `NUM_CLASSES` | 10 | `src/config.py`, linea 30 | Salida de modelos y one-hot |
| `CLASS_NAMES` | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck | `src/config.py`, lineas 31-42 | Etiquetas y graficas |
| `INPUT_SHAPE` | `(32, 32, 3)` | `src/config.py`, lineas 44-47 | Entrada base de modelos |
| `DEFAULT_BATCH_SIZE` | 64 | `src/config.py`, linea 52 | Entrenamiento |
| `DEFAULT_EPOCHS` | 20 | `src/config.py`, linea 53 | Entrenamiento |
| `DEFAULT_LEARNING_RATE` | 0.001 | `src/config.py`, linea 54 | CNN base y mejorada |
| `VALIDATION_SPLIT` | 0.1 | `src/config.py`, linea 55 | Division train/validacion |
| Nombres de modelos CLI | `baseline_cnn`, `improved_cnn`, `transfer_model` | `main.py`, lineas 90-95 | Seleccion por argumento |

No se encontraron archivos de configuracion adicionales dentro de `configs/`.

# 7. Flujo de ejecucion de main.py

Comandos verificables:

```bash
python main.py --model baseline_cnn
python main.py --model improved_cnn
python main.py --model transfer_model
```

El argumento `--model` acepta exactamente esos tres valores (`main.py`, lineas 88-99). La funcion `get_model` selecciona el constructor correspondiente (`main.py`, lineas 20-30).

| Fase | Archivo que interviene | Operacion | Entrada | Salida |
|---|---|---|---|---|
| Semilla | `src/seeds.py` | Fija semillas de Python, NumPy y TensorFlow | Valor 42 | Estado reproducible parcial |
| Carga | `src/data/data_loader.py` | Carga CIFAR-10 desde Keras | Dataset publico CIFAR-10 | `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, `y_test` |
| Division | `src/data/data_loader.py` | `train_test_split` con `test_size=0.1`, `stratify=y_train_full` | 50,000 imagenes train originales | 45,000 train, 5,000 validacion |
| Preprocesamiento | `src/data/preprocessing.py` | Normaliza imagenes y aplica one-hot | Imagenes y etiquetas | Tensores preprocesados |
| Seleccion modelo | `main.py`, `src/models/*.py` | Construye modelo segun `--model` | Nombre de modelo | Modelo Keras compilado |
| Entrenamiento | `src/training/trainer.py` | Ejecuta `model.fit` | Train/validacion | `history` |
| Checkpoint/logs | `src/training/callbacks.py` | Guarda mejor modelo, CSV log, early stopping, reduce LR | Modelo y validacion | `.keras` y `.csv` |
| Curvas | `src/visualization/training_plots.py` | Grafica accuracy/loss | `history` | PNG en `outputs/figures/training/` |
| Evaluacion | `main.py` | Predice sobre test | `x_test_p` | `y_pred` |
| Metricas | `src/evaluation/metrics.py` | Calcula metricas globales | `y_test_p`, `y_pred` | JSON en `outputs/metrics/` |
| Matriz confusion | `src/evaluation/confusion_matrix.py` | Genera heatmap | `y_test_p`, `y_pred` | PNG en `outputs/figures/evaluation/` |
| Reporte por clase | `src/evaluation/reports.py` | Genera classification report | `y_test_p`, `y_pred` | JSON y CSV |
| Errores | `src/evaluation/error_analysis.py` | Guarda errores y figura de muestras | Test y predicciones | CSV y PNG |

Flujo textual:

```text
Dataset
↓
Carga
↓
Division
↓
Preprocesamiento
↓
Seleccion del modelo
↓
Entrenamiento
↓
Checkpoint
↓
Evaluacion
↓
Metricas
↓
Predicciones
↓
Figuras
↓
Reportes
```

# 8. Datos analizados

## Dataset original

| Aspecto | Valor verificable | Evidencia |
|---|---|---|
| Dataset | CIFAR-10 | README, lineas 15-24; `src/data/data_loader.py`, linea 2 |
| Total imagenes | 60,000 | README, linea 21 |
| Resolucion | 32 x 32 pixeles | README, linea 22; `src/config.py`, lineas 44-47 |
| Canales | 3 | `src/config.py`, linea 46 |
| Clases | 10 clases balanceadas | README, lineas 21-24; `src/config.py`, lineas 30-42 |
| Split original Keras | Train/test de CIFAR-10 | `src/data/data_loader.py`, linea 19 |

## Dataset usado en el proyecto

| Fase | Datos utilizados | Transformacion | Evidencia |
|---|---|---|---|
| Entrenamiento | 45,000 imagenes | Division estratificada del train original | `src/data/data_loader.py`, lineas 24-30; `reports/tables/cifar10_class_distribution.csv` |
| Validacion | 5,000 imagenes | 10% del train original | `src/config.py`, linea 55 |
| Prueba | 10,000 imagenes | Test original CIFAR-10 | Reportes por clase con support total 10,000 |
| Normalizacion | CNNs e inicio del flujo | `x.astype("float32") / 255.0` | `src/data/preprocessing.py`, lineas 7-12 |
| Etiquetas | Todos los modelos en `main.py` | One-hot encoding | `src/data/preprocessing.py`, lineas 15-20 |
| Estratificacion | Train/validacion | `stratify=y_train_full` | `src/data/data_loader.py`, linea 29 |
| Augmentation | `improved_cnn` | RandomFlip, RandomRotation, RandomZoom, RandomTranslation | `src/data/augmentation.py`, lineas 12-18; `src/models/improved_cnn.py`, lineas 18-24 |
| MobileNetV2 | `transfer_model` | Redimensiona a 96x96 y aplica `preprocess_input` tras multiplicar por 255 | `src/models/transfer_model.py`, lineas 16-18 |

Distribucion verificada por clase: cada clase tiene 4,500 imagenes de entrenamiento usadas por el proyecto y 1,000 de prueba en `reports/tables/cifar10_class_distribution.csv`.

# 9. Trazabilidad de los artefactos

| Proceso | Codigo responsable | Archivo generado | Carpeta |
|---|---|---|---|
| Entrenamiento CNN base | `src/training/trainer.py`; `src/training/callbacks.py` | `baseline_cnn_best.keras` | `outputs/models/` |
| Entrenamiento CNN mejorada | `src/training/trainer.py`; `src/training/callbacks.py` | `improved_cnn_best.keras` | `outputs/models/` |
| Entrenamiento transfer | `src/training/trainer.py`; `src/training/callbacks.py` | `transfer_model_best.keras` | `outputs/models/` |
| Logs entrenamiento | `src/training/callbacks.py` | `*_training_log.csv` | `outputs/logs/` |
| Metricas globales | `src/evaluation/metrics.py` | `*_metrics.json` | `outputs/metrics/` |
| Reporte por clase JSON | `src/evaluation/reports.py` | `*_classification_report.json` | `outputs/metrics/` |
| Reporte por clase CSV | `src/evaluation/reports.py` | `*_classification_report.csv` | `reports/tables/` |
| Predicciones erroneas | `src/evaluation/error_analysis.py` | `*_misclassified_samples.csv` | `outputs/predictions/` |
| Curvas entrenamiento | `src/visualization/training_plots.py` | `*_accuracy.png`, `*_loss.png` | `outputs/figures/training/` |
| Matrices de confusion | `src/evaluation/confusion_matrix.py` | `*_confusion_matrix.png` | `outputs/figures/evaluation/` |
| Imagenes mal clasificadas | `src/evaluation/error_analysis.py` | `*_misclassified_samples.png` | `outputs/figures/evaluation/` |
| Comparacion de modelos | `src/evaluation/comparison.py` | `model_comparison.csv`, `model_comparison.png` | `reports/tables/`, `outputs/figures/evaluation/` |
| Grad-CAM | `generate_gradcam.py`; `src/interpretability/gradcam.py` | `improved_cnn_gradcam_*.png` | `outputs/figures/gradcam/` |
| EDA | `generate_eda_figures.py` | `cifar10_class_distribution.csv`, `cifar10_class_distribution.png`, `cifar10_sample_images.png` | `reports/tables/`, `outputs/figures/eda/` |
| Evidencias complementarias transfer | `generate_evaluation_evidence.py` | `transfer_model_confused_classes.csv`, `transfer_model_confused_classes.png` | `reports/tables/`, `outputs/figures/evaluation/` |

# 10. Artefactos generados

| Tipo | Archivo | Modelo | Formato | Finalidad |
|---|---|---|---|---|
| Checkpoint | `outputs/models/baseline_cnn_best.keras` | baseline_cnn | `.keras` | Modelo guardado |
| Checkpoint | `outputs/models/improved_cnn_best.keras` | improved_cnn | `.keras` | Modelo guardado |
| Checkpoint | `outputs/models/transfer_model_best.keras` | transfer_model | `.keras` | Modelo guardado |
| Metrica | `outputs/metrics/baseline_cnn_metrics.json` | baseline_cnn | JSON | Metricas globales |
| Metrica | `outputs/metrics/improved_cnn_metrics.json` | improved_cnn | JSON | Metricas globales |
| Metrica | `outputs/metrics/transfer_model_metrics.json` | transfer_model | JSON | Metricas globales |
| Log | `outputs/logs/baseline_cnn_training_log.csv` | baseline_cnn | CSV | Historial de entrenamiento |
| Log | `outputs/logs/improved_cnn_training_log.csv` | improved_cnn | CSV | Historial de entrenamiento |
| Log | `outputs/logs/transfer_model_training_log.csv` | transfer_model | CSV | Historial de entrenamiento |
| Predicciones | `outputs/predictions/*_misclassified_samples.csv` | tres modelos | CSV | Errores de clasificacion |
| Reporte | `reports/tables/*_classification_report.csv` | tres modelos | CSV | Reporte por clase |
| Comparacion | `reports/tables/model_comparison.csv` | tres modelos | CSV | Tabla comparativa |
| EDA | `reports/tables/cifar10_class_distribution.csv` | CIFAR-10 | CSV | Distribucion por clase |
| Curvas | `outputs/figures/training/*_accuracy.png`, `*_loss.png` | tres modelos | PNG | Evolucion entrenamiento/validacion |
| Evaluacion | `outputs/figures/evaluation/*_confusion_matrix.png` | tres modelos | PNG | Matriz de confusion visual |
| Evaluacion | `outputs/figures/evaluation/*_misclassified_samples.png` | tres modelos | PNG | Ejemplos mal clasificados |
| Comparacion | `outputs/figures/evaluation/model_comparison.png` | tres modelos | PNG | Grafica comparativa |
| Grad-CAM | `outputs/figures/gradcam/improved_cnn_gradcam_*.png` | improved_cnn | PNG | Interpretabilidad visual |
| Evidencia transfer | `reports/tables/transfer_model_confused_classes.csv` | transfer_model | CSV | Clases confundidas |

# 11. Reproducibilidad

## Reproducibilidad disponible

- Dataset publico: CIFAR-10 se carga desde `tensorflow.keras.datasets.cifar10` (`src/data/data_loader.py`, linea 2).
- Dependencias versionadas en `requirements.txt`.
- Semilla central 42 (`src/config.py`, linea 29) y funcion `set_seed` para Python, NumPy y TensorFlow (`src/seeds.py`, lineas 7-25).
- Configuracion central de rutas, clases, input shape, batch size, epochs, learning rate y validation split (`src/config.py`).
- Comandos de ejecucion documentados en README (`README.md`, lineas 148-192).
- Checkpoints `.keras` existentes en `outputs/models/`.
- Logs CSV de entrenamiento en `outputs/logs/`.
- Metricas, reportes, predicciones y figuras conservadas en `outputs/` y `reports/`.
- Arquitectura modular que separa datos, modelos, entrenamiento, evaluacion, visualizacion e interpretabilidad.

## Informacion que falta para reproducibilidad completa

| Elemento faltante | Estado |
|---|---|
| Version exacta de Python | No encontrado en el repositorio |
| Sistema operativo de ejecucion original | No encontrado en el repositorio |
| Hardware | No encontrado en el repositorio |
| GPU | No encontrado en el repositorio |
| CUDA/cuDNN | No encontrado en el repositorio |
| Tiempos de entrenamiento | No encontrado en logs |
| Entorno virtual exportado | No encontrado |
| Dockerfile o contenedor | No encontrado |
| Multiples semillas o ejecuciones repetidas | No encontrado |
| Tests automatizados | Carpeta `tests/` vacia |

# 12. Procedimiento de reproduccion

Sin ejecutar comandos, la secuencia verificable para reproducir el proyecto seria:

1. Clonar el repositorio:

```bash
git clone https://github.com/WiloCh/tfm-deep-learning-image-classification.git
```

2. Entrar en la carpeta del proyecto.

3. Crear y activar un entorno Python. El comando exacto no esta documentado en el repositorio.

4. Instalar dependencias:

```bash
pip install -r requirements.txt
```

5. Ejecutar la CNN base:

```bash
python main.py --model baseline_cnn
```

6. Ejecutar la CNN mejorada:

```bash
python main.py --model improved_cnn
```

7. Ejecutar el modelo de transferencia:

```bash
python main.py --model transfer_model
```

8. Generar figuras EDA:

```bash
python generate_eda_figures.py
```

9. Generar Grad-CAM:

```bash
python generate_gradcam.py
```

10. Generar evidencias complementarias de evaluacion, si se requieren:

```bash
python generate_evaluation_evidence.py
```

11. Localizar resultados en `outputs/` y tablas en `reports/tables/`.

Advertencia: volver a ejecutar `main.py` reentrena modelos y podria generar resultados diferentes si el entorno, hardware o versiones reales difieren.

# 13. Diferencias entre codigo, README y documento

| Elemento | Codigo | README | Resultado real | Accion recomendada |
|---|---|---|---|---|
| Fine-tuning parcial | `base_model.trainable = False` en `src/models/transfer_model.py`, linea 26 | Menciona "Fine-tuning parcial", linea 78 | No se observa descongelacion de capas | Cambiar redaccion a "extraccion de caracteristicas con MobileNetV2 congelado" o implementar fine-tuning real |
| Grad-CAM | `generate_gradcam.py` usa `model_name = "improved_cnn"` | Se presenta de forma general | Solo hay PNG `improved_cnn_gradcam_*.png` | Aclarar que las Grad-CAM disponibles corresponden a `improved_cnn` |
| Estructura | README muestra estructura general | Coincide a nivel general | Existen subcarpetas de `data/`, `outputs/` y `reports/` no detalladas en README | Ampliar estructura del Capitulo 6 con arbol real |
| Configs | `CONFIGS_DIR` definido | Lista `configs/` | Carpeta vacia | No atribuir configuraciones externas inexistentes |
| Tests | Carpeta existe | Se lista `tests/` | Carpeta vacia | No afirmar cobertura de pruebas |
| Modelos CLI | `choices=["baseline_cnn", "improved_cnn", "transfer_model"]` | Comandos coinciden | Coherente | Mantener |
| Outputs | Codigo guarda en `outputs/` y `reports/tables/` | README menciona outputs/reports de forma general | Artefactos reales existentes | Describir trazabilidad en Capitulo 6 |
| Encoding README | Texto muestra caracteres corruptos en consola | No aplica | Aparente problema de codificacion al leer | Revisar codificacion antes de version final del documento |

# 14. Propuesta del nuevo Capitulo 6

| Apartado | Informacion que debe aparecer | Tabla util | Figura util | Que NO repetir del Capitulo 5 |
|---|---|---|---|---|
| 6.1 Organizacion del codigo fuente | Arbol real, carpetas y scripts | Responsabilidad de directorios | Diagrama de repositorio | Arquitecturas internas de modelos en detalle |
| 6.2 Tecnologias y dependencias | Librerias y versiones de `requirements.txt` | Tecnologias y versiones | No necesaria | Discusion de rendimiento |
| 6.3 Arquitectura modular del proyecto | Modulos `src/data`, `src/models`, `src/training`, etc. | Responsabilidad de modulos | Diagrama modular | Comparacion de metricas |
| 6.4 Flujo de ejecucion | Pasos de `main.py` y scripts auxiliares | Fases entrada/salida | Pipeline de ejecucion | Curvas y matrices |
| 6.5 Datos analizados | CIFAR-10, split, normalizacion, one-hot, augmentation | Dataset original vs usado | Flujo de datos | Analisis por clase |
| 6.6 Configuracion experimental | Constantes de `src/config.py` | Parametros de configuracion | No necesaria | Resultados |
| 6.7 Gestion y trazabilidad de resultados | Codigo -> artefacto -> carpeta | Trazabilidad de artefactos | Diagrama trazabilidad | Interpretacion de metricas |
| 6.8 Reproducibilidad del codigo | Dependencias, seed, comandos, artefactos | Reproducibilidad disponible | No necesaria | Discusion cientifica de modelos |
| 6.9 Limitaciones de reproducibilidad | Faltantes: Python, hardware, GPU, Docker, tiempos | Informacion faltante | No necesaria | Limitaciones de rendimiento ya tratadas |
| 6.10 Sintesis del capitulo | Cierre sobre codigo, datos y trazabilidad | No necesaria | No necesaria | Conclusiones de resultados del Capitulo 5 |

# 15. Figuras recomendadas

1. Diagrama de arquitectura del repositorio. Deberia mostrar raiz del proyecto, scripts principales, `src/`, `outputs/`, `reports/` y carpetas vacias o reservadas. No debe incluir resultados numericos.

2. Diagrama del pipeline de ejecucion. Deberia seguir el flujo `Dataset -> Carga -> Division -> Preprocesamiento -> Modelo -> Entrenamiento -> Evaluacion -> Artefactos`.

3. Diagrama de trazabilidad. Deberia representar `codigo -> modelo -> metricas -> figuras -> tablas`, indicando carpetas: `src/`, `outputs/models/`, `outputs/metrics/`, `outputs/figures/`, `reports/tables/`.

No se deben generar todavia estas figuras; solo se recomienda su contenido.

# 16. Tablas recomendadas

| Tabla | Valor para el Capitulo 6 |
|---|---|
| Tecnologias y versiones | Alta: permite documentar entorno dependiente del codigo |
| Responsabilidad de directorios | Alta: explica organizacion sin repetir resultados |
| Responsabilidad de modulos | Alta: muestra arquitectura modular |
| Configuracion experimental | Alta: centraliza parametros verificables |
| Artefactos generados | Alta: conecta codigo con entregables |
| Trazabilidad del experimento | Muy alta: seccion clave del capitulo |
| Informacion faltante de reproducibilidad | Alta: delimita afirmaciones defendibles |

# 17. Informacion faltante

## Indispensable

| Informacion | Motivo |
|---|---|
| Version exacta de Python usada en ejecuciones originales | Necesaria para reproducibilidad completa |
| Hardware/GPU/CUDA/cuDNN | Necesario si se quiere replicar tiempos o comportamiento exacto |
| Entorno virtual o instrucciones de version de Python | Facilita instalacion reproducible |

## Recomendable

| Informacion | Motivo |
|---|---|
| Dockerfile o archivo de entorno | Mejora portabilidad |
| Tiempos de entrenamiento | Ayuda a dimensionar recursos |
| Matriz de dependencias entre modulos | Facilita mantenimiento |
| Tests automatizados | Aumenta confiabilidad del codigo |

## Opcional

| Informacion | Motivo |
|---|---|
| Notebooks exploratorios | Podrian apoyar comprension, pero no son necesarios si el codigo modular esta documentado |
| Multiples seeds | Relevante para robustez experimental, pero pertenece mas al diseno experimental que a la estructura del codigo |
| Diagrama generado automaticamente | Util para presentacion, no indispensable |

# 18. Conclusion

Existe informacion suficiente para redactar el Capitulo 6 si se enfoca correctamente en codigo fuente, datos analizados, trazabilidad y reproducibilidad. Son totalmente verificables la estructura modular, los scripts de entrada, las dependencias de `requirements.txt`, la configuracion central de `src/config.py`, el flujo de `main.py`, el uso de CIFAR-10 y la ubicacion de los artefactos generados.

No deben afirmarse elementos no documentados en el repositorio, como version exacta de Python, sistema operativo original, GPU, CUDA, cuDNN, Docker, tiempos de entrenamiento o cobertura de tests. Tampoco debe afirmarse fine-tuning parcial de MobileNetV2, porque el codigo congela el modelo base.

Del Capitulo 6 anterior, si existia, deberian mantenerse las partes sobre estructura del repositorio, dependencias, configuracion, flujo de ejecucion, datos usados y artefactos generados. Deben eliminarse o reducirse las secciones que repitan arquitectura detallada de modelos, metricas, comparacion de rendimiento, curvas, matrices de confusion, Grad-CAM y discusion de resultados, porque esos contenidos corresponden al Capitulo 5.
