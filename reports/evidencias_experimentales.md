# 1. Resumen ejecutivo

Este informe tecnico recopila y verifica la evidencia experimental disponible en el repositorio `C:\Users\William Chugchilan\OneDrive\Documentos\TFM\Codigo\tfm-image-classification`. Se reviso el flujo implementado en `main.py`, los modulos de datos, modelos, entrenamiento, evaluacion, visualizacion e interpretabilidad, asi como los artefactos en `outputs/` y `reports/`.

Hecho verificado: el proyecto compara tres modelos de clasificacion de imagenes sobre CIFAR-10: `baseline_cnn`, `improved_cnn` y `transfer_model`. El flujo principal carga CIFAR-10, divide entrenamiento/validacion, normaliza imagenes, codifica etiquetas, construye el modelo seleccionado, entrena, evalua sobre prueba y guarda metricas, reportes, figuras y errores (`main.py`, lineas 33-85).

Hecho verificado: las metricas globales disponibles indican que `transfer_model` obtuvo el mayor rendimiento de prueba entre los tres modelos: accuracy 0.878800, precision macro 0.878775, recall macro 0.878800 y F1 macro 0.878532 (`outputs/metrics/transfer_model_metrics.json`, lineas 1-7; `reports/tables/model_comparison.csv`, filas 1-4). Le siguen `improved_cnn` con accuracy 0.775900 y F1 macro 0.770678, y `baseline_cnn` con accuracy 0.735200 y F1 macro 0.735830.

Interpretacion respaldada por resultados: la mejora de `improved_cnn` frente a `baseline_cnn` es compatible con el uso de mayor profundidad, batch normalization, dropout y data augmentation (`src/models/improved_cnn.py`, lineas 18-55), aunque no se puede atribuir causalmente a un componente aislado. La diferencia de `transfer_model` es compatible con MobileNetV2 preentrenado en ImageNet, redimensionamiento a 96x96, `preprocess_input`, GlobalAveragePooling y una cabeza densa ligera (`src/models/transfer_model.py`, lineas 14-43).

Informacion faltante para redactar con maxima solidez: no se encontraron tiempos de entrenamiento, hardware, version de Python ejecutada, loss de prueba, multiples ejecuciones por modelo, medias/desviaciones estandar ni matrices de confusion en formato numerico para todos los modelos. Tambien existe una contradiccion entre el CSV de errores de `transfer_model` generado en abril y la tabla `transfer_model_confused_classes.csv` generada en junio; por tanto, no deben mezclarse esas evidencias sin aclaracion.

# 2. Inventario de evidencias disponibles

| Tipo de evidencia | Modelo | Ruta exacta | Formato | Contenido | Estado | Utilidad documental |
|---|---|---|---|---|---|---|
| Modelo Keras | baseline_cnn | `outputs/models/baseline_cnn_best.keras` | Keras | Mejor checkpoint por `val_accuracy` | Encontrado | Evidencia del modelo entrenado |
| Modelo Keras | improved_cnn | `outputs/models/improved_cnn_best.keras` | Keras | Mejor checkpoint por `val_accuracy` | Encontrado | Evidencia del modelo entrenado |
| Modelo Keras | transfer_model | `outputs/models/transfer_model_best.keras` | Keras | Mejor checkpoint por `val_accuracy` | Encontrado | Evidencia del modelo entrenado |
| Metricas globales | baseline_cnn | `outputs/metrics/baseline_cnn_metrics.json` | JSON | Accuracy, precision macro, recall macro, F1 macro | Encontrado | Comparacion cuantitativa |
| Metricas globales | improved_cnn | `outputs/metrics/improved_cnn_metrics.json` | JSON | Accuracy, precision macro, recall macro, F1 macro | Encontrado | Comparacion cuantitativa |
| Metricas globales | transfer_model | `outputs/metrics/transfer_model_metrics.json` | JSON | Accuracy, precision macro, recall macro, F1 macro | Encontrado | Comparacion cuantitativa |
| Predicciones erroneas | baseline_cnn | `outputs/predictions/baseline_cnn_misclassified_samples.csv` | CSV | Indice, clase real y clase predicha | Encontrado | Analisis de errores |
| Predicciones erroneas | improved_cnn | `outputs/predictions/improved_cnn_misclassified_samples.csv` | CSV | Indice, clase real y clase predicha | Encontrado | Analisis de errores |
| Predicciones erroneas | transfer_model | `outputs/predictions/transfer_model_misclassified_samples.csv` | CSV | Indice, clase real y clase predicha | Encontrado | Analisis de errores |
| Curvas entrenamiento | baseline_cnn | `outputs/figures/training/baseline_cnn_accuracy.png`, `outputs/figures/training/baseline_cnn_loss.png` | PNG | Accuracy/loss entrenamiento-validacion | Encontrado | Estabilidad y sobreajuste |
| Curvas entrenamiento | improved_cnn | `outputs/figures/training/improved_cnn_accuracy.png`, `outputs/figures/training/improved_cnn_loss.png` | PNG | Accuracy/loss entrenamiento-validacion | Encontrado | Estabilidad y sobreajuste |
| Curvas entrenamiento | transfer_model | `outputs/figures/training/transfer_model_accuracy.png`, `outputs/figures/training/transfer_model_loss.png` | PNG | Accuracy/loss entrenamiento-validacion | Encontrado | Estabilidad y sobreajuste |
| Matriz de confusion | baseline_cnn | `outputs/figures/evaluation/baseline_cnn_confusion_matrix.png` | PNG | Matriz visual | Encontrado | Analisis por clase visual |
| Matriz de confusion | improved_cnn | `outputs/figures/evaluation/improved_cnn_confusion_matrix.png` | PNG | Matriz visual | Encontrado | Analisis por clase visual |
| Matriz de confusion | transfer_model | `outputs/figures/evaluation/transfer_model_confusion_matrix.png` | PNG | Matriz visual, regenerada en junio | Encontrado con cautela | Analisis por clase visual |
| Reporte por clase | baseline_cnn | `reports/tables/baseline_cnn_classification_report.csv` | CSV | Precision, recall, F1, support | Encontrado | Analisis por clase |
| Reporte por clase | improved_cnn | `reports/tables/improved_cnn_classification_report.csv` | CSV | Precision, recall, F1, support | Encontrado | Analisis por clase |
| Reporte por clase | transfer_model | `reports/tables/transfer_model_classification_report.csv` | CSV | Precision, recall, F1, support | Encontrado | Analisis por clase |
| Imagenes mal clasificadas | baseline_cnn | `outputs/figures/evaluation/baseline_cnn_misclassified_samples.png` | PNG | 16 errores iniciales | Encontrado | Analisis cualitativo |
| Imagenes mal clasificadas | improved_cnn | `outputs/figures/evaluation/improved_cnn_misclassified_samples.png` | PNG | 16 errores iniciales | Encontrado | Analisis cualitativo |
| Imagenes mal clasificadas | transfer_model | `outputs/figures/evaluation/transfer_model_misclassified_samples.png` | PNG | 16 errores, regenerada en junio | Encontrado con cautela | Analisis cualitativo |
| Clases confundidas | transfer_model | `reports/tables/transfer_model_confused_classes.csv` | CSV | Top 10 confusiones y posible causa | Encontrado con contradiccion | Discusion de errores |
| Figura clases confundidas | transfer_model | `outputs/figures/evaluation/transfer_model_confused_classes.png` | PNG | Barras top confusiones | Encontrado | Visualizacion de errores |
| Figuras EDA | CIFAR-10 | `outputs/figures/eda/cifar10_class_distribution.png`, `outputs/figures/eda/cifar10_sample_images.png` | PNG | Distribucion y muestras | Encontrado | Preparacion de datos |
| Grad-CAM | improved_cnn | `outputs/figures/gradcam/improved_cnn_gradcam_*.png` | PNG | 6 visualizaciones | Encontrado | Interpretabilidad |
| Logs | tres modelos | `outputs/logs/*_training_log.csv` | CSV | Epoch, accuracy, loss, val_accuracy, val_loss, learning_rate | Encontrado | Entrenamiento y callbacks |
| Tabla comparativa | tres modelos | `reports/tables/model_comparison.csv` | CSV | Metricas globales ordenadas por F1 | Encontrado | Comparacion principal |
| Configuracion | global | `src/config.py` | Python | Rutas, clases, semilla, batch, epochs, LR | Encontrado | Reproducibilidad |
| Tests | global | `tests/` | Carpeta | Sin archivos visibles | No encontrado | No aporta evidencia |
| Notebooks | global | `notebooks/` | Carpeta | Sin archivos visibles | No encontrado | No aporta evidencia |
| Reports figures | global | `reports/figures/` | Carpeta | Sin archivos visibles | No encontrado | No aporta evidencia |

# 3. Entorno tecnologico y reproducibilidad

| Elemento | Valor verificado | Fuente |
|---|---|---|
| Python | No encontrado; `python.exe` no fue accesible desde la sesion de auditoria | Intento de inspeccion local fallido |
| TensorFlow | 2.21.0 | `requirements.txt`, linea 1 |
| Keras | Incluido via TensorFlow/Keras; version separada no encontrada | `requirements.txt`, linea 1 |
| NumPy | 2.0.2 | `requirements.txt`, linea 2 |
| pandas | 2.2.3 | `requirements.txt`, linea 3 |
| scikit-learn | 1.5.2 | `requirements.txt`, linea 5 |
| matplotlib | 3.9.2 | `requirements.txt`, linea 4 |
| seaborn | 0.13.2 | `requirements.txt`, linea 6 |
| Sistema operativo | No encontrado en archivos del proyecto | No encontrado |
| CPU/GPU | No encontrado | No encontrado |
| Semilla | 42 | `src/config.py`, linea 29; `main.py`, linea 34; `src/seeds.py`, lineas 7-25 |
| Commit Git | `79d36ab94e8cd63c469ce986195f7fb037d1cd84` | `git rev-parse HEAD` |
| Estado Git | Hay cambios previos no relacionados o sin seguimiento | `git status --short`: README modificado, figuras de transfer modificadas, `generate_evaluation_evidence.py` y evidencias de clases confundidas sin seguimiento |

Comandos documentados en README y verificados por codigo:

| Comando | Que hace segun codigo | Fuente |
|---|---|---|
| `python main.py --model baseline_cnn` | Ejecuta carga, preprocesamiento, entrenamiento, evaluacion y guardado de evidencias para `baseline_cnn` | `main.py`, lineas 88-99; README, lineas 150-154 |
| `python main.py --model improved_cnn` | Igual flujo para `improved_cnn` | `main.py`, lineas 20-28 y 88-99; README, lineas 156-160 |
| `python main.py --model transfer_model` | Igual flujo para `transfer_model` | `main.py`, lineas 20-28 y 88-99; README, lineas 162-166 |
| `python generate_eda_figures.py` | Carga CIFAR-10 y genera distribucion de clases y muestras por clase | `generate_eda_figures.py`, lineas 70-82 |
| `python generate_gradcam.py` | Carga `improved_cnn_best.keras`, genera predicciones sobre test y 6 Grad-CAM | `generate_gradcam.py`, lineas 11-50 |
| `python generate_evaluation_evidence.py` | Carga `transfer_model_best.keras`, predice test y sobrescribe matriz, errores visuales, top confusiones CSV/PNG | `generate_evaluation_evidence.py`, lineas 151-198 |

# 4. Preparacion de CIFAR-10

| Aspecto | Valor | Tipo | Evidencia |
|---|---:|---|---|
| Imagenes de entrenamiento usadas tras split | 45,000 | Calculo derivado: 50,000 x 0.90 | `src/data/data_loader.py`, lineas 19-30; `src/config.py`, linea 55 |
| Imagenes de validacion | 5,000 | Calculo derivado: 50,000 x 0.10 | `src/data/data_loader.py`, lineas 24-30 |
| Imagenes de prueba | 10,000 | Hecho verificado por CIFAR-10 y supports | `reports/tables/*_classification_report.csv`, fila `macro avg`, support 10000 |
| Dimensiones originales | 32 x 32 | Hecho verificado | README, lineas 21-23; `src/config.py`, lineas 44-47 |
| Canales | 3 | Hecho verificado | `src/config.py`, lineas 44-47 |
| Clases | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck | Hecho verificado | `src/config.py`, lineas 31-42 |
| Distribucion train/test por clase | 4500 train, 1000 test, 5500 total | Hecho verificado | `reports/tables/cifar10_class_distribution.csv`, filas 2-11 |
| Normalizacion | `x.astype("float32") / 255.0` | Hecho verificado | `src/data/preprocessing.py`, lineas 7-12 |
| Redimensionamiento CNNs | No encontrado | Hecho verificado | `baseline_cnn` e `improved_cnn` usan `INPUT_SHAPE` 32x32x3; `src/models/*.py` |
| Redimensionamiento MobileNetV2 | 96 x 96 | Hecho verificado | `src/models/transfer_model.py`, lineas 14-21 |
| Codificacion etiquetas | One-hot con `to_categorical(y, NUM_CLASSES)` | Hecho verificado | `src/data/preprocessing.py`, lineas 15-20 |
| Division train/validacion | `train_test_split`, `test_size=VALIDATION_SPLIT` | Hecho verificado | `src/data/data_loader.py`, lineas 24-30 |
| Estratificacion | `stratify=y_train_full` | Hecho verificado | `src/data/data_loader.py`, linea 29 |
| Semilla | 42 | Hecho verificado | `src/config.py`, linea 29; `src/seeds.py`, lineas 7-25 |
| Aumento de datos | RandomFlip horizontal, RandomRotation 0.08, RandomZoom 0.10, RandomTranslation 0.08 | Hecho verificado | `src/data/augmentation.py`, lineas 12-18 |
| Uso real de augmentation | Solo `improved_cnn` lo incorpora directamente | Hecho verificado | `src/models/improved_cnn.py`, lineas 18-24 |
| Diferencia MobileNetV2 | Recibe entrada normalizada, la multiplica por 255 y aplica `mobilenet_v2.preprocess_input` | Hecho verificado | `src/models/transfer_model.py`, lineas 16-18 |
| Riesgo de fuga de datos | No se observa fuga explicita: test solo se usa despues del entrenamiento en `main.py` | Interpretacion respaldada por codigo | `main.py`, lineas 50-83 |

# 5. Arquitectura y configuracion de los modelos

## 5.1 CNN base

Hecho verificado: `baseline_cnn` usa entrada 32x32x3, tres bloques Conv2D + MaxPooling, Flatten, Dense(128) y salida Dense(10, softmax). Usa Adam con learning rate 0.001, perdida `categorical_crossentropy` y metrica `accuracy` (`src/models/baseline_cnn.py`, lineas 13-37).

Calculo derivado: con las formulas de parametros de Conv2D y Dense, el total aproximado es 356,810 parametros, todos entrenables. No se verifico mediante carga del `.keras` porque `python.exe` no estuvo accesible.

## 5.2 CNN mejorada

Hecho verificado: `improved_cnn` incorpora data augmentation, dos convoluciones por bloque, BatchNormalization, MaxPooling, Dropout 0.25/0.30/0.40, Dense(256), BatchNormalization, Dropout 0.50 y salida softmax de 10 clases (`src/models/improved_cnn.py`, lineas 18-64).

Calculo derivado: usando la arquitectura del codigo, el total aproximado es 816,938 parametros: 815,530 entrenables y 1,408 no entrenables asociados a medias/varianzas de BatchNormalization. No se verifico mediante carga del `.keras`.

## 5.3 MobileNetV2

Hecho verificado: `transfer_model` redimensiona 32x32 a 96x96, aplica `mobilenet_v2.preprocess_input`, carga MobileNetV2 con `include_top=False` y `weights="imagenet"`, congela `base_model.trainable = False`, agrega GlobalAveragePooling2D, Dropout 0.30, Dense(128), Dropout 0.30 y Dense(10, softmax). Usa Adam con learning rate 0.0005 (`src/models/transfer_model.py`, lineas 14-43).

Informacion no verificable: no se encontro una estrategia de fine-tuning parcial real; el codigo congela todo MobileNetV2 y no descongela capas. Por tanto, la mencion de "Fine-tuning parcial" en README es contradictoria o no demostrada (`README.md`, lineas 73-78 frente a `src/models/transfer_model.py`, linea 26).

| Caracteristica | baseline_cnn | improved_cnn | transfer_model |
|---|---|---|---|
| Entrada | 32x32x3 | 32x32x3 | 32x32x3, luego 96x96 |
| Augmentation | No | Si | No encontrado |
| Convoluciones propias | 3 | 6 | MobileNetV2 preentrenado |
| Batch normalization | No | Si | Presente dentro de MobileNetV2, pero base congelada; no detallado en repo |
| Dropout | No | 0.25, 0.30, 0.40, 0.50 | 0.30 y 0.30 |
| GlobalAveragePooling | No | No | Si |
| Dense final | 10 softmax | 10 softmax | 10 softmax |
| Optimizador | Adam | Adam | Adam |
| LR inicial | 0.001 | 0.001 | 0.0005 |
| Loss | categorical_crossentropy | categorical_crossentropy | categorical_crossentropy |
| Batch size | 64 | 64 | 64 |
| Epocas maximas | 20 | 20 | 20 |
| Callbacks | ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger | Igual | Igual |
| Criterio checkpoint | max `val_accuracy` | max `val_accuracy` | max `val_accuracy` |
| Parametros totales | 356,810 derivado | 816,938 derivado | No encontrado verificable |
| Parametros entrenables | 356,810 derivado | 815,530 derivado | No encontrado verificable |
| Parametros congelados | 0 derivado | 1,408 no entrenables derivado | MobileNetV2 congelado, conteo no encontrado |

# 6. Resultados cuantitativos verificados

| Modelo | Accuracy prueba | Precision macro | Recall macro | F1 macro | Loss prueba | Train acc final | Val acc final | Train loss final | Val loss final | Mejor epoca val_acc | Epocas ejecutadas | Tiempo | Parametros totales | Tamano modelo |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| baseline_cnn | 0.735200 | 0.745124 | 0.735200 | 0.735830 | No encontrado | 0.944133 | 0.757000 | 0.181309 | 0.994782 | 11 | 13 | No encontrado | 356,810 derivado | 4,327,007 bytes |
| improved_cnn | 0.775900 | 0.789194 | 0.775900 | 0.770678 | No encontrado | 0.770844 | 0.780800 | 0.662727 | 0.632153 | 18 | 20 | No encontrado | 816,938 derivado | 9,915,067 bytes |
| transfer_model | 0.878800 | 0.878775 | 0.878800 | 0.878532 | No encontrado | 0.891756 | 0.883000 | 0.300070 | 0.332897 | 16 | 20 | No encontrado | No encontrado | 11,614,226 bytes |

Fuentes: `outputs/metrics/*_metrics.json`; `outputs/logs/*_training_log.csv`; `outputs/models/*.keras`.

Formulas usadas:

| Calculo | Formula |
|---|---|
| Diferencia en puntos porcentuales | `(metrica_A - metrica_B) x 100` |
| Diferencia relativa porcentual | `((metrica_A - metrica_B) / metrica_B) x 100` |
| Brecha entrenamiento-validacion | `accuracy_entrenamiento_final - accuracy_validacion_final` |
| Brecha validacion-prueba | `accuracy_validacion_final - accuracy_prueba` |
| Diferencia accuracy-F1 macro | `accuracy_prueba - F1_macro_prueba` |

| Comparacion | Metrica | Diferencia decimal | Puntos porcentuales | Diferencia relativa |
|---|---|---:|---:|---:|
| improved_cnn - baseline_cnn | Accuracy | 0.040700 | 4.07 | 5.54% |
| transfer_model - baseline_cnn | Accuracy | 0.143600 | 14.36 | 19.53% |
| transfer_model - improved_cnn | Accuracy | 0.102900 | 10.29 | 13.26% |
| improved_cnn - baseline_cnn | F1 macro | 0.034849 | 3.48 | 4.74% |
| transfer_model - baseline_cnn | F1 macro | 0.142702 | 14.27 | 19.39% |
| transfer_model - improved_cnn | F1 macro | 0.107853 | 10.79 | 13.99% |

| Modelo | Brecha train-val | Brecha val-prueba | Accuracy - F1 macro |
|---|---:|---:|---:|
| baseline_cnn | 0.187133 | 0.021800 | -0.000630 |
| improved_cnn | -0.009956 | 0.004900 | 0.005222 |
| transfer_model | 0.008756 | 0.004200 | 0.000268 |

# 7. Interpretacion cuantitativa de los resultados

| Modelo | Observacion | Clasificacion | Evidencia |
|---|---|---|---|
| baseline_cnn | La accuracy de entrenamiento sube hasta 0.944133 mientras la validacion final queda en 0.757000 | Confirmada por archivos | `outputs/logs/baseline_cnn_training_log.csv`, filas epoca 0-12 |
| baseline_cnn | Existe una brecha train-val de 18.71 puntos porcentuales | Calculo derivado | Formula seccion 6 |
| baseline_cnn | Hay senales compatibles con sobreajuste | Interpretacion respaldada por resultados | Curvas `outputs/figures/training/baseline_cnn_accuracy.png` y `baseline_cnn_loss.png`; log |
| baseline_cnn | EarlyStopping intervino probablemente antes de 20 epocas | Interpretacion respaldada por archivos | Se ejecutaron 13 epocas frente a maximo 20; `src/training/trainer.py`, lineas 34-40; `outputs/logs/baseline_cnn_training_log.csv` |
| baseline_cnn | ReduceLROnPlateau modifico LR | Confirmada por archivos | LR 0.001, 0.0005, 0.00025 en log |
| improved_cnn | Entrenamiento y validacion terminan cercanos: 0.770844 vs 0.780800 | Confirmada por archivos | `outputs/logs/improved_cnn_training_log.csv`, epoca 19 |
| improved_cnn | No se detuvo antes del maximo de 20 epocas | Confirmada por archivos | 20 epocas ejecutadas; `src/config.py`, linea 53 |
| improved_cnn | ReduceLROnPlateau modifico LR a 0.0005 | Confirmada por archivos | `outputs/logs/improved_cnn_training_log.csv`, epocas 15-19 |
| improved_cnn | La curva sugiere aprendizaje menos sobreajustado que la base | Interpretacion respaldada por resultados | Brecha train-val -0.9956 pp; curvas de training |
| transfer_model | Validacion final 0.883000 y prueba 0.878800 son cercanas | Confirmada por archivos | Log y JSON |
| transfer_model | La mejor `val_accuracy` fue 0.885400 en epoca 16, no la ultima | Confirmada por archivos | `outputs/logs/transfer_model_training_log.csv`, epoca 16 |
| transfer_model | ReduceLROnPlateau modifico LR de 0.0005 a 0.00025 | Confirmada por archivos | `outputs/logs/transfer_model_training_log.csv`, epocas 13-19 |
| transfer_model | Aprendizaje estable y convergente dentro de 20 epocas | Interpretacion respaldada por resultados | Curvas `outputs/figures/training/transfer_model_accuracy.png` y `transfer_model_loss.png` |

# 8. Explicacion de las diferencias entre modelos

| Diferencia observada | Evidencia en codigo | Evidencia en resultados | Interpretacion | Nivel de respaldo |
|---|---|---|---|---|
| `improved_cnn` supera a `baseline_cnn` en accuracy por 4.07 pp | Mayor profundidad, BN, dropout y augmentation en `src/models/improved_cnn.py`, lineas 18-55 | Accuracy 0.775900 vs 0.735200 | La combinacion de regularizacion y mayor capacidad podria mejorar generalizacion | Respaldada por resultados |
| `baseline_cnn` muestra mayor brecha train-val | Sin dropout ni BN; `src/models/baseline_cnn.py`, lineas 13-28 | Brecha 18.71 pp | Compatible con sobreajuste | Respaldada por resultados |
| `transfer_model` supera a ambos | MobileNetV2 ImageNet congelado, resizing 96x96, preprocess_input; `src/models/transfer_model.py`, lineas 16-28 | Accuracy 0.878800; F1 0.878532 | La representacion preentrenada y el cambio de resolucion podrian aportar rasgos mas discriminativos | Respaldada por resultados |
| `improved_cnn` tiene recall bajo en bird y cat | Arquitectura no identifica causas por clase | bird recall 0.521, cat recall 0.512 | Las clases animales pequenas o ambiguas siguen siendo dificiles | Interpretacion plausible |
| `transfer_model` reduce errores totales | Codigo usa MobileNetV2 congelado | Errores CSV: 1212 vs 2241 y 2648 | La cabeza sobre rasgos preentrenados generaliza mejor en este experimento | Respaldada por resultados |
| Fine-tuning parcial | README lo menciona | Codigo congela todo MobileNetV2 | No puede afirmarse fine-tuning parcial | No verificable |

# 9. Analisis por clase

Nota: aciertos y errores se calcularon desde los CSV de errores, usando `support=1000` por clase verificado en los reportes. Formula: `aciertos = 1000 - errores_por_clase`.

## baseline_cnn

| Clase | Precision | Recall | F1-score | Support | Aciertos | Errores | Mayor confusion | N |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| airplane | 0.729055 | 0.818000 | 0.770971 | 1000 | 818 | 182 | ship | 62 |
| automobile | 0.889371 | 0.820000 | 0.853278 | 1000 | 820 | 180 | truck | 90 |
| bird | 0.775385 | 0.504000 | 0.610909 | 1000 | 504 | 496 | cat | 123 |
| cat | 0.486651 | 0.638000 | 0.552142 | 1000 | 638 | 362 | dog | 147 |
| deer | 0.697698 | 0.697000 | 0.697349 | 1000 | 697 | 303 | cat | 88 |
| dog | 0.655670 | 0.636000 | 0.645685 | 1000 | 636 | 364 | cat | 209 |
| frog | 0.794949 | 0.787000 | 0.790955 | 1000 | 787 | 213 | cat | 103 |
| horse | 0.785429 | 0.787000 | 0.786214 | 1000 | 787 | 213 | deer | 61 |
| ship | 0.822846 | 0.850000 | 0.836203 | 1000 | 850 | 150 | airplane | 69 |
| truck | 0.814186 | 0.815000 | 0.814593 | 1000 | 815 | 185 | automobile | 54 |

Top F1: automobile, ship, truck. Menor F1: cat, bird, dog. Top confusiones: dog->cat 209, cat->dog 147, bird->cat 123, frog->cat 103, bird->airplane 90.

## improved_cnn

| Clase | Precision | Recall | F1-score | Support | Aciertos | Errores | Mayor confusion | N |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| airplane | 0.813508 | 0.807000 | 0.810241 | 1000 | 807 | 193 | truck | 76 |
| automobile | 0.886139 | 0.895000 | 0.890547 | 1000 | 895 | 105 | truck | 96 |
| bird | 0.886054 | 0.521000 | 0.656171 | 1000 | 521 | 479 | frog | 146 |
| cat | 0.764179 | 0.512000 | 0.613174 | 1000 | 512 | 488 | dog | 138 |
| deer | 0.742032 | 0.745000 | 0.743513 | 1000 | 745 | 255 | frog | 121 |
| dog | 0.747078 | 0.703000 | 0.724369 | 1000 | 703 | 297 | cat | 75 |
| frog | 0.651195 | 0.926000 | 0.764657 | 1000 | 926 | 74 | truck | 25 |
| horse | 0.821705 | 0.848000 | 0.834646 | 1000 | 848 | 152 | truck | 42 |
| ship | 0.900628 | 0.861000 | 0.880368 | 1000 | 861 | 139 | airplane | 52 |
| truck | 0.679422 | 0.941000 | 0.789099 | 1000 | 941 | 59 | automobile | 35 |

Top F1: automobile, ship, horse. Menor F1: cat, bird, dog. Top confusiones: bird->frog 146, cat->dog 138, cat->frog 121, deer->frog 121, automobile->truck 96. Precision alta y recall bajo: bird y cat. Recall alto y precision baja: frog y truck.

## transfer_model

| Clase | Precision | Recall | F1-score | Support | Aciertos | Errores | Mayor confusion | N |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| airplane | 0.867180 | 0.901000 | 0.883767 | 1000 | 901 | 99 | ship | 46 |
| automobile | 0.932406 | 0.938000 | 0.935194 | 1000 | 938 | 62 | truck | 38 |
| bird | 0.896368 | 0.839000 | 0.866736 | 1000 | 839 | 161 | deer | 43 |
| cat | 0.774549 | 0.773000 | 0.773774 | 1000 | 773 | 227 | dog | 87 |
| deer | 0.846305 | 0.859000 | 0.852605 | 1000 | 859 | 141 | horse | 41 |
| dog | 0.847208 | 0.804000 | 0.825038 | 1000 | 804 | 196 | cat | 111 |
| frog | 0.886320 | 0.920000 | 0.902846 | 1000 | 920 | 80 | cat | 25 |
| horse | 0.890099 | 0.899000 | 0.894527 | 1000 | 899 | 101 | deer | 36 |
| ship | 0.911736 | 0.940000 | 0.925652 | 1000 | 940 | 60 | airplane | 38 |
| truck | 0.935583 | 0.915000 | 0.925177 | 1000 | 915 | 85 | automobile | 48 |

Top F1: automobile, ship, truck. Menor F1: cat, dog, deer. Top confusiones derivadas del CSV de errores: dog->cat 111, cat->dog 87, truck->automobile 48, airplane->ship 46, bird->deer 43. Contradiccion: `reports/tables/transfer_model_confused_classes.csv` reporta dog->cat 147 y cat->dog 61, por lo que debe tratarse como evidencia de una ejecucion o regeneracion distinta.

# 10. Analisis cualitativo de imagenes mal clasificadas

| Indice | Modelo | Clase real | Clase predicha | Confianza | Patron observado | Interpretacion prudente | Ruta |
|---:|---|---|---|---|---|---|---|
| 0 | baseline_cnn | cat | dog | No encontrado | Confusion animal domestico | Podria estar asociado con similitud visual entre clases animales; no puede establecerse causalidad | `outputs/predictions/baseline_cnn_misclassified_samples.csv`; `outputs/figures/evaluation/baseline_cnn_misclassified_samples.png` |
| 2 | baseline_cnn | ship | airplane | No encontrado | Objeto/fondo amplio | Es compatible con confusion por fondo y baja resolucion | mismas rutas |
| 24 | baseline_cnn | dog | deer | No encontrado | Animal cuadrupedo | Sugiere posible influencia de forma global y textura | mismas rutas |
| 9 | improved_cnn | automobile | truck | No encontrado | Vehiculos terrestres | Podria estar asociado con similitud entre vehiculos | `outputs/predictions/improved_cnn_misclassified_samples.csv`; `outputs/figures/evaluation/improved_cnn_misclassified_samples.png` |
| 25 | improved_cnn | bird | deer | No encontrado | Clase animal/natural | La evidencia visual no permite confirmar la causa; es compatible con baja resolucion | mismas rutas |
| 33 | improved_cnn | dog | cat | No encontrado | Animal domestico | Prediccion razonable visualmente en algunas imagenes ambiguas, pero no causal | mismas rutas |
| 10 | transfer_model | airplane | dog | No encontrado | Error no explicado por tabla CSV | Requiere inspeccion visual directa antes de usarlo en texto final | `outputs/predictions/transfer_model_misclassified_samples.csv` |
| 37 | transfer_model | automobile | truck | No encontrado | Vehiculos terrestres | Compatible con similitud de forma y contexto | misma ruta |
| 42 | transfer_model | dog | horse | No encontrado | Animal cuadrupedo | Podria estar asociado con postura/fondo; no demostrable | misma ruta |

# 11. Analisis de Grad-CAM

Fuente tecnica: `src/interpretability/gradcam.py` busca la ultima capa Conv2D, calcula gradientes sobre la clase predicha, normaliza el heatmap y lo superpone a la imagen (`src/interpretability/gradcam.py`, lineas 8-66 y 86-129). `generate_gradcam.py` genera Grad-CAM solo para `improved_cnn` (`generate_gradcam.py`, lineas 14-47).

| Archivo | Modelo | Real | Predicha | Conf. | Observacion visible | Interpretacion posible | Conclusion no demostrable |
|---|---|---|---|---:|---|---|---|
| `outputs/figures/gradcam/improved_cnn_gradcam_0.png` | improved_cnn | cat | cat | 0.86 | Activacion sobre zona central del animal y parte inferior | Compatible con uso de rasgos del objeto | No demuestra causalmente que esas zonas expliquen toda la decision |
| `outputs/figures/gradcam/improved_cnn_gradcam_1.png` | improved_cnn | ship | ship | 0.71 | Activacion dispersa entre barco y zona inferior/fondo | Puede indicar atencion compartida entre objeto y contexto | No confirma dependencia del fondo |
| `outputs/figures/gradcam/improved_cnn_gradcam_2.png` | improved_cnn | ship | ship | 0.91 | Activacion sobre estructura superior y zonas del barco | Compatible con localizacion parcial del objeto | No cuantifica importancia relativa |
| `outputs/figures/gradcam/improved_cnn_gradcam_9.png` | improved_cnn | automobile | truck | 0.58 | Activacion amplia sobre vehiculo | Error compatible con similitud automobile/truck | No prueba que el tamano del vehiculo sea la causa |
| `outputs/figures/gradcam/improved_cnn_gradcam_17.png` | improved_cnn | horse | horse | 0.67 | Activacion sobre parte del cuerpo y entorno cercano | Compatible con uso de rasgos del animal | No confirma separacion precisa del objeto |
| `outputs/figures/gradcam/improved_cnn_gradcam_24.png` | improved_cnn | dog | deer | 0.83 | Activacion parcial sobre objeto y fondo inferior | Compatible con confusion entre animales cuadrupedos | No puede demostrar una causa unica del error |

Limitacion: Grad-CAM se presenta como visualizacion de apoyo, no como explicacion causal completa.

# 12. Valoracion interna de los resultados

| Modelo | Fortalezas | Debilidades | Evidencia cuantitativa | Evidencia cualitativa | Riesgo | Valoracion |
|---|---|---|---|---|---|---|
| baseline_cnn | Arquitectura simple; linea base clara | Brecha train-val alta; peores F1 en cat, bird, dog | Accuracy 0.735200; F1 0.735830; brecha train-val 18.71 pp | Errores frecuentes entre animales | Sobreinterpretar accuracy global | Aceptable con limitaciones |
| improved_cnn | Mejor que base; regularizacion y augmentation | Recall bajo en bird/cat; no supera a transfer | Accuracy 0.775900; F1 0.770678 | Grad-CAM util para ejemplos, pero no causal | Atribuir mejora a un componente aislado | Solido para piloto, con limitaciones por clase |
| transfer_model | Mejor rendimiento global y por clase | Evidencias de junio contradicen CSV de abril; fine-tuning parcial no demostrado | Accuracy 0.878800; F1 0.878532; errores 1212 en CSV abril | Errores restantes cat/dog y vehiculos | Mezclar artefactos de ejecuciones distintas | Solido para el objetivo del piloto |

# 13. Valoracion externa y comparacion cientifica

Comparacion interna valida: es defendible comparar los tres modelos entre si porque comparten dataset CIFAR-10, split reproducible, conjunto de prueba y metricas globales generadas por el mismo flujo (`main.py`, lineas 38-83), con la salvedad de que `transfer_model` tiene preprocesamiento y resolucion interna diferentes.

Comparacion externa posible: requeriria mismo dataset, misma particion, mismo test set, mismo preprocesamiento, misma resolucion, mismo uso de pesos preentrenados, numero de ejecuciones, semillas, hardware, metrica y estrategia de fine-tuning.

Comparacion externa no verificable: el repositorio no contiene referencias bibliograficas ni protocolos externos comparables. No debe afirmarse superioridad frente a otros estudios.

# 14. Validez del experimento y limitaciones

| Aspecto | Estado | Tipo |
|---|---|---|
| Numero de ejecuciones por modelo | Una ejecucion registrada por log | Limitacion real |
| Media/desviacion estandar | No encontrado | Limitacion real |
| Control de semillas | Semilla 42 y determinismo TensorFlow intentado | Fortaleza parcial |
| Repetibilidad | Parcial; faltan hardware/Python y entorno real | Limitacion real |
| Seleccion de mejor resultado | Checkpoint por `val_accuracy`; EarlyStopping por `val_loss` | Limitacion potencial por criterio doble |
| Fuga de datos | No observada explicitamente en codigo | Riesgo potencial bajo |
| Uso del test durante desarrollo | No encontrado | No verificable |
| Comparabilidad | Alta para flujo general; menor por MobileNetV2 96x96/preprocess_input | Limitacion real |
| Hardware | No encontrado | Limitacion real |
| Generalizacion fuera de CIFAR-10 | No evaluada | Limitacion real |
| Resolucion 32x32 | Puede limitar analisis visual fino | Limitacion potencial respaldada por dataset |
| Grad-CAM | Solo para `improved_cnn`, no para los tres modelos | Limitacion real |
| Analisis cualitativo | Sin confianza en CSV de errores | Limitacion real |
| Validacion con usuarios | No encontrada | Limitacion real |
| Analisis estadistico | No encontrado | Limitacion real |

# 15. Contradicciones y datos faltantes

| Elemento | Archivo 1 | Valor 1 | Archivo 2 | Valor 2 | Contradiccion | Impacto | Accion recomendada |
|---|---|---|---|---|---|---|---|
| Fine-tuning | README, lineas 73-78 | "Fine-tuning parcial" | `src/models/transfer_model.py`, linea 26 | `base_model.trainable = False` | Si | No afirmar fine-tuning parcial | Corregir README o implementar/describir solo feature extraction |
| Transfer confusiones | `outputs/predictions/transfer_model_misclassified_samples.csv` | dog->cat 111 | `reports/tables/transfer_model_confused_classes.csv` | dog->cat 147 | Si | No mezclar evidencias abril/junio | Regenerar todos los artefactos de transfer en una sola ejecucion o documentar fecha |
| JSON vs comparison | `outputs/metrics/*_metrics.json` | metricas globales | `reports/tables/model_comparison.csv` | mismos valores | No | Comparacion solida | Usar ambos como evidencia |
| Logs vs curvas | `outputs/logs/*.csv` | datos numericos | `outputs/figures/training/*.png` | visualizacion | No detectable | Curvas apoyan logs | Usar logs para numeros |
| Modelos guardados | `src/training/callbacks.py` | `{model_name}_best.keras` | `outputs/models/*.keras` | nombres coinciden | No | Evidencia consistente | Usar |
| Configs | `src/config.py` | `CONFIGS_DIR` definido | `configs/` | sin archivos | Dato faltante | No hay configs externas | Indicar carpeta vacia |
| Tests | `tests/` | sin archivos | README | no menciona tests | Dato faltante | No hay validacion automatizada | Indicar limitacion |
| Loss de prueba | Codigo de metricas | no calcula | requerimiento capitulo | solicita loss prueba | Dato faltante | Tabla incompleta | Evaluar modelo sobre test si se requiere, sin reentrenar |

# 16. Evidencias listas para incorporar a la documentacion del TFM

| Evidencia | Titulo academico sugerido | Seccion recomendada | Archivo | Lista | Requiere regeneracion | Inconsistencia |
|---|---|---|---|---|---|---|
| Tabla comparativa | Comparacion global de rendimiento de los modelos | 5.7 | `reports/tables/model_comparison.csv` | Si | No | No |
| Reportes por clase | Metricas por clase para cada modelo | 5.8 | `reports/tables/*_classification_report.csv` | Si | No | No |
| Curvas entrenamiento | Evolucion de accuracy y loss | 5.5/5.6 | `outputs/figures/training/*.png` | Si | No | No |
| Matrices confusion | Matrices de confusion por modelo | 5.8 | `outputs/figures/evaluation/*_confusion_matrix.png` | Parcial | Recomendable guardar CSV | Transfer junio vs abril |
| Errores CSV | Muestras mal clasificadas | 5.9 | `outputs/predictions/*_misclassified_samples.csv` | Si | No | Transfer no coincide con top confusiones junio |
| Grad-CAM | Visualizacion Grad-CAM de `improved_cnn` | 5.10 | `outputs/figures/gradcam/*.png` | Si | Opcional ampliar | Solo improved_cnn |
| EDA | Distribucion y ejemplos CIFAR-10 | 5.1 | `outputs/figures/eda/*.png` | Si | No | No |
| Modelos | Checkpoints entrenados | 5.2-5.4 | `outputs/models/*.keras` | Si | No | No |
| Logs | Historial de entrenamiento | 5.5 | `outputs/logs/*.csv` | Si | No | No |

# 17. Propuesta de estructura para documentar los resultados experimentales

| Apartado | Finalidad | Evidencias | Informacion faltante | Relacion con objetivos |
|---|---|---|---|---|
| 5.1 Preparacion del entorno y datos | Describir implementacion real de CIFAR-10 | `src/data/*`, `src/config.py`, EDA | Python/hardware | Reproducibilidad |
| 5.2 Implementacion CNN base | Presentar linea base | `baseline_cnn.py`, modelo, metricas | Parametros verificados por Keras | Modelo de referencia |
| 5.3 Implementacion CNN mejorada | Explicar mejoras tecnicas | `improved_cnn.py`, augmentation, logs | Ablacion de componentes | Evaluar regularizacion |
| 5.4 Implementacion MobileNetV2 | Explicar transferencia | `transfer_model.py`, modelo, metricas | Fine-tuning no demostrado | Comparar transferencia |
| 5.5 Configuracion entrenamiento | Documentar callbacks y logs | `trainer.py`, `callbacks.py`, logs | Tiempo/hardware | Control experimental |
| 5.6 Resultados cuantitativos | Presentar metricas globales | JSON, comparison | Loss test | Evaluacion |
| 5.7 Comparacion de modelos | Integrar diferencias | Tablas y curvas | Significancia estadistica | Discusion |
| 5.8 Analisis por clase | Detectar clases dificiles | classification reports, CSV errores | Matrices numericas CSV | Analisis granular |
| 5.9 Analisis cualitativo | Interpretar errores con cautela | misclassified PNG/CSV | Confianza por error | Visual analytics |
| 5.10 Grad-CAM | Mostrar interpretabilidad | Grad-CAM improved | Grad-CAM para transfer/base | Interpretabilidad |
| 5.11 Discusion integrada | Relacionar arquitectura-resultados | Todas las evidencias | Ablaciones | Sintesis |
| 5.12 Limitaciones | Delimitar validez | Logs, faltantes | Repeticiones | Rigor |
| 5.13 Sintesis | Cerrar hallazgos | Metricas y limitaciones | No aplica | Conclusion del capitulo |

# 18. Informacion faltante y comandos sugeridos

## 18.1 Indispensable

| Faltante | Comando sugerido | Comentario |
|---|---|---|
| Resolver contradiccion de evidencias `transfer_model` | `python generate_evaluation_evidence.py` | Sobrescribe matriz, errores visuales, top confusiones CSV/PNG de transfer; no reentrena; carga `transfer_model_best.keras`; no modifica metricas JSON |
| Matriz de confusion numerica para todos los modelos | No hay script especifico | Recomendable crear script nuevo o ampliar evaluacion, cuidando no mezclar ejecuciones |

## 18.2 Recomendable

| Faltante | Comando sugerido | Comentario |
|---|---|---|
| Grad-CAM para mas modelos o casos | `python generate_gradcam.py` | Actualmente solo usa `improved_cnn_best.keras`, sobrescribe/crea PNG Grad-CAM; no reentrena; requiere TensorFlow, NumPy y modelo guardado |
| Figuras EDA actualizadas | `python generate_eda_figures.py` | Sobrescribe `cifar10_class_distribution.csv` y dos PNG; no reentrena; carga CIFAR-10 |
| Loss de prueba | No ejecutar `main.py` salvo necesidad | Se podria evaluar modelos guardados sin entrenar; `main.py` reentrena y puede producir resultados diferentes |

## 18.3 Opcional

| Faltante | Comando sugerido | Comentario |
|---|---|---|
| Variabilidad estadistica | Repetir entrenamientos con varias semillas | Costoso; cambiaria resultados |
| Hardware y tiempos | Registrar ejecuciones futuras | No recuperable de artefactos actuales |

# 19. Matriz de afirmaciones defendibles

| Afirmacion propuesta | Modelo | Evidencia cuantitativa | Evidencia cualitativa | Archivo | Confianza | Redaccion academica recomendada | Advertencia |
|---|---|---|---|---|---|---|---|
| MobileNetV2 obtuvo el mejor rendimiento interno | transfer_model | Accuracy 0.878800, F1 0.878532 | Menor numero de errores CSV | JSON, comparison, predictions | Totalmente respaldada | "En la comparacion interna, el modelo basado en MobileNetV2 alcanzo el mayor rendimiento global." | No extender a otros estudios |
| La CNN mejorada supera a la CNN base | improved_cnn | +4.07 pp accuracy, +3.48 pp F1 | Menos errores totales | JSON y CSV errores | Totalmente respaldada | "La CNN mejorada incremento el rendimiento respecto de la linea base." | No atribuir a un unico componente |
| La CNN base presenta senales de sobreajuste | baseline_cnn | Brecha train-val 18.71 pp | Curvas training/loss | logs y PNG | Respaldada con limitaciones | "Los logs son compatibles con sobreajuste en la CNN base." | Falta validacion estadistica |
| Cat y dog son clases dificiles | todos, especialmente transfer | Menor F1 en transfer: cat 0.773774, dog 0.825038 | Confusiones cat/dog | reports y predictions | Respaldada con limitaciones | "Las clases cat y dog concentran dificultades relativas." | No afirmar causa visual unica |
| Grad-CAM ayuda a interpretar predicciones | improved_cnn | No cuantitativa | 6 PNG con activaciones | Grad-CAM PNG | Respaldada con limitaciones | "Grad-CAM aporta evidencia visual complementaria." | No causal |
| El experimento es reproducible parcialmente | todos | Semilla 42, logs y artefactos | No aplica | config, seeds, logs | Respaldada con limitaciones | "La reproducibilidad esta parcialmente documentada." | Falta hardware/Python |
| Hubo fine-tuning parcial | transfer_model | No encontrado | No aplica | README vs codigo | No defendible | No usar | Codigo congela MobileNetV2 |

# 20. Conclusion del informe

El repositorio contiene evidencia suficiente para documentar de forma solida la implementacion, los resultados cuantitativos, el analisis por clase, los errores y la interpretabilidad. Las afirmaciones mas defendibles son la superioridad interna de `transfer_model`, la mejora de `improved_cnn` frente a `baseline_cnn`, la existencia de dificultades por clase y la utilidad de Grad-CAM como apoyo visual limitado.

Deben formularse con cautela las explicaciones causales sobre errores, sobreajuste y efecto de cada componente arquitectonico, porque no hay estudios de ablacion, multiples ejecuciones ni analisis estadistico. No debe afirmarse fine-tuning parcial de MobileNetV2, ya que el codigo congela el modelo base.

Informacion faltante principal: hardware, tiempos de entrenamiento, loss de prueba, version real de Python, parametros verificados desde Keras, matrices de confusion numericas y una regeneracion consistente de las evidencias de `transfer_model`. El comando mas relevante para ordenar las evidencias experimentales, sin reentrenar, es `python generate_evaluation_evidence.py`, entendiendo que sobrescribe los artefactos de confusion y errores del modelo de transferencia.

Con la evidencia actual, el capitulo puede cerrarse academicamente si se explicitan las limitaciones y se evita mezclar artefactos de fechas o ejecuciones distintas.
