# Clasificación Automática de Imágenes mediante Deep Learning y Visual Analytics

Proyecto desarrollado como Trabajo Fin de Máster (TFM) enfocado en la clasificación automática de imágenes utilizando técnicas de aprendizaje profundo, redes neuronales convolucionales (CNN) y aprendizaje por transferencia.

El proyecto incorpora análisis visual de resultados e interpretabilidad mediante Grad-CAM, permitiendo comprender el comportamiento de los modelos de clasificación.

---

# Objetivo del proyecto

Desarrollar y evaluar diferentes modelos de clasificación de imágenes utilizando deep learning, comparando su rendimiento y complementando el análisis con herramientas de visualización e interpretabilidad.

---

# Dataset utilizado

Se utilizó el dataset público **CIFAR-10**, ampliamente empleado en tareas académicas de clasificación de imágenes.

Características principales:

- 60.000 imágenes a color
- Resolución de 32x32 píxeles
- 10 clases balanceadas
- Dataset público y reproducible

Clases del dataset:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

# Modelos implementados

## 1. Baseline CNN

Red neuronal convolucional base utilizada como línea de referencia.

Características:

- Capas convolucionales
- MaxPooling
- Capas densas
- Arquitectura simple y ligera

---

## 2. Improved CNN

Versión optimizada de la CNN base incorporando técnicas de regularización.

Técnicas utilizadas:

- Data Augmentation
- Batch Normalization
- Dropout
- Arquitectura más profunda

---

## 3. Transfer Learning

Modelo basado en aprendizaje por transferencia utilizando MobileNetV2 preentrenado sobre ImageNet.

Características:

- Transfer Learning
- MobileNetV2
- GlobalAveragePooling
- Fine-tuning parcial

---

# Resultados obtenidos

| Modelo | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Baseline CNN | 0.7352 | 0.7451 | 0.7352 | 0.7358 |
| Improved CNN | 0.7759 | 0.7891 | 0.7759 | 0.7707 |
| Transfer Learning | 0.8788 | 0.8787 | 0.8788 | 0.8785 |

El modelo basado en aprendizaje por transferencia obtuvo el mejor rendimiento global del experimento.

---

# Visual Analytics e Interpretabilidad

El proyecto incluye diferentes herramientas de análisis visual:

- Curvas de entrenamiento
- Matrices de confusión
- Comparación de modelos
- Análisis de errores
- Grad-CAM

Grad-CAM permite visualizar las regiones de la imagen utilizadas por el modelo para realizar las predicciones.

---

# Estructura del proyecto

```text
tfm-image-classification/
├── configs/
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── visualization/
│   └── interpretability/
├── outputs/
├── reports/
├── tests/
├── requirements.txt
├── README.md
└── main.py
```

---

# Instalación

Clonar repositorio:

```bash
git clone https://github.com/WiloCh/tfm-deep-learning-image-classification.git
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

# Ejecución

## Entrenar modelo base

```bash
python main.py --model baseline_cnn
```

## Entrenar modelo mejorado

```bash
python main.py --model improved_cnn
```

## Entrenar modelo con transfer learning

```bash
python main.py --model transfer_model
```

---

# Generación de Grad-CAM

```bash
python generate_gradcam.py
```

---

# Generación de figuras EDA

```bash
python generate_eda_figures.py
```

---

# Generación de evidencias de evaluación

```bash
python generate_evaluation_evidence.py
```

Genera evidencias complementarias de evaluación: matriz de confusión, ejemplos mal clasificados y clases más confundidas del modelo `transfer_model_best.keras`.

---

# Tecnologías utilizadas

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# Autor

William Anderson Chugchilan Hinojosa

Máster en Big Data y Visual Analytics  
Universidad Internacional de La Rioja (UNIR)

---

# Licencia

Proyecto desarrollado con fines académicos y de investigación.
