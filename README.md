# 📂 Clasificador Automático de Documentos con IA

¡Bienvenido al **Clasificador Automático de Documentos con IA**! Esta aplicación de escritorio te permite clasificar de forma rápida y sencilla tus archivos de texto (`.txt`) y PDF (`.pdf`) en categorías predefinidas como "Contratos" o "Facturas", utilizando un modelo de Machine Learning.

🚀 Ideal para organizar grandes volúmenes de documentos de manera eficiente.

---

## ✨ Características

* **Clasificación Inteligente:** Utiliza un modelo de Support Vector Machine (SVM) con TF-IDF para aprender de tus documentos y clasificarlos.
* **Soporte Multi-formato:** Lee y procesa tanto archivos `.txt` como `.pdf` (PDFs con texto seleccionable).
* **Interfaz Gráfica Intuitiva (GUI):** Desarrollada con Tkinter para una experiencia de usuario amigable.
* **Selección de Archivos Flexible:** Selecciona archivos individuales o múltiples desde cualquier ubicación de tu ordenador.
* **Re-entrenamiento Sencillo:** Opción para re-entrenar el modelo si añades nuevos ejemplos a tus datos de entrenamiento.

---

## 🛠️ Tecnologías Utilizadas

* **Python 3.x**
* **Scikit-learn:** Para el modelo de Machine Learning (TF-IDF y SVM).
* **Pandas:** Para la gestión de datos.
* **PDFPlumber:** Para la extracción de texto de archivos PDF.
* **Tkinter (ttk):** Para la interfaz gráfica de usuario.

---

## 🚀 Cómo Empezar

Sigue estos pasos para poner en marcha el clasificador en tu máquina.

### 1. Clonar el Repositorio

Primero, clona este repositorio en tu máquina local:

```bash
git clone [https://github.com/maanuuvegasCreator/ClasificadorArchivosIA.git](https://github.com/maanuuvegasCreator/ClasificadorArchivosIA.git)
cd ClasificadorArchivosIA
