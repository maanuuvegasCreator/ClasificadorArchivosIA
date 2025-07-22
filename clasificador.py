import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import pdfplumber
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox

# --- Función para leer el contenido de un archivo (con soporte para .txt y .pdf) ---
def read_file_content(filepath):
    try:
        if filepath.lower().endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        elif filepath.lower().endswith('.pdf'):
            text = ""
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        else:
            return None # Tipo de archivo no soportado
    except Exception as e:
        # print(f"Error al leer el archivo {filepath}: {e}") # No imprimir en consola para GUI
        return None

# --- PARTE 1: RECOPILACIÓN Y ENTRENAMIENTO DEL MODELO (AL INICIO DE LA APP) ---
# Se recomienda entrenar el modelo una vez al inicio o cargarlo si ya está entrenado
# Para este ejemplo, lo entrenamos al iniciar la aplicación.

model = None # El modelo se inicializará y entrenará aquí

def train_model():
    global model # Indicamos que vamos a modificar la variable global 'model'

    train_data_path = 'data/train'
    texts = []
    categories = []

    if not os.path.exists(train_data_path):
        messagebox.showerror("Error de Datos", f"La carpeta de entrenamiento '{train_data_path}' no existe. Asegúrate de tener tus archivos de entrenamiento.")
        return False

    for category_name in os.listdir(train_data_path):
        category_path = os.path.join(train_data_path, category_name)

        if os.path.isdir(category_path):
            # print(f"Procesando categoría: {category_name}") # Quitar para GUI
            for filename in os.listdir(category_path):
                filepath = os.path.join(category_path, filename)
                if os.path.isfile(filepath) and (filename.lower().endswith('.txt') or filename.lower().endswith('.pdf')):
                    content = read_file_content(filepath)
                    if content:
                        texts.append(content)
                        categories.append(category_name)

    df = pd.DataFrame({'text': texts, 'category': categories})

    if df.empty:
        messagebox.showwarning("Advertencia", "No se encontraron datos de entrenamiento. Asegúrate de tener archivos .txt o .pdf en 'data/train/facturas' y 'data/train/contratos'.")
        return False

    # Evitar el error de "The number of classes has to be greater than one"
    # Si tenemos menos de 2 categorías o muy pocos ejemplos por categoría
    if len(df['category'].unique()) < 2:
        messagebox.showwarning("Advertencia", f"Se necesita al menos 2 categorías para el entrenamiento. Solo se encontró {len(df['category'].unique())} categoría.")
        return False
    
    # Asegurarse de tener suficientes ejemplos por cada clase para que la división no falle
    # Esto es heurístico, un mínimo de 5-10 ejemplos por clase es lo ideal para empezar
    # Si tienes muy pocos datos, train_test_split puede dejar una clase sin ejemplos en el train set.
    # Si tienes muy pocos datos y aún da el error, puedes probar con test_size=0 (aunque no es lo ideal para evaluación)
    if len(df) < 5: # Si hay menos de 5 documentos en total, el split puede ser problemático
        messagebox.showwarning("Advertencia", "Tienes muy pocos documentos para un entrenamiento robusto. Considera añadir más.")
        # Para evitar errores con dataset muy pequeños, podemos forzar test_size a 0
        # Esto es solo para depuración con pocos datos y no recomendado en producción.
        X_train, y_train = df['text'], df['category']
        X_test, y_test = pd.Series(), pd.Series() # Conjuntos de prueba vacíos
    else:
        # Dividimos los datos, asegurando que haya representación de clases si es posible
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42, stratify=df['category'])
        # stratify=df['category'] ayuda a mantener la proporción de clases en el split

    print(f"Número de ejemplos para entrenamiento: {len(X_train)}")
    print(f"Número de ejemplos para prueba: {len(X_test)}")
    print("Entrenando el modelo...")

    model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
    model.fit(X_train, y_train)

    print("¡Modelo entrenado exitosamente!")
    
    # Evaluación del modelo (opcional, solo para depuración en consola)
    if not X_test.empty:
        y_pred = model.predict(X_test)
        print("--- Informe de Clasificación (Evaluación) ---")
        print(classification_report(y_test, y_pred, zero_division=0)) # zero_division=0 para evitar warnings con pocos datos
        print("\n" + "-"*30 + "\n")
    else:
        print("No hay datos de prueba para evaluar el modelo.")
    
    return True

# --- FUNCIONES DE LA INTERFAZ GRÁFICA ---

def classify_selected_files():
    if model is None:
        messagebox.showerror("Error", "El modelo no ha sido entrenado. Por favor, asegúrate de que haya datos en 'data/train'.")
        return

    # Abrir el diálogo para seleccionar archivos
    file_paths = filedialog.askopenfilenames(
        title="Selecciona Archivos para Clasificar",
        filetypes=[("Text files", "*.txt"), ("PDF files", "*.pdf"), ("All files", "*.*")]
    )

    if not file_paths:
        result_text.insert(tk.END, "No se seleccionaron archivos.\n\n")
        return

    results = []
    for filepath in file_paths:
        content = read_file_content(filepath)
        if content:
            prediction = model.predict([content])
            file_name = os.path.basename(filepath)
            results.append(f"'{file_name}' -> Clasificado como: {prediction[0]}")
        else:
            file_name = os.path.basename(filepath)
            results.append(f"'{file_name}' -> NO SE PUDO LEER o TIPO NO SOPORTADO.")
    
    # Mostrar resultados en el área de texto
    result_text.delete(1.0, tk.END) # Limpiar el texto anterior
    result_text.insert(tk.END, "\n".join(results) + "\n\n")
    messagebox.showinfo("Clasificación Completada", "Archivos clasificados exitosamente.")

# --- CONFIGURACIÓN DE LA INTERFAZ GRÁFICA ---
root = tk.Tk()
root.title("Clasificador de Archivos con IA")
root.geometry("600x500") # Tamaño inicial de la ventana

# Etiqueta de título
title_label = tk.Label(root, text="Clasificador Automático de Documentos", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

# Botón para seleccionar y clasificar archivos
select_button = tk.Button(root, text="Seleccionar y Clasificar Archivos", command=classify_selected_files, font=("Helvetica", 12))
select_button.pack(pady=10)

# Área de texto para mostrar resultados
result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20, font=("Helvetica", 10))
result_text.pack(pady=10)

# --- ENTRENAR EL MODELO AL INICIAR LA APLICACIÓN ---
# Intentamos entrenar el modelo. Si falla, mostramos el mensaje y la app sigue, pero el botón de clasificar no funcionará.
if train_model():
    result_text.insert(tk.END, "Modelo entrenado con éxito. ¡Listo para clasificar!\n\n")
else:
    result_text.insert(tk.END, "Error en el entrenamiento del modelo. Revisa los datos de 'data/train'.\n\n")

# Iniciar el bucle principal de la interfaz
root.mainloop()