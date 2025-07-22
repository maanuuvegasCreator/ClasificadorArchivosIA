import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import pdfplumber
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk

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
                    # extract_text() puede devolver None si no hay texto, lo manejamos con or ""
                    text += page.extract_text() or ""
            return text
        else:
            # Tipo de archivo no soportado o extensión incorrecta
            return None
    except Exception as e:
        # En una GUI, generalmente no imprimimos errores de lectura de archivo aquí
        # sino que la función devuelve None y el llamador lo maneja.
        return None

# --- PARTE 1: RECOPILACIÓN Y ENTRENAMIENTO DEL MODELO (AL INICIO DE LA APP) ---
model = None
training_successful = False # Variable para rastrear si el entrenamiento fue exitoso

def train_model():
    global model, training_successful
    
    # Limpiar el área de resultados y actualizar la barra de estado
    if tk.current_task_is_retrain: # Si se llama desde el botón de re-entrenar
        result_text.delete(1.0, tk.END)
        status_bar.config(text="Iniciando re-entrenamiento del modelo...")
    else: # Si se llama al inicio de la aplicación
        status_bar.config(text="Cargando y entrenando el modelo inicial...")
    
    root.update_idletasks() # Forzar la actualización visual de la GUI

    train_data_path = 'data/train'
    texts = []
    categories = []

    # Verificar si la carpeta de entrenamiento existe
    if not os.path.exists(train_data_path):
        messagebox.showerror("Error de Datos", f"La carpeta de entrenamiento '{train_data_path}' no existe. Asegúrate de tener tus archivos de entrenamiento.")
        status_bar.config(text="Error: Carpeta de entrenamiento no encontrada.")
        training_successful = False
        return False

    # Recorrer las subcarpetas (categorías) y leer los archivos
    for category_name in os.listdir(train_data_path):
        category_path = os.path.join(train_data_path, category_name)

        if os.path.isdir(category_path): # Asegurarse de que sea un directorio
            for filename in os.listdir(category_path):
                filepath = os.path.join(category_path, filename)
                # Procesar solo archivos .txt o .pdf
                if os.path.isfile(filepath) and (filename.lower().endswith('.txt') or filename.lower().endswith('.pdf')):
                    content = read_file_content(filepath)
                    if content: # Solo añadir si se pudo leer el contenido
                        texts.append(content)
                        categories.append(category_name)

    df = pd.DataFrame({'text': texts, 'category': categories})

    # Verificar si se encontraron datos de entrenamiento
    if df.empty:
        messagebox.showwarning("Advertencia", "No se encontraron datos de entrenamiento. Asegúrate de tener archivos .txt o .pdf en 'data/train/facturas' y 'data/train/contratos'.")
        status_bar.config(text="Advertencia: No hay datos de entrenamiento. Modelo no entrenado.")
        training_successful = False
        return False

    # Verificar que haya al menos dos categorías para la clasificación
    unique_categories = df['category'].unique()
    if len(unique_categories) < 2:
        messagebox.showwarning("Advertencia", f"Se necesita al menos 2 categorías para el entrenamiento. Solo se encontró {len(unique_categories)} categoría: {', '.join(unique_categories)}.")
        status_bar.config(text="Advertencia: Insuficientes categorías para entrenar.")
        training_successful = False
        return False
    
    # Advertencia si hay muy pocos documentos en general (afecta la robustez del modelo)
    if len(df) < 5: 
        # Esta advertencia es más para el desarrollador que para el usuario final en la GUI
        # messagebox.showwarning("Advertencia", "Tienes muy pocos documentos para un entrenamiento robusto. Considera añadir más.")
        X_train, y_train = df['text'], df['category']
        X_test, y_test = pd.Series(), pd.Series() # Conjuntos de prueba vacíos si hay muy pocos datos
    else:
        # Dividir los datos en conjuntos de entrenamiento y prueba, estratificando por categoría
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42, stratify=df['category'])

    # Entrenar el modelo
    model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True)) # probability=True para posibles futuras mejoras (confianza)
    model.fit(X_train, y_train)

    # Actualizar la barra de estado y la variable de éxito del entrenamiento
    status_bar.config(text="Modelo entrenado exitosamente. ¡Listo para clasificar!")
    training_successful = True
    return True

# --- FUNCIONES DE LA INTERFAZ GRÁFICA ---

def classify_selected_files():
    # Verificar si el modelo ha sido entrenado exitosamente
    if model is None or not training_successful:
        messagebox.showerror("Error", "El modelo no ha sido entrenado correctamente. Por favor, revisa los datos de entrenamiento y re-entrena el modelo.")
        return

    # Obtener la ruta de la carpeta de Descargas del usuario para el diálogo de archivos
    downloads_folder = os.path.join(os.path.expanduser('~'), 'Downloads')

    # Abrir el diálogo para seleccionar archivos
    file_paths = filedialog.askopenfilenames(
        title="Selecciona Archivos para Clasificar",
        initialdir=downloads_folder, # Abre el diálogo en la carpeta de Descargas
        filetypes=[("Text files", "*.txt"), ("PDF files", "*.pdf"), ("All files", "*.*")] # Tipos de archivo permitidos
    )

    # Si no se seleccionaron archivos, limpiar y mostrar mensaje
    if not file_paths:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "No se seleccionaron archivos para clasificar.\n\n")
        return

    results = [] # Lista para almacenar los resultados de clasificación
    result_text.delete(1.0, tk.END) # Limpiar el área de texto de resultados antes de nuevas clasificaciones
    status_bar.config(text=f"Clasificando {len(file_paths)} archivos...")
    root.update_idletasks() # Forzar actualización de la GUI para mostrar el mensaje de estado

    # Bucle para clasificar cada archivo seleccionado
    for i, filepath in enumerate(file_paths):
        content = read_file_content(filepath)
        file_name = os.path.basename(filepath) # Obtener solo el nombre del archivo
        
        if content: # Si el contenido del archivo se pudo leer
            prediction = model.predict([content])
            # Formato de salida conciso: 'nombre_archivo' -> categoria
            results.append(f"'{file_name}' -> {prediction[0]}")
        else:
            # Si el archivo no se pudo leer o el tipo no es soportado
            results.append(f"'{file_name}' -> Error al leer o tipo no soportado.")
        
        # Actualizar la barra de estado con el progreso
        status_bar.config(text=f"Procesando {i+1}/{len(file_paths)}: {file_name}")
        root.update_idletasks() # Forzar actualización

    # Mostrar todos los resultados en el área de texto de la GUI
    result_text.insert(tk.END, "\n--- Resultados de Clasificación ---\n")
    if not results: # Si no se clasificó ningún archivo exitosamente
        result_text.insert(tk.END, "No se pudo clasificar ningún archivo seleccionado.\n")
    else:
        result_text.insert(tk.END, "\n".join(results) + "\n\n")
    
    status_bar.config(text="Clasificación completada. ¡Listo para nuevas operaciones!")
    messagebox.showinfo("Clasificación Completada", "Archivos clasificados exitosamente.")

# --- CONFIGURACIÓN DE LA INTERFAZ GRÁFICA ---
root = tk.Tk()
root.title("Clasificador de Archivos con IA")
root.geometry("700x600") # Tamaño inicial de la ventana
root.resizable(True, True) # Permitir que el usuario redimensione la ventana

# Configuración de estilos para widgets ttk (apariencia más moderna)
style = ttk.Style()
style.theme_use('clam') # Puedes probar otros temas: 'default', 'alt', 'clam', 'vista', 'xpnative'

# Definición de colores y fuentes para una estética consistente
BG_COLOR = "#F0F0F0"       # Gris claro para el fondo
BUTTON_COLOR = "#4CAF50"   # Verde para el botón principal
BUTTON_TEXT_COLOR = "white" # Texto blanco para el botón principal
FONT_TITLE = ("Helvetica", 18, "bold") # Fuente para el título
FONT_BUTTON = ("Helvetica", 12)       # Fuente para los botones
FONT_TEXT = ("Consolas", 10)          # Fuente monoespaciada para el área de texto de resultados

root.configure(bg=BG_COLOR) # Establecer el color de fondo de la ventana principal

# Marco principal para contener y organizar todos los widgets, con relleno (padding)
main_frame = ttk.Frame(root, padding="15 15 15 15", style='My.TFrame')
main_frame.pack(fill=tk.BOTH, expand=True) # El marco se expandirá para llenar la ventana
style.configure('My.TFrame', background=BG_COLOR) # Aplicar color de fondo al marco

# Etiqueta de título de la aplicación
title_label = ttk.Label(main_frame, text="Clasificador Automático de Documentos", font=FONT_TITLE, background=BG_COLOR)
title_label.pack(pady=(0, 15)) # Espacio vertical debajo del título

# Marco para agrupar los botones
button_frame = ttk.Frame(main_frame, style='My.TFrame')
button_frame.pack(pady=(0, 15))

# Botón para seleccionar y clasificar archivos
select_button = ttk.Button(button_frame, text="Seleccionar y Clasificar Archivos", command=classify_selected_files, style='Accent.TButton')
select_button.pack(side=tk.LEFT, padx=5) # Colocar a la izquierda dentro del marco de botones, con espacio horizontal

# Botón para re-entrenar el modelo
# Usa un lambda para pasar un argumento (una variable auxiliar) a train_model
retrain_button = ttk.Button(button_frame, text="Re-entrenar Modelo", 
                            command=lambda: (setattr(tk, 'current_task_is_retrain', True), train_model()), # Establecer variable y llamar
                            style='TButton')
retrain_button.pack(side=tk.LEFT, padx=5) # Colocar a la izquierda, al lado del botón anterior

# Estilos personalizados para los botones
style.configure('Accent.TButton', background=BUTTON_COLOR, foreground=BUTTON_TEXT_COLOR, font=FONT_BUTTON, padding=10)
style.map('Accent.TButton', background=[('active', '#5CB85C')]) # Color al pasar el ratón por encima (hover)
style.configure('TButton', font=FONT_BUTTON, padding=10) # Estilo para el botón de re-entrenar (por defecto)

# Etiqueta para indicar el área de resultados
results_label = ttk.Label(main_frame, text="Resultados de la Operación:", font=("Helvetica", 12, "bold"), background=BG_COLOR)
results_label.pack(pady=(0, 5), anchor=tk.W) # Alinear a la izquierda (West)

# Área de texto desplazable para mostrar los resultados de clasificación
result_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, font=FONT_TEXT, bg="#FFFFFF", fg="#333333", relief=tk.FLAT, bd=2)
result_text.pack(expand=True, fill=tk.BOTH) # El área de texto se expandirá y rellenará el espacio disponible

# Barra de estado en la parte inferior de la ventana
status_bar = ttk.Label(root, text="Iniciando...", relief=tk.SUNKEN, anchor=tk.W, background=BG_COLOR, font=("Helvetica", 9))
status_bar.pack(side=tk.BOTTOM, fill=tk.X) # Se ancla a la parte inferior y se expande horizontalmente


# --- ENTRENAR EL MODELO AL INICIAR LA APLICACIÓN ---
# Esta variable auxiliar se usa para que train_model sepa si se está llamando por primera vez
tk.current_task_is_retrain = False 
train_model() # Llamada inicial para entrenar el modelo cuando la app se abre

# Si el entrenamiento inicial falló, se muestra un mensaje en el área de texto
if not training_successful:
    result_text.insert(tk.END, "Error crítico: El modelo no pudo ser entrenado inicialmente. Por favor, resuelve los problemas de datos y reinicia la aplicación.\n\n")


# Iniciar el bucle principal de la interfaz
root.mainloop()