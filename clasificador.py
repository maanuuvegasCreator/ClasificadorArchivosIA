import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os # Importamos la librería 'os' para manejar rutas de archivos

# --- Función para leer el contenido de un archivo ---
def read_file_content(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error al leer el archivo {filepath}: {e}")
        return None

# --- PARTE 1: RECOPILACIÓN DE DATOS DE ENTRENAMIENTO DESDE ARCHIVOS REALES ---
# Definimos la ruta base donde están nuestros datos de entrenamiento
train_data_path = 'data/train'

texts = []
categories = []

# Recorremos cada carpeta (categoría) dentro de 'train'
for category_name in os.listdir(train_data_path):
    category_path = os.path.join(train_data_path, category_name) # Ruta completa a la carpeta de categoría

    # Asegurarse de que sea un directorio
    if os.path.isdir(category_path):
        print(f"Procesando categoría: {category_name}")
        # Recorremos cada archivo dentro de la carpeta de la categoría
        for filename in os.listdir(category_path):
            filepath = os.path.join(category_path, filename) # Ruta completa al archivo
            if os.path.isfile(filepath) and filename.endswith('.txt'): # Solo procesar archivos .txt
                content = read_file_content(filepath)
                if content:
                    texts.append(content)
                    categories.append(category_name) # La categoría es el nombre de la carpeta

# Crear el DataFrame con los datos leídos
df = pd.DataFrame({'text': texts, 'category': categories})

if df.empty:
    print("¡Advertencia! No se encontraron datos de entrenamiento. Asegúrate de tener archivos .txt en 'data/train/facturas' y 'data/train/contratos'.")
    exit() # Sale del script si no hay datos para evitar errores

print("--- Datos de Entrenamiento leídos desde Archivos ---")
print(df)
print("\n" + "-"*30 + "\n")

# --- PARTE 2, 3 y 4: PREPROCESAMIENTO, ENTRENAMIENTO Y EVALUACIÓN (sin cambios) ---
model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)

print(f"Número de ejemplos para entrenamiento: {len(X_train)}")
print(f"Número de ejemplos para prueba: {len(X_test)}")
print("\nEntrenando el modelo...")

model.fit(X_train, y_train)

print("¡Modelo entrenado exitosamente!\n")

y_pred = model.predict(X_test)

print("--- Informe de Clasificación (Evaluación) ---")
print(classification_report(y_test, y_pred))
print("\n" + "-"*30 + "\n")

# --- PARTE 5: CLASIFICAR NUEVOS ARCHIVOS REALES ---
print("--- Probando con Nuevos Archivos de la Carpeta 'data/test' ---")

new_documents_content = []
new_documents_names = []
test_data_path = 'data/test'

for filename in os.listdir(test_data_path):
    filepath = os.path.join(test_data_path, filename)
    if os.path.isfile(filepath) and filename.endswith('.txt'):
        content = read_file_content(filepath)
        if content:
            new_documents_content.append(content)
            new_documents_names.append(filename)

if not new_documents_content:
    print("No se encontraron nuevos archivos .txt en 'data/test' para clasificar.")
else:
    for i, doc_content in enumerate(new_documents_content):
        prediction = model.predict([doc_content])
        print(f"Archivo '{new_documents_names[i]}' -> Clasificado como: {prediction[0]}")