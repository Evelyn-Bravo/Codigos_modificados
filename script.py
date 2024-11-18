import numpy as np
import tensorflow as tf
import sys
import zipfile
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import tempfile
import shutil
import os

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path=sys.argv[1])
interpreter.allocate_tensors()

# Obtener los detalles de las entradas y salidas
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Cargar las etiquetas desde labels.txt
labels = []
with open('labels.txt', 'r') as file:
    for line in file:
        labels.append(line.strip().split(' ')[1])  # Solo tomar la etiqueta, no el índice

# Cargar la imagen
with zipfile.ZipFile(sys.argv[2], 'r') as zip_ref:
    dirpath = tempfile.mkdtemp()
    zip_ref.extractall(dirpath)

    img = load_img(os.path.join(dirpath, sys.argv[3]), target_size=(224, 224))  # Redimensionar a 224x224
    img_array = img_to_array(img)
    img_array = (img_array.astype(np.float32) / 127.0) - 1  # Normalización de la imagen

    # Establecer el tensor de entrada
    interpreter.set_tensor(input_details[0]['index'], img_array.reshape((1, 224, 224, 3)))
    
    # Ejecutar el modelo
    interpreter.invoke()

    shutil.rmtree(dirpath)

# Obtener los resultados de la salida
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class_index = np.argmax(output_data)  # Índice de la clase predicha

# Mostrar la etiqueta correspondiente
predicted_label = labels[predicted_class_index]
print(f"Predicción: {predicted_label}")
