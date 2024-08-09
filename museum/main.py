# Carga las librerías
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Desactiva la notación científica
np.set_printoptions(suppress=True)

# Carga el modelo
model = load_model("model/keras_model.h5", compile=False)

# Carga las etiquetas
class_names = open("model/labels.txt", "r").readlines()

print(f"""
        #################################
        MODELO DE INTELIGENCIA ARTIFICIAL
        #################################
        """)

# Revisa cada imagen de la carpeta 'recordings' y realiza una predicción
for file in os.listdir("recordings/"):
    if file.endswith(".png"):
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        image = Image.open(f"recordings/{file}").convert("RGB")

        # Cambia la resolución de la imagen
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # Cambia el formato de la imagen a numpy
        image_array = np.asarray(image)

        # Normaliza la imagen
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Carga la imagen en el array
        data[0] = normalized_image_array

        # Realiza la predicción del modelo
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Imprime la predicción y el puntaje de confianza
        if "OSCURIDAD" in class_name[2:]:
            print(f"[PREDICCIÓN] En la grabación {file}, el museo está en modo {class_name[2:]}")
            print(f"[CONFIANZA] Estoy seguro con un {round(confidence_score * 100, 3)}% de certeza.")

        elif "NO PINTURA" in class_name[2:]:
            print(f"[PREDICCIÓN] En la grabación {file}, el museo tiene {class_name[2:]}")
            print(f"[CONFIANZA] Estoy seguro con un {round(confidence_score * 100, 3)}% de certeza.")
