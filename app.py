import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

app = Flask(__name__)

# Configurar credenciales desde variable de entorno
credentials_json = os.environ['GOOGLE_APPLICATION_CREDENTIALS_CONTENT']
with open('credentials.json', 'w') as f:
    f.write(credentials_json)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'

# Autenticar con Google Drive
credentials = service_account.Credentials.from_service_account_file('credentials.json')
drive_service = build('drive', 'v3', credentials=credentials)

# Función para descargar archivos desde Drive
def download_file(file_id, destination):
    request = drive_service.files().get_media(fileId=file_id)
    with open(destination, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Descargando {destination}: {int(status.progress() * 100)}%")

# Descargar modelos al iniciar la aplicación
MODEL_JSON_ID = 'TU_ID_DEL_ARCHIVO_JSON'  # Reemplaza con el ID real
WEIGHTS_ID = 'TU_ID_DEL_ARCHIVO_HDF5'     # Reemplaza con el ID real
download_file(MODEL_JSON_ID, 'resnet-50-MRI.json')
download_file(WEIGHTS_ID, 'weights.hdf5')

# Cargar el modelo
with open('resnet-50-MRI.json', 'r') as json_file:
    model_json = json_file.read()
model = tf.keras.models.model_from_json(model_json)
model.load_weights('weights.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Ruta para predicción
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se proporcionó archivo'}), 400
    file = request.files['file']
    # Leer y procesar la imagen
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    # Hacer predicción
    prediction = model.predict(img)
    has_tumor = np.argmax(prediction[0])
    return jsonify({'has_tumor': int(has_tumor)})  # 0 = no tumor, 1 = tumor

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)