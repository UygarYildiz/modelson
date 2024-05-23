from flask import Flask, request, jsonify
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from g4f.client import Client
import keras

app = Flask(__name__)
client = Client()

# Bitki hastalığı modelini ykleme
model=load_model("bitki_hastaligi_model.keras")


def preprocess_image(image_path):
    # Resmi modelin beklediği formatta ön işleme
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_disease(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    disease_index = np.argmax(prediction)
    return disease_index

def get_disease_info(disease_index):
    disease_names = ["Bakteri", "Hasere", "Mantar", "Saglikli", "Virus"]  # Modelin tahmin ettiği hastalık isimleri
    disease_name = disease_names[disease_index]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Yetiştirmekte olduğum bitkimdeki {disease_name} sorununu çözmek için izleyeceğim adımlar nelerdir?"}]
    )
    # `ChatCompletion` nesnesinden metni doğru şekilde alın
    disease_info = response.choices[0].message.content.strip()
    return disease_info

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = os.path.join('uploads', file.filename)
        file.save(filename)
        
        disease_index = predict_disease(filename)
        disease_info = get_disease_info(disease_index)

        # Tahmin edilen bilgiyi bir txt dosyasına yazma
        with open('disease_info.txt', 'w') as f:
            f.write(disease_info)
        
        return jsonify({'disease_info': disease_info})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)