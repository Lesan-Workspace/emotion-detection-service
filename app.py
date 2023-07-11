from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/analyze-emotion', methods=['POST'])
def analyze_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'})

    image_data = request.files['image']
    image_array = np.frombuffer(image_data.read(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'No se pudo decodificar la imagen'})

    result = DeepFace.analyze(image, actions=['emotion'])
    emotion = result[0]['dominant_emotion']

    response = {'emotion': emotion}
    return jsonify(response)


if __name__ == '__main__':
    app.run(port = 5000)