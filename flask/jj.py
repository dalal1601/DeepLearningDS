from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('CNN_fruits.h5')
class_names = ['Black Grapes', 'Green Grapes', 'Apple', 'Avocado', 'Banana', 'Orange']

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected.'})

    # Save the uploaded file to a temporary location
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Preprocess the image
    img_height, img_width = 32, 32
    img = load_img(file_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalization

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]

    # Remove the temporary file
    os.remove(file_path)

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    # Ensure the uploads folder exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
