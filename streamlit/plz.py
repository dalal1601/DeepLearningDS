import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Path to the model file
model_path = 'CNN/CNN_fruits.h5'

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Define class names (ensure these match the classes used during training)
    class_names = ['Black Grapes', 'Green Grapes', 'apple', 'avocado', 'banana', 'orange']

    # Streamlit interface
    st.title("Fruit Classifier")
    st.write("Upload an image of a fruit, and the model will classify it.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)

        # Check the number of channels and convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        st.image(image, caption='Uploaded Image.', use_container_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        img_height, img_width = 32, 32
        img = image.resize((img_height, img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict the class
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_index = np.argmax(score)

        # Check if the predicted index is within the range of class names
        if predicted_index < len(class_names):
            predicted_class = class_names[predicted_index]
            st.write(f"This image most likely belongs to {predicted_class} with a {100 * np.max(score):.2f}% confidence.")
        else:
            st.error("Prediction index out of range. Please check the class names and model output.")