import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Color Classifier", layout="centered")

st.title("🎨 Color Classification App")
st.markdown("Upload an image and the model will predict its dominant color.")

@st.cache_resource
def load_model():
    try:
        model_path = "/content/drive/MyDrive/my_trained_model.h5"
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

class_names = ['blue', 'green', 'purple', 'red', 'yellow']

uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🧠 Making prediction...")

    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence = float(np.max(predictions)) * 100

    st.success(f"🎯 Predicted Color: **{predicted_label.capitalize()}**")
    st.progress(min(int(confidence), 100))
    st.caption(f"Confidence: {confidence:.2f}%")
