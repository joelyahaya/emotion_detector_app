
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("emotion_model.h5")  # or "emotion_model.h5" if renamed
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("🧠 Emotion Detector App")
st.write("Upload a facial image and I'll try to guess the mood!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L").resize((48, 48))
    img_array = np.array(image).reshape(1, 48, 48, 1) / 255.0

    prediction = model.predict(img_array)
    predicted_emotion = emotion_labels[np.argmax(prediction)]

    st.image(image, caption="Uploaded Image", width=200)
    st.subheader(f"Predicted Emotion: **{predicted_emotion}**")
