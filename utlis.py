import numpy as np
from tensorflow.keras.models import load_model
import cv2
import streamlit as st

IMAGE_SIZE = 224

@st.cache_resource
def load_model_cached():
    return load_model("model/melanoma_model_final.h5")

def preprocess_image(img):
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_class(model, img_array):
    prediction = model.predict(img_array)
    return np.argmax(prediction)
