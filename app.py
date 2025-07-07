import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import os
from utlis import preprocess_image, load_model_cached, predict_class

st.set_page_config(page_title="Melanoma Detection through AI", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        color: #0b3d91;
        font-size: 36px;
        font-weight: 700;
        text-align: center;
    }
    .subtext {
        color: #333;
        font-size: 16px;
        text-align: center;
        margin-bottom: 20px;
            
    .warn { 
            color: #ffff
            font-size: 36px;
            font-weight: 700;
            text-align: center;}
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Melanoma Detection AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Upload a dermoscopic image to classify as benign or malignant.</div>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

model = load_model_cached()

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        img_array = preprocess_image(img)
        prob = model.predict(img_array)[0]

        label = "🔴 Malignant" if prob[1] > 0.2 else "🔵 Benign"

        st.markdown(f"### Prediction: {label}")
        st.markdown(f"**Confidence Scores:**<br>Benign: `{prob[0]:.2f}` Malignant: `{prob[1]:.2f}`", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f0f0;
        color: red;
        text-align: center;
        padding: 10px 0;
        font-weight: bold;
        font-size: 12px;
        z-index: 1000;
    }
    </style>
    <div class="footer">
        This tool is for educational purposes only and should not be used as a medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)
