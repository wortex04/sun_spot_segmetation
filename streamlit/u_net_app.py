import base64

import streamlit as st
import cv2
import matplotlib.pyplot as plt
import torch
from main_model.model_init import model_initification
import numpy as np
import torch.nn.functional as F

# Инициализация модели
import streamlit as st

@st.cache
def model_initification_cache():
    return model_initification()
model = model_initification_cache()

# Переменная для сохранения предсказаний
saved_predictions = None

count = 0
def predict(image):
    with torch.no_grad():
        masked, out, x, _ = model.infer(image)
    mask = np.ones((600, 600, 3), dtype=np.int16) * 255
    mask[out == 1] = 0
    return (masked, mask, x)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('data_example/background.jpg')

st.title("Spot Detection")
uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])
status = True
if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    last_uploaded_file = uploaded_file

    # Check if we already have a prediction for this image
    if 'prediction' not in st.session_state or st.session_state['prediction'] is None:
        saved_predictions = predict(image.copy())
        st.session_state['prediction'] = saved_predictions
    else:
        saved_predictions = st.session_state['prediction']
    option = st.selectbox("Выберите что показать", ["masked", "mask", "x"])
    col1, col2 = st.columns(2)
    threshold = st.slider("Порог", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    with col1:
        st.image(image, caption=f'Uploaded Image', use_column_width=True)

    with col2:
        if option == "masked":
            st.image(saved_predictions[0], caption='Masked Image', use_column_width=True)
        elif option == "mask":
            st.image(saved_predictions[1], caption='Mask', use_column_width=True)
        elif option == "x":
            x = F.softmax(saved_predictions[2])[0].squeeze(0)

            mask = np.where(x > threshold, 255, 0)
            mask = np.stack([mask, mask, mask], axis=0).transpose(1, 2, 0)
            st.image(mask, caption='X', use_column_width=True)


