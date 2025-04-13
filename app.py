import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Real AI Eyes", page_icon="🧠", layout="centered")

# Load model
model = load_model("best_model.h5")

st.title("🧠 AI Real Eyes")
st.subheader("Detect whether an image is AI-Generated or Real")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((256, 256))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    label = "AI-Generated" if prediction > 0.5 else "Real"
    confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)

    st.markdown(f"### 🔍 Prediction: `{label}`")
    st.markdown(f"### 🎯 Confidence: `{confidence}%`")
