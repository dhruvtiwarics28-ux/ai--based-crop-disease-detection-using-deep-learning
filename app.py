import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("crop_disease_model.keras")

@st.cache_data
def load_class_names():
    with open("class_names.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]

st.title("🌿 AI-Based Crop Disease Detection")
st.write(
    "Upload a crop leaf image to predict its disease "
    "using a deep learning model."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    if st.button("Detect Disease"):
        try:
            with st.spinner("Analyzing leaf image..."):
                model = load_model()
                class_names = load_class_names()

                image = image.resize((224, 224))
                image_array = np.array(image, dtype=np.float32)
                image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                    image_array
                )
                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(image_array, verbose=0)[0]
                predicted_index = int(np.argmax(predictions))
                confidence = float(predictions[predicted_index]) * 100

            st.success("Prediction completed!")
            st.subheader("Predicted Class")
            st.write(class_names[predicted_index])
            st.metric("Model Confidence", f"{confidence:.2f}%")

            st.caption(
                "This is an AI prediction, not a professional diagnosis. "
                "Results may be less reliable for images outside the training dataset."
            )

        except Exception as e:
            st.error(f"Could not process image: {e}")
