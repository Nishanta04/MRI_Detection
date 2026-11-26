
"""Streamlit app for brain MRI tumor classification using VGG16."""

import os

# Configure TensorFlow before importing it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps, UnidentifiedImageError
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Brain MRI Tumor Detection",
    page_icon="🧠",
    layout="wide",
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

HF_REPO_ID = "Nishanta04/brain-tumor-model"
MODEL_FILENAME = "model.keras"

CLASS_NAMES = [
    "meningioma",
    "pituitary",
    "notumor",
    "glioma",
]

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

IMAGE_SIZE = (128, 128)


# --------------------------------------------------
# TensorFlow configuration
# --------------------------------------------------

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    pass


# --------------------------------------------------
# Download and load model from Hugging Face
# --------------------------------------------------

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model",
    )

    loaded_model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    if loaded_model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Model has {loaded_model.output_shape[-1]} outputs, "
            f"but {len(CLASS_NAMES)} class names were provided."
        )

    return loaded_model


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image):

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.NEAREST,
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    array = np.expand_dims(array, axis=0)

    return array


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🧠 Brain MRI Tumor Detection")

st.markdown(
    """
    Upload a brain MRI scan to classify it using a
    **VGG16 deep learning model** trained to identify
    four brain MRI image classes.
    """
)

st.caption(
    "Classes: Meningioma • Pituitary • No Tumor • Glioma"
)

st.divider()


# --------------------------------------------------
# Load model
# --------------------------------------------------

try:
    with st.spinner("Downloading/loading model from Hugging Face..."):
        model = load_model()

except Exception as e:
    st.error("Unable to download or load the trained model.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)
        image.load()

        image = ImageOps.exif_transpose(image)

        left, right = st.columns([1, 1])

        with left:
            st.subheader("Uploaded MRI")
            st.image(
                image,
                use_container_width=True,
            )

        with right:
            st.subheader("Classification")

            if st.button(
                "Analyze MRI Scan",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner("Analyzing MRI scan..."):

                    processed_image = preprocess_image(image)

                    predictions = model.predict(
                        processed_image,
                        verbose=0,
                    )

                    probabilities = predictions[0]

                    predicted_index = int(np.argmax(probabilities))

                    predicted_class = CLASS_NAMES[predicted_index]

                    confidence = float(
                        probabilities[predicted_index] * 100
                    )

                st.success(
                    f"Prediction: {DISPLAY_NAMES[predicted_class]}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%",
                )

                st.divider()

                st.subheader("Class-wise probabilities")

                for i, class_name in enumerate(CLASS_NAMES):

                    probability = float(probabilities[i] * 100)

                    st.write(
                        f"**{DISPLAY_NAMES[class_name]}** "
                        f"— {probability:.2f}%"
                    )

                    st.progress(
                        min(max(probability / 100.0, 0.0), 1.0)
                    )

    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file could not be read as an image.")

    except Exception as e:
        st.error("Something went wrong while analyzing the image.")
        st.exception(e)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "VGG16 Brain MRI Classification | Academic Project"
)

st.warning(
    "This project is for educational purposes only. "
    "Predictions are not a medical diagnosis."
)
