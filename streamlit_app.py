"""Streamlit app for brain MRI tumor classification using VGG16.

The page itself (HTML, CSS and JS from the original Flask app) lives in the
``frontend/`` folder and is mounted as a Streamlit custom component, so the UI
is the same as before. This file only does the Python work: loading the model,
validating uploads and running predictions.
"""

import base64
import io
import logging
import os
import time
from pathlib import Path

# Configure TensorFlow before importing it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps, UnidentifiedImageError

# Must be the first Streamlit command.
st.set_page_config(
    page_title="NeuroScan · MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logger = logging.getLogger("neuroscan")


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


def get_setting(name: str, default: str) -> str:
    """Read a setting from Streamlit secrets, then env vars, then the default."""

    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        # No secrets.toml present - fall back to environment variables.
        pass

    return os.environ.get(name, default)


# Hugging Face model repository
HF_REPO_ID = get_setting("HF_REPO_ID", "Nishanta04/brain-tumor-model")
HF_MODEL_FILENAME = get_setting("HF_MODEL_FILENAME", "model.keras")

# Optional: set MODEL_PATH to use a local model instead of Hugging Face.
LOCAL_MODEL_PATH = get_setting("MODEL_PATH", "")

# Keep this order identical to the class-folder order used during training.
CLASS_NAMES = [
    name.strip()
    for name in get_setting(
        "CLASS_NAMES",
        "meningioma,pituitary,notumor,glioma",
    ).split(",")
    if name.strip()
]

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

IMAGE_SIZE = (128, 128)
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def display_name(class_name: str) -> str:
    return DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").title())


# --------------------------------------------------
# TensorFlow configuration
# --------------------------------------------------

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    # Streamlit re-runs this script on every interaction, and TensorFlow only
    # allows these to be set once, before it initialises.
    pass


# --------------------------------------------------
# Download & load the trained model
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Download the model from Hugging Face if a local model path has not been
    configured, then load it once and share it across all sessions.
    """

    if LOCAL_MODEL_PATH:
        model_path = Path(LOCAL_MODEL_PATH)

        if not model_path.is_file():
            raise FileNotFoundError(f"Local model not found: {model_path}")

        logger.info("Loading model from local path: %s", model_path)
    else:
        logger.info("Downloading model from Hugging Face: %s", HF_REPO_ID)

        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME,
            repo_type="model",
        )

        logger.info("Model downloaded/cached at: %s", model_path)

    loaded_model = tf.keras.models.load_model(model_path, compile=False)

    # Verify output classes against configured class names.
    if loaded_model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Model outputs {loaded_model.output_shape[-1]} classes, "
            f"but CLASS_NAMES contains {len(CLASS_NAMES)} entries: "
            f"{CLASS_NAMES}"
        )

    logger.info("Model loaded successfully.")
    return loaded_model


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess(image: Image.Image) -> np.ndarray:
    """Convert an MRI image to the model's expected input format."""

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    # Match the resizing method used during training.
    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.NEAREST,
    )

    array = np.asarray(image, dtype=np.float32) / 255.0

    # Shape: (128, 128, 3) -> (1, 128, 128, 3)
    return np.expand_dims(array, axis=0)


# --------------------------------------------------
# Upload validation
# --------------------------------------------------

class UserFacingError(Exception):
    """An error whose message is safe to show in the page."""


def validate_upload(filename: str, size: int) -> None:
    """Validate that an image was provided with an allowed extension and size."""

    if not filename:
        raise UserFacingError("Please choose an image file.")

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UserFacingError(
            f"Unsupported file type '{extension or 'unknown'}'. "
            f"Allowed: {allowed}."
        )

    if size > MAX_CONTENT_LENGTH:
        raise UserFacingError(
            f"File is too large. Maximum upload size is "
            f"{MAX_CONTENT_LENGTH // (1024 * 1024)} MB."
        )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def classify(request: dict) -> dict:
    """Run the model on one uploaded image and build the response payload."""

    try:
        raw = base64.b64decode(request.get("data") or "", validate=True)
    except (ValueError, TypeError):
        raise UserFacingError("That file could not be read as an image.")

    validate_upload(str(request.get("name") or ""), len(raw))

    # Read and preprocess the uploaded image.
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            batch = preprocess(image)
    except (UnidentifiedImageError, OSError):
        raise UserFacingError("That file could not be read as an image.")
    except Image.DecompressionBombError:
        raise UserFacingError("Image dimensions are too large to process.")

    # Run inference using the already-loaded model.
    start_time = time.time()
    probabilities = load_model()(batch, training=False).numpy()[0]
    logger.info("Inference completed in %.2f seconds", time.time() - start_time)

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]

    return {
        "prediction": predicted_class,
        "display_name": display_name(predicted_class),
        "confidence": float(probabilities[predicted_index]),
        "probabilities": [
            {
                "class_name": class_name,
                "display_name": display_name(class_name),
                "probability": float(probability),
            }
            for class_name, probability in zip(CLASS_NAMES, probabilities)
        ],
        "model": "VGG16",
        "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
    }


def handle_request(request: dict) -> dict:
    """Wrap classify() so the page always gets a response it can display."""

    request_id = request.get("request_id")

    try:
        return {"request_id": request_id, "ok": True, "data": classify(request)}
    except UserFacingError as exc:
        return {"request_id": request_id, "ok": False, "error": str(exc)}
    except Exception:
        logger.exception("Model inference failed")
        return {
            "request_id": request_id,
            "ok": False,
            "error": "Prediction failed while processing the image.",
        }


# --------------------------------------------------
# Page: hide Streamlit's own chrome so only the app UI shows
# --------------------------------------------------

st.markdown(
    """
    <style>
      #MainMenu, footer, header,
      [data-testid="stHeader"], [data-testid="stToolbar"],
      [data-testid="stDecoration"], [data-testid="stStatusWidget"],
      [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
      }
      html, body, .stApp,
      [data-testid="stApp"], [data-testid="stAppViewContainer"],
      [data-testid="stMain"] {
        background: #070b14 !important;
      }
      .block-container,
      [data-testid="stMainBlockContainer"],
      [data-testid="stAppViewBlockContainer"] {
        padding: 0 !important;
        max-width: 100% !important;
      }
      [data-testid="stVerticalBlock"] { gap: 0 !important; }
      [data-stale="true"] { opacity: 1 !important; transition: none !important; }
      /* Streamlit renders the component iframe inline, which adds a few pixels of
         baseline gap underneath it and would make the page scroll. */
      iframe, [data-testid="stCustomComponentV1"] {
        display: block !important;
        vertical-align: top;
        border: 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Page: mount the original UI and handle its requests
# --------------------------------------------------

neuroscan_ui = components.declare_component("neuroscan_ui", path=str(FRONTEND_DIR))

request = neuroscan_ui(
    config={
        "model": "VGG16",
        "image_size": IMAGE_SIZE[0],
        "class_count": len(CLASS_NAMES),
        "max_upload_mb": MAX_CONTENT_LENGTH // (1024 * 1024),
    },
    response=st.session_state.get("neuroscan_response"),
    key="neuroscan_ui",
    default=None,
)

# The page sends one request per click on "Analyze scan". Each has a unique
# request_id, so a request is only processed once even though Streamlit keeps
# the last component value across reruns.
if request and request.get("request_id") != st.session_state.get("neuroscan_last_id"):
    st.session_state["neuroscan_last_id"] = request.get("request_id")
    st.session_state["neuroscan_response"] = handle_request(request)
    st.rerun()

# Warm up the model after the page is visible so the first scan is fast.
try:
    load_model()
except Exception as exc:
    logger.exception("Model could not be loaded")
    st.error(f"The model could not be loaded: {exc}")
