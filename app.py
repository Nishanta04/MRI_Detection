"""Flask web app for brain MRI tumor classification with the trained VGG16 model."""

import io
import os
from pathlib import Path

# Configure TensorFlow before importing it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        str(BASE_DIR / "VGG16_model" / "model.keras"),
    )
)

# Keep this order identical to the class-folder order used during training.
CLASS_NAMES = [
    name.strip()
    for name in os.environ.get(
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


# --------------------------------------------------
# TensorFlow configuration
# --------------------------------------------------

# Set thread limits before TensorFlow initializes its computation pools.
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# --------------------------------------------------
# Load the trained model once
# --------------------------------------------------

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Set MODEL_PATH to the .keras file."
    )

app.logger.info("Loading model from %s", MODEL_PATH)

# The application performs inference only, so compilation is unnecessary.
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

app.logger.info("Model loaded successfully.")

if model.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError(
        f"Model outputs {model.output_shape[-1]} classes, "
        f"but CLASS_NAMES contains {len(CLASS_NAMES)} entries: "
        f"{CLASS_NAMES}"
    )


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess(image: Image.Image) -> np.ndarray:
    """Match the training pipeline: RGB, 128x128, scaled to [0, 1]."""

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    # Match the nearest-neighbour resizing used during training.
    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.NEAREST,
    )

    array = np.asarray(image, dtype=np.float32) / 255.0

    # Add the batch dimension: (128, 128, 3) -> (1, 128, 128, 3).
    return np.expand_dims(array, axis=0)


# --------------------------------------------------
# Upload validation
# --------------------------------------------------

def validate_upload(file_storage) -> str | None:
    """Validate that an image was provided with an allowed extension."""

    if file_storage is None or not file_storage.filename:
        return "Please choose an image file."

    extension = Path(file_storage.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return (
            f"Unsupported file type '{extension or 'unknown'}'. "
            f"Allowed: {allowed}."
        )

    return None


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def index():
    """Render the home page."""

    return render_template(
        "index.html",
        class_names=CLASS_NAMES,
        display_names=DISPLAY_NAMES,
        image_size=IMAGE_SIZE[0],
        max_upload_mb=MAX_CONTENT_LENGTH // (1024 * 1024),
    )


@app.post("/predict")
def predict():
    """Predict the tumor class for an uploaded MRI image."""

    file = request.files.get("image")

    error = validate_upload(file)

    if error:
        return jsonify({"error": error}), 400

    try:
        raw = file.read()

        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            batch = preprocess(image)

    except (UnidentifiedImageError, OSError):
        return jsonify(
            {"error": "That file could not be read as an image."}
        ), 400

    except Image.DecompressionBombError:
        return jsonify(
            {"error": "Image dimensions are too large to process."}
        ), 400

    # Reuse the model loaded during application startup.
    probabilities = model.predict(batch, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]

    return jsonify(
        {
            "prediction": predicted_class,
            "display_name": DISPLAY_NAMES.get(
                predicted_class,
                predicted_class.replace("_", " ").title(),
            ),
            "confidence": float(probabilities[predicted_index]),
            "probabilities": [
                {
                    "class_name": class_name,
                    "display_name": DISPLAY_NAMES.get(
                        class_name,
                        class_name.replace("_", " ").title(),
                    ),
                    "probability": float(probability),
                }
                for class_name, probability in zip(
                    CLASS_NAMES,
                    probabilities,
                )
            ],
            "model": "VGG16",
            "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        }
    )


@app.get("/health")
def health():
    """Return application health information."""

    return jsonify(
        {
            "status": "ok",
            "model": MODEL_PATH.name,
        }
    )


# --------------------------------------------------
# Error handlers
# --------------------------------------------------

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_error):
    """Return a JSON response when an upload exceeds the size limit."""

    return (
        jsonify(
            {
                "error": (
                    f"File is too large. Maximum upload size is "
                    f"{MAX_CONTENT_LENGTH // (1024 * 1024)} MB."
                )
            }
        ),
        413,
    )


# --------------------------------------------------
# Local development
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )