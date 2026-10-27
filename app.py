"""Flask web app for brain MRI tumor classification with the trained VGG16 model."""

import io
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "VGG16_model" / "model.keras"))

# Must stay in the same order as the class folders used by the training notebook
# (`os.listdir(train_dir)`), because the model's softmax columns follow that order.
# This is the order recorded in VGG16.ipynb when the saved model was evaluated.
CLASS_NAMES = [
    name.strip()
    for name in os.environ.get(
        "CLASS_NAMES", "meningioma,pituitary,notumor,glioma"
    ).split(",")
    if name.strip()
]
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

IMAGE_SIZE = (128, 128)  # training resolution
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Set MODEL_PATH to the .keras file."
    )

app.logger.info("Loading model from %s", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

if model.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError(
        f"Model outputs {model.output_shape[-1]} classes but CLASS_NAMES has "
        f"{len(CLASS_NAMES)} entries: {CLASS_NAMES}"
    )


def preprocess(image: Image.Image) -> np.ndarray:
    """Match the notebook pipeline: RGB, 128x128, scaled to [0, 1]."""
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    # tf.keras.utils.load_img used during training resizes with nearest-neighbour.
    image = image.resize(IMAGE_SIZE, Image.Resampling.NEAREST)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def validate_upload(file_storage) -> str | None:
    if file_storage is None or not file_storage.filename:
        return "Please choose an image file."
    extension = Path(file_storage.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return f"Unsupported file type '{extension or 'unknown'}'. Allowed: {allowed}."
    return None


@app.get("/")
def index():
    return render_template(
        "index.html",
        class_names=CLASS_NAMES,
        display_names=DISPLAY_NAMES,
        image_size=IMAGE_SIZE[0],
        max_upload_mb=MAX_CONTENT_LENGTH // (1024 * 1024),
    )


@app.post("/predict")
def predict():
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
        return jsonify({"error": "That file could not be read as an image."}), 400
    except Image.DecompressionBombError:
        return jsonify({"error": "Image dimensions are too large to process."}), 400

    probabilities = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]

    return jsonify(
        {
            "prediction": predicted_class,
            "display_name": DISPLAY_NAMES.get(
                predicted_class, predicted_class.replace("_", " ").title()
            ),
            "confidence": float(probabilities[predicted_index]),
            "probabilities": [
                {
                    "class_name": class_name,
                    "display_name": DISPLAY_NAMES.get(
                        class_name, class_name.replace("_", " ").title()
                    ),
                    "probability": float(probability),
                }
                for class_name, probability in zip(CLASS_NAMES, probabilities)
            ],
            "model": "VGG16",
            "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": str(MODEL_PATH.name)})


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_error):
    return (
        jsonify(
            {
                "error": f"File is too large. Maximum upload size is "
                f"{MAX_CONTENT_LENGTH // (1024 * 1024)} MB."
            }
        ),
        413,
    )


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
