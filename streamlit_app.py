
import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from huggingface_hub import hf_hub_download


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NeuroScan | MRI Classification",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_REPO = os.getenv(
    "HF_MODEL_REPO",
    "Nishanta04/brain-tumor-model"
)

MODEL_FILENAME = os.getenv(
    "HF_MODEL_FILENAME",
    "model.keras"
)

MODEL_PATH = BASE_DIR / "VGG16_model" / "model.keras"

IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "meningioma",
    "pituitary",
    "notumor",
    "glioma",
]

CLASS_DESCRIPTIONS = {
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "notumor": "No Tumor",
    "glioma": "Glioma",
}


# =========================================================
# SESSION STATE
# =========================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None


# =========================================================
# THEME
# =========================================================

PALETTES = {
    "Dark": {
        "bg": "#070B14",
        "panel": "#0E1725",
        "panel_2": "#16202F",
        "ink": "#EEF2F9",
        "muted": "#93A1B8",
        "border": "#233149",
        "accent": "#2DD4BF",
        "accent_strong": "#5EEAD4",
        "accent_ink": "#04231F",
        "focus": "#7FF3E4",
    },
    "Light": {
        "bg": "#F2F5F9",
        "panel": "#FCFDFF",
        "panel_2": "#EEF2F8",
        "ink": "#0E1728",
        "muted": "#56677D",
        "border": "#DBE3EE",
        "accent": "#0F766E",
        "accent_strong": "#115E59",
        "accent_ink": "#FFFFFF",
        "focus": "#0F766E",
    },
}


def token_block(theme):
    return "\n".join(
        f"  --color-{name.replace('_', '-')}: {value};"
        for name, value in PALETTES[theme].items()
    )


def inject_css(theme):
    css = (BASE_DIR / "streamlit_ui.css").read_text(encoding="utf-8")
    css = css.replace("__PALETTE__", token_block(theme))

    st.markdown(
        f"<style>\n{css}\n</style>",
        unsafe_allow_html=True,
    )


inject_css(st.session_state.theme)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():

    model_path = MODEL_PATH

    if not model_path.exists():

        st.info("Downloading the trained VGG16 model...")

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        filenames = [
            MODEL_FILENAME,
            "VGG16_model/model.keras",
        ]

        downloaded_path = None

        for filename in dict.fromkeys(filenames):

            try:
                downloaded_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=filename,
                    local_dir=str(BASE_DIR / "model_download"),
                )
                break

            except Exception:
                continue

        if downloaded_path is None:
            raise RuntimeError(
                "Could not download model.keras. "
                "Check the Hugging Face repository and model filename."
            )

        model_path = Path(downloaded_path)

    model = tf.keras.models.load_model(
        str(model_path),
        compile=False
    )

    return model


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.NEAREST
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================================================
# PREDICTION
# =========================================================

def predict_image(image):

    model = load_model()

    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    if len(predictions) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} outputs, "
            f"but model returned {len(predictions)}."
        )

    predicted_index = int(np.argmax(predictions))

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(predictions[predicted_index]) * 100

    probabilities = {
        CLASS_NAMES[i]: float(predictions[i]) * 100
        for i in range(len(CLASS_NAMES))
    }

    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
    }


# =========================================================
# HEADER
# =========================================================

with st.container(key="topbar"):

    header_left, header_right = st.columns(
        [1, 0.3],
        vertical_alignment="center",
    )

    with header_left:

        st.markdown(
            '<h1 class="brand">Neuro<span class="brand-scan">Scan</span></h1>'
            '<p class="brand-sub">AI-powered Brain MRI Classification Dashboard</p>',
            unsafe_allow_html=True,
        )

    with header_right:

        selected_theme = st.toggle(
            "Light theme",
            value=(st.session_state.theme == "Light"),
            key="theme_toggle",
        )

        new_theme = "Light" if selected_theme else "Dark"

        if new_theme != st.session_state.theme:

            st.session_state.theme = new_theme

            st.rerun()


# =========================================================
# WORKSPACE
# =========================================================

with st.container(key="workspace"):

    left_column, right_column = st.columns(
        [1, 1.15],
        gap="large",
    )

    # -----------------------------------------------------
    # Left pane - image upload
    # -----------------------------------------------------

    with left_column:

        st.markdown(
            '<h2 class="pane-title">MRI scan analysis</h2>'
            '<p class="pane-sub">Upload a brain MRI image to begin classification</p>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload MRI image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:

            file_id = getattr(uploaded_file, "file_id", None) or (
                uploaded_file.name,
                uploaded_file.size,
            )

            if st.session_state.get("uploaded_file_id") != file_id:

                st.session_state.uploaded_file_id = file_id

                st.session_state.prediction = None

            image = Image.open(uploaded_file).convert("RGB")

            st.session_state.uploaded_image = image

            st.image(
                image,
                caption="Uploaded MRI scan",
                width="stretch",
            )

            st.markdown(
                '<p class="status">Image uploaded</p>',
                unsafe_allow_html=True,
            )

            analyze_button = st.button(
                "Analyze MRI Scan",
                use_container_width=True,
            )

            if analyze_button:

                try:

                    with st.spinner("Analyzing MRI scan..."):

                        result = predict_image(image)

                        st.session_state.prediction = result

                except Exception as error:

                    st.error(
                        f"Prediction failed: {error}"
                    )

        else:

            if st.session_state.get("uploaded_file_id") is not None:

                st.session_state.uploaded_file_id = None

                st.session_state.prediction = None

            st.markdown(
                '<p class="hint">Your image preview appears here.</p>',
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------
    # Right pane - results
    # -----------------------------------------------------

    with right_column:

        result = st.session_state.prediction

        if result is not None:

            results_sub = "Ranked class probabilities for the uploaded scan"

        elif uploaded_file is not None:

            results_sub = "Ready to classify the uploaded scan"

        else:

            results_sub = "Your classification results will appear here"

        st.markdown(
            '<h2 class="pane-title">Analysis results</h2>'
            f'<p class="pane-sub">{results_sub}</p>',
            unsafe_allow_html=True,
        )

        if result is None:

            if uploaded_file is not None:

                empty_title = "Ready to analyze"
                empty_copy = "Click Analyze MRI Scan to classify the uploaded image."

            else:

                empty_title = "Waiting for MRI scan"
                empty_copy = "Upload an image and click Analyze MRI Scan."

            st.markdown(
                '<div class="empty">'
                f'<p class="empty-title">{empty_title}</p>'
                f'<p class="empty-copy">{empty_copy}</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        else:

            predicted_class = result["class"]

            confidence = result["confidence"]

            probabilities = result["probabilities"]

            display_name = CLASS_DESCRIPTIONS[predicted_class]

            sorted_probabilities = sorted(
                probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            probability_rows = "".join(
                '<div class="prob-row{top}">'
                '<div class="prob-label">{label}</div>'
                '<div class="prob-track">'
                '<div class="prob-fill{top}" style="width:{probability:.2f}%"></div>'
                '</div>'
                '<div class="prob-value">{probability:.1f}%</div>'
                '</div>'.format(
                    top=" is-top" if class_name == predicted_class else "",
                    label=CLASS_DESCRIPTIONS[class_name],
                    probability=probability,
                )
                for class_name, probability in sorted_probabilities
            )

            st.markdown(
                '<div class="result-head">'
                '<div>'
                '<div class="result-label">Predicted class</div>'
                f'<div class="result-class">{display_name}</div>'
                '</div>'
                '<div class="result-score">'
                f'<div class="score-value">{confidence:.1f}<span class="score-unit">%</span></div>'
                '<div class="score-label">confidence</div>'
                '</div>'
                '</div>'
                '<p class="prob-title">Class probabilities</p>'
                f'{probability_rows}',
                unsafe_allow_html=True,
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<footer class="foot">'
    '<p>NeuroScan | Educational AI Classification Project</p>'
    '<p>This model is for educational and research purposes only. It is not a medical diagnostic tool.</p>'
    '</footer>',
    unsafe_allow_html=True,
)
