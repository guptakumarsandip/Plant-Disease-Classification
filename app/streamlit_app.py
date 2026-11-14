
from pathlib import Path

import json
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "mobilenet_best.keras"
MAPPING_PATH = ROOT / "outputs" / "splits" / "class_mapping.json"

IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)


# --------------------------------------------------
# 3. LOAD MODEL AND CLASS MAPPING
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(MAPPING_PATH, "r", encoding="utf-8") as file:
        class_mapping = json.load(file)

    return [
        name
        for name, index in sorted(
            class_mapping.items(),
            key=lambda item: item[1]
        )
    ]


# --------------------------------------------------
# 4. IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = preprocess_input(image_array)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# --------------------------------------------------
# 5. FORMAT DISEASE NAMES
# --------------------------------------------------

def format_class_name(class_name):

    plant, disease = class_name.split("___", 1)

    plant = (
        plant
        .replace("Corn_(maize)", "Corn (Maize)")
        .replace("Pepper,_bell", "Bell Pepper")
    )

    disease = (
        disease
        .replace("_", " ")
        .replace("  ", " ")
        .strip()
    )

    return plant, disease


# --------------------------------------------------
# 6. PREDICTION FUNCTION
# --------------------------------------------------

def predict_disease(image, model, class_names):

    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    top_indices = np.argsort(predictions)[::-1][:3]

    results = []

    for index in top_indices:

        plant, disease = format_class_name(
            class_names[index]
        )

        results.append({
            "plant": plant,
            "disease": disease,
            "confidence": float(predictions[index]) * 100
        })

    return results


# --------------------------------------------------
# 7. APPLICATION HEADER
# --------------------------------------------------

st.title("🌿 Plant Disease Classification")

st.markdown(
    """
    Upload a plant leaf image to identify its likely disease.

    Our application uses a **MobileNetV2 deep learning model**
    trained to classify 23 plant health and disease categories.
    """
)

st.divider()


# --------------------------------------------------
# 8. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("About the Project")

    st.write(
        """
        This application uses transfer learning with
        MobileNetV2 to classify plant leaf images.
        """
    )

    st.subheader("Supported Plants")

    st.write(
        """
        - 🍎 Apple
        - 🌽 Corn (Maize)
        - 🫑 Bell Pepper
        - 🥔 Potato
        - 🍅 Tomato
        """
    )

    st.divider()

    st.caption(
        "Model: MobileNetV2 | Image size: 224 × 224 | Classes: 23"
    )


# --------------------------------------------------
# 9. CHECK MODEL FILES
# --------------------------------------------------

if not MODEL_PATH.exists():

    st.error(
        "Model file not found. Please place "
        "'mobilenet_best.keras' inside the models folder."
    )

    st.stop()


if not MAPPING_PATH.exists():

    st.error(
        "Class mapping file not found. "
        "Please generate the dataset splits first."
    )

    st.stop()


# --------------------------------------------------
# 10. LOAD MODEL
# --------------------------------------------------

try:

    with st.spinner("Loading the trained model..."):

        model = load_model()
        class_names = load_class_names()

except Exception as error:

    st.error(f"Unable to load the model: {error}")
    st.stop()


# --------------------------------------------------
# 11. IMAGE UPLOAD
# --------------------------------------------------

st.subheader("Upload a Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a leaf photograph",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photograph of a single plant leaf."
)


if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded Leaf Image",
                use_container_width=True
            )

        with col2:

            st.subheader("Image Information")

            st.write(f"Image dimensions: {image.size[0]} × {image.size[1]}")
            st.write(f"Image format: {image.format or 'Unknown'}")

            if st.button(
                "🔍 Analyze Leaf",
                type="primary",
                use_container_width=True
            ):

                with st.spinner("Analyzing the leaf image..."):

                    results = predict_disease(
                        image,
                        model,
                        class_names
                    )

                st.session_state["prediction_results"] = results

    except Exception as error:

        st.error(f"Unable to process this image: {error}")


# --------------------------------------------------
# 12. DISPLAY RESULTS
# --------------------------------------------------

if "prediction_results" in st.session_state:

    results = st.session_state["prediction_results"]

    best_result = results[0]

    st.divider()

    st.header("Prediction Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.success(
            f"Predicted Plant: {best_result['plant']}"
        )

        st.subheader(best_result["disease"])

    with result_col2:

        st.metric(
            label="Model Confidence",
            value=f"{best_result['confidence']:.2f}%"
        )

    st.subheader("Top 3 Predictions")

    for rank, result in enumerate(results, start=1):

        st.write(
            f"**{rank}. {result['plant']} — {result['disease']}**"
        )

        st.progress(
            min(result["confidence"] / 100, 1.0)
        )

        st.caption(
            f"Model score: {result['confidence']:.2f}%"
        )

    st.info(
        "These scores represent the model's predicted class "
        "probabilities, not guaranteed diagnostic certainty. "
        "For agricultural decisions, consult an expert."
    )


# --------------------------------------------------
# 13. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Plant Disease Classification | Powered by TensorFlow "
    "and MobileNetV2 | Educational project"
)
