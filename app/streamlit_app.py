
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_best.keras"

CLASS_MAPPING_PATH = (
    PROJECT_ROOT / "outputs" / "splits" / "class_mapping.json"
)

IMAGE_SIZE = (224, 224)
TOP_K = 3


# =====================================================
# 3. CUSTOM DESIGN
# =====================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #f5f8f5;
    }

    /* Sidebar */

    [data-testid="stSidebar"] {
        background-color: #153b2c;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Main headings */

    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3 {
        color: #153b2c !important;
        font-weight: 800;
    }

    /* Main text */

    [data-testid="stMain"] p,
    [data-testid="stMain"] li {
        color: #354f40;
    }

    /* Hero banner */

    .hero {
        background: linear-gradient(120deg, #153b2c, #2e7650);
        padding: 30px;
        border-radius: 18px;
        margin-bottom: 25px;
    }

    .hero h2 {
        color: #ffffff !important;
        font-size: 28px;
        font-weight: 800;
    }

    .hero p {
        color: #e5f5e9 !important;
        font-size: 16px;
    }

    /* Result cards */

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce9df;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #52665a !important;
    }

    [data-testid="stMetricValue"] {
        color: #1d6842 !important;
    }

    /* About section */

    [data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #dce9df;
        border-radius: 12px;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #153b2c !important;
    }

    /* Buttons */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }

    /* Footer */

    .footer {
        text-align: center;
        color: #52665a;
        font-size: 13px;
        padding: 20px 0;
    }
    /* Upload information message */
[data-testid="stAlert"] {
    background-color: #1d6842 !important;
    border: 1px solid #2e7650 !important;
    border-radius: 10px !important;
}

[data-testid="stAlert"] p {
    color: #ffffff !important;
}

[data-testid="stAlert"] svg {
    color: #ffffff !important;
}

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 4. LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


# =====================================================
# 5. LOAD CLASS MAPPING
# =====================================================

@st.cache_data
def load_class_mapping():

    if not CLASS_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {CLASS_MAPPING_PATH}"
        )

    with open(
        CLASS_MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        mapping = json.load(file)

    # Handle common nested mapping formats

    if isinstance(mapping, dict):

        for key in (
            "class_mapping",
            "class_indices",
            "class_to_idx",
            "class_names",
            "classes"
        ):

            if key in mapping and isinstance(
                mapping[key], (dict, list)
            ):
                mapping = mapping[key]
                break

    return mapping


# =====================================================
# 6. GET CLASS NAME
# =====================================================

def get_class_name(class_mapping, index):

    index = int(index)

    # Format 1: List of class names

    if isinstance(class_mapping, list):
        return class_mapping[index]

    if not isinstance(class_mapping, dict):
        raise ValueError("Unsupported class mapping format.")

    # Format 2: Index -> class name

    if str(index) in class_mapping:
        return class_mapping[str(index)]

    if index in class_mapping:
        return class_mapping[index]

    # Format 3: Class name -> index

    for class_name, class_index in class_mapping.items():

        try:
            if int(class_index) == index:
                return class_name
        except (ValueError, TypeError):
            continue

    raise ValueError(
        f"Class index {index} was not found in the class mapping."
    )


# =====================================================
# 7. IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # MobileNetV2 preprocessing: [0,255] -> [-1,1]

    image_array = (
        tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )
    )

    # Add batch dimension

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =====================================================
# 8. PREDICTION FUNCTION
# =====================================================

def predict_disease(image, model, class_mapping):

    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    top_indices = np.argsort(predictions)[::-1][:TOP_K]

    results = []

    for index in top_indices:

        index = int(index)

        class_name = get_class_name(
            class_mapping,
            index
        )

        confidence = float(predictions[index]) * 100

        results.append({
            "class_name": class_name,
            "confidence": confidence
        })

    return results


# =====================================================
# 9. FORMAT CLASS NAMES
# =====================================================

def format_class_name(class_name):

    return (
        class_name
        .replace("___", " - ")
        .replace("_", " ")
    )


def get_plant_name(class_name):

    parts = class_name.split("___")

    return parts[0].replace("_", " ")


def get_disease_name(class_name):

    parts = class_name.split("___")

    if len(parts) > 1:
        return parts[1].replace("_", " ")

    return class_name.replace("_", " ")


# =====================================================
# 10. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("# 🌿 PlantCare AI")

    st.markdown("### Plant Disease Detection")

    st.divider()

    st.markdown("### 🌱 Supported Plants")

    st.markdown("""
    - 🍎 Apple
    - 🌽 Corn (Maize)
    - 🫑 Bell Pepper
    - 🥔 Potato
    - 🍅 Tomato
    """)

    st.divider()

    st.markdown("### 🤖 Model Information")

    st.markdown("""
    **Model:** MobileNetV2

    **Input size:** 224 × 224

    **Number of classes:** 23

    **Test accuracy:** 92.21%
    """)

    st.divider()

    st.caption("PlantCare AI | Academic Project")


# =====================================================
# =====================================================
# 11. MAIN HEADER
# =====================================================

st.title("🌿 PlantCare AI")

st.markdown(
    "AI-powered plant disease detection using deep learning."
)


# =====================================================
# 12. HERO BANNER
# =====================================================

st.markdown(
    """
    <div class="hero">
        <h2>Protect Your Plants with AI 🌱</h2>
        <p>
            Upload a leaf image and let our trained deep learning
            model identify possible plant diseases in seconds.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# 13. LOAD MODEL AND MAPPING
# =====================================================

try:

    model = load_model()

    class_mapping = load_class_mapping()

except Exception as error:

    st.error(f"Unable to load model or class mapping: {error}")

    st.stop()


# =====================================================
# 14. IMAGE UPLOAD
# =====================================================

st.subheader("📸 Upload a Leaf Image")

st.write(
    "Choose a clear image of a plant leaf in JPG, JPEG, or PNG format."
)

uploaded_file = st.file_uploader(
    "Upload plant leaf image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# =====================================================
# 15. DISPLAY IMAGE AND ANALYZE
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Identify the currently uploaded file

    current_file = (
        uploaded_file.name,
        uploaded_file.size
    )

    # Reset old results when a new image is uploaded

    if st.session_state.get("current_file") != current_file:

        st.session_state["current_file"] = current_file

        st.session_state.pop(
            "prediction_results",
            None
        )

    st.divider()

    st.subheader("🖼️ Uploaded Image")

    st.image(
        image,
        caption="Your uploaded plant leaf",
        width=450
    )

    st.write("")

    # Analyze button

    analyze_clicked = st.button(
        "🔍 Analyze Plant Image",
        type="primary",
        use_container_width=True
    )

    if analyze_clicked:

        with st.spinner("🌱 Analyzing your plant image..."):

            try:

                results = predict_disease(
                    image,
                    model,
                    class_mapping
                )

                st.session_state["prediction_results"] = results

            except Exception as error:

                st.error(f"Prediction failed: {error}")

    # =================================================
    # 16. DISPLAY RESULTS
    # =================================================

    if "prediction_results" in st.session_state:

        results = st.session_state["prediction_results"]

        best_result = results[0]

        st.divider()

        st.header("🌱 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="🌿 Predicted Plant",
                value=get_plant_name(
                    best_result["class_name"]
                )
            )

        with col2:

            st.metric(
                label="🔬 Predicted Disease",
                value=get_disease_name(
                    best_result["class_name"]
                )
            )

        st.success(
            f"Prediction confidence: "
            f"{best_result['confidence']:.2f}%"
        )

        st.subheader("📊 Top 3 Predictions")

        for rank, result in enumerate(results, start=1):

            class_label = format_class_name(
                result["class_name"]
            )

            confidence = result["confidence"]

            st.write(
                f"**#{rank} {class_label} — {confidence:.2f}%**"
            )

            st.progress(
                min(confidence / 100, 1.0)
            )

        st.info(
            "Note: This AI prediction is based on the trained "
            "PlantVillage dataset. Results on real-world field "
            "images may vary."
        )

        if st.button("🔄 Clear Prediction"):

            st.session_state.pop(
                "prediction_results",
                None
            )

            st.rerun()

else:

    st.info(
        "👆 Upload a plant leaf image to begin disease detection."
    )


# =====================================================
# 17. ABOUT PROJECT
# =====================================================

st.divider()

with st.expander("ℹ️ About this AI Project"):

    st.subheader("About PlantCare AI")

    st.write(
        "PlantCare AI is a plant disease classification application "
        "developed using deep learning and computer vision."
    )

    st.write(
        "The application uses a MobileNetV2 model trained on "
        "the PlantVillage color image dataset."
    )

    st.subheader("Supported Plants")

    st.write(
        "Apple, Corn (Maize), Bell Pepper, Potato and Tomato."
    )

    st.subheader("Model Details")

    st.write("**Architecture:** MobileNetV2")

    st.write("**Number of classes:** 23")

    st.write("**Input image size:** 224 × 224 pixels")

    st.write("**Test accuracy:** 92.21%")

    st.subheader("How It Works")

    st.write("1. Upload an image of a plant leaf.")

    st.write("2. The image is resized and preprocessed.")

    st.write("3. The trained model analyzes the image.")

    st.write("4. The application displays the top three predictions.")

    st.warning(
        "The reported test accuracy was measured on a held-out "
        "PlantVillage dataset. Performance on real-world field "
        "images may differ."
    )


# =====================================================
# 18. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🌿 PlantCare AI | Plant Disease Classification Project
        <br><br>
        Built with Python, TensorFlow, MobileNetV2 and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
