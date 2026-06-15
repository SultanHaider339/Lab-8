import os
import urllib.parse

import streamlit as st
import tensorflow as tf


# ============================================================
# Optional Hugging Face model settings
# ============================================================
# Agar aap ne apna TensorFlow/Keras model Hugging Face par upload kiya hai,
# to direct public URL yahan paste kar dein:
#
# Example:
# HF_MODEL_URL = "https://huggingface.co/username/repo-name/resolve/main/model.keras"
# HF_LABELS_URL = "https://huggingface.co/username/repo-name/resolve/main/labels.txt"
#
# No API key required. Repo public hona chahiye.
HF_MODEL_URL = os.getenv("HF_MODEL_URL", "").strip()
HF_LABELS_URL = os.getenv("HF_LABELS_URL", "").strip()


APP_TITLE = "AI Image Classifier"
DEFAULT_IMAGE_SIZE = 224


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="centered"
)


def clean_label(line: str) -> str:
    """
    Supports labels like:
    0 cat
    1 dog
    OR:
    cat
    dog
    """
    line = line.strip()
    if not line:
        return ""

    parts = line.split(" ", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1].strip()

    return line


def file_name_from_url(url: str, fallback: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = os.path.basename(parsed.path)
    return name if name else fallback


@st.cache_resource(show_spinner="Loading AI model...")
def load_ai_model():
    """
    If HF_MODEL_URL is provided, load TensorFlow/Keras model from Hugging Face direct URL.
    Otherwise, use TensorFlow/Keras free MobileNetV2 model.
    """
    if HF_MODEL_URL:
        model_file_name = file_name_from_url(HF_MODEL_URL, "model.keras")

        model_path = tf.keras.utils.get_file(
            fname=model_file_name,
            origin=HF_MODEL_URL
        )

        model = tf.keras.models.load_model(model_path, compile=False)
        return model, "custom_hf"

    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        input_shape=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3)
    )
    return model, "mobilenetv2_imagenet"


@st.cache_resource(show_spinner=False)
def load_labels():
    """
    Optional labels.txt from Hugging Face.
    One class per line is recommended.
    """
    if not HF_LABELS_URL:
        return []

    labels_file_name = file_name_from_url(HF_LABELS_URL, "labels.txt")

    labels_path = tf.keras.utils.get_file(
        fname=labels_file_name,
        origin=HF_LABELS_URL
    )

    labels = []
    with tf.io.gfile.GFile(labels_path, "r") as file:
        for line in file:
            label = clean_label(line)
            if label:
                labels.append(label)

    return labels


def get_model_image_size(model) -> tuple[int, int]:
    """
    Reads image height/width from model input shape.
    Falls back to 224x224.
    """
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = input_shape[1] if len(input_shape) > 1 and input_shape[1] else DEFAULT_IMAGE_SIZE
    width = input_shape[2] if len(input_shape) > 2 and input_shape[2] else DEFAULT_IMAGE_SIZE

    return int(height), int(width)


def preprocess_image(file_bytes: bytes, height: int, width: int, model_type: str, custom_preprocess: str):
    """
    Uses TensorFlow only for image decoding and preprocessing.
    No PIL, no NumPy, no requests, no transformers.
    """
    image = tf.io.decode_image(
        file_bytes,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(image, (height, width))
    image = tf.cast(image, tf.float32)

    if model_type == "mobilenetv2_imagenet":
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    else:
        if custom_preprocess == "Scale 0 to 1":
            image = image / 255.0
        elif custom_preprocess == "Scale -1 to 1":
            image = (image / 127.5) - 1.0
        else:
            image = image

    image = tf.expand_dims(image, axis=0)
    return image


def normalize_scores(raw_output):
    """
    Converts logits or probabilities into probability-like scores.
    """
    scores = tf.convert_to_tensor(raw_output)

    if len(scores.shape) > 2:
        scores = tf.reshape(scores, (scores.shape[0], -1))

    scores = tf.cast(scores[0], tf.float32)

    min_score = float(tf.reduce_min(scores).numpy())
    max_score = float(tf.reduce_max(scores).numpy())
    total_score = float(tf.reduce_sum(scores).numpy())

    looks_like_probabilities = (
        min_score >= 0.0 and
        max_score <= 1.0 and
        0.98 <= total_score <= 1.02
    )

    if not looks_like_probabilities:
        scores = tf.nn.softmax(scores)

    return scores


def predict_custom_model(model, image_batch, labels):
    raw_output = model(image_batch, training=False)
    scores = normalize_scores(raw_output)

    total_classes = int(scores.shape[-1])
    top_k = min(5, total_classes)

    values, indices = tf.math.top_k(scores, k=top_k)

    results = []
    for score, index in zip(values.numpy(), indices.numpy()):
        index = int(index)
        label = labels[index] if index < len(labels) else f"Class {index}"
        results.append((label, float(score)))

    return results


def predict_mobilenetv2(model, image_batch):
    predictions = model.predict(image_batch, verbose=0)

    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
        predictions,
        top=5
    )[0]

    results = []
    for _, label, score in decoded:
        label = label.replace("_", " ").title()
        results.append((label, float(score)))

    return results


def show_results(results):
    best_label, best_score = results[0]

    st.success(f"Prediction: **{best_label}**")
    st.metric("Confidence", f"{best_score * 100:.2f}%")

    chart_data = {
        label: score
        for label, score in results
    }

    st.subheader("Top Predictions")
    st.bar_chart(chart_data)

    st.write("Detailed scores:")
    for label, score in results:
        st.write(f"- **{label}**: {score * 100:.2f}%")


# ============================================================
# UI
# ============================================================

st.title("🤖 AI Image Classifier")
st.write(
    "Upload an image and this AI model will classify it. "
    "This app uses TensorFlow inference and does not use any paid API."
)

with st.sidebar:
    st.header("Model Settings")

    if HF_MODEL_URL:
        st.success("Using Hugging Face direct model URL")
        st.caption("Public `.keras` / `.h5` model loaded without API key.")

        custom_preprocess = st.selectbox(
            "Custom model preprocessing",
            ["Scale 0 to 1", "Scale -1 to 1", "Raw 0 to 255"],
            index=0
        )
    else:
        st.info("Using TensorFlow MobileNetV2 default model")
        st.caption(
            "For strict Hugging Face usage, upload a TensorFlow/Keras model "
            "to a public HF repo and set HF_MODEL_URL."
        )
        custom_preprocess = "Scale -1 to 1"

    st.divider()
    st.write("Allowed files: JPG, JPEG, PNG, WEBP")


try:
    model, model_type = load_ai_model()
    labels = load_labels()
    image_height, image_width = get_model_image_size(model)

    st.caption(f"Model input size: {image_width} x {image_height}")

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()

        st.image(
            file_bytes,
            caption="Uploaded Image",
            use_container_width=True
        )

        with st.spinner("AI is analyzing the image..."):
            image_batch = preprocess_image(
                file_bytes=file_bytes,
                height=image_height,
                width=image_width,
                model_type=model_type,
                custom_preprocess=custom_preprocess
            )

            if model_type == "mobilenetv2_imagenet":
                results = predict_mobilenetv2(model, image_batch)
            else:
                results = predict_custom_model(model, image_batch, labels)

        show_results(results)

    else:
        st.info("Please upload an image to start prediction.")

except Exception as error:
    st.error("App could not load or run the model.")
    st.exception(error)
