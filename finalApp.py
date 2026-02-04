import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageEnhance
from tensorflow.keras.preprocessing.image import img_to_array
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import io
import tempfile
import uuid
import csv

# Constants
MODEL_PATH = "leukemia_cnn_model.h5"
CLASS_LABELS = ['ALL', 'AML', 'CLL', 'CML', 'Healthy']
TARGET_SIZE = (224, 224)
RESULTS_CSV = "diagnosis_results.csv"

@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    dummy_input = np.zeros((1, *TARGET_SIZE, 3), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)
    return model

def preprocess_image(image):
    image = image.resize(TARGET_SIZE)
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_gradcam_heatmap(img_array, model, layer_name=None):
    if layer_name is None:
        conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        layer_name = conv_layers[-1].name

    input_tensor = tf.keras.Input(shape=(224, 224, 3))
    x = input_tensor
    conv_output_tensor, final_output_tensor = None, None
    for i, layer in enumerate(model.layers):
        x = layer(x)
        if layer.name == layer_name:
            conv_output_tensor = x
        if i == len(model.layers) - 1:
            final_output_tensor = x

    grad_model = tf.keras.models.Model(inputs=input_tensor, outputs=[conv_output_tensor, final_output_tensor])

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = tf.zeros_like(heatmap) if max_val == 0 else heatmap / max_val
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, TARGET_SIZE[::-1])
    return heatmap

def overlay_heatmap(heatmap, image, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    original = cv2.cvtColor(np.array(image.resize(TARGET_SIZE)), cv2.COLOR_RGB2BGR)
    overlayed = cv2.addWeighted(heatmap_color, alpha, original, 1 - alpha, 0)
    return cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)

def plot_confidence(prediction):
    fig, ax = plt.subplots(figsize=(8, 4))
    indices = np.argsort(prediction[0])[::-1]
    scores = prediction[0][indices]
    labels = [CLASS_LABELS[i] for i in indices]
    ax.barh(labels, scores, color='skyblue')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("Confidence Scores")
    return fig

def save_bulk_pdf(reports):
    pdf_path = os.path.join(tempfile.gettempdir(), f"bulk_report_{uuid.uuid4().hex}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    for report in reports:
        c.setFont("Helvetica", 12)
        c.drawString(50, 750, f"Leukemia Diagnostic Report")
        c.drawString(50, 730, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(50, 700, f"Prediction: {report['prediction']}")
        c.drawString(50, 680, f"Confidence: {report['confidence']:.2f}%")
        temp_img_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.jpg")
        report['image'].save(temp_img_path)
        c.drawImage(temp_img_path, 50, 400, width=200, height=200)
        c.showPage()
    c.save()
    return pdf_path

def save_result_csv(filename, predicted_class, confidence):
    with open(RESULTS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, predicted_class, f"{confidence:.2f}%", datetime.now()])

def enhance_image(image, brightness=1.0, contrast=1.0):
    enhancer_b = ImageEnhance.Brightness(image)
    image = enhancer_b.enhance(brightness)
    enhancer_c = ImageEnhance.Contrast(image)
    return enhancer_c.enhance(contrast)

def main():
    st.set_page_config(page_title="Leukemia Detection", layout="wide")
    st.title("🩸 Leukemia Detection with Explainable AI")

    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Leukemia+XAI", width=150)
        st.header("About")
        st.markdown("""
        This app analyzes blood smear images to classify leukemia types using CNN and explains the predictions via Grad-CAM.

        **Supported Types:**
        - ALL, AML, CLL, CML, Healthy

        **Features:**
        - Upload or capture image via webcam
        - Enhance contrast/brightness
        - View Grad-CAM and overlay
        - Export PDF report (supports batch)
        """)

    model = load_model()

    st.subheader("Step 1: Select Input Images")
    input_method = st.radio("Input Method", ["Upload", "Webcam"])

    uploaded_images = []

    if input_method == "Upload":
        uploaded_files = st.file_uploader("Upload blood smear image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        for file in uploaded_files:
            img = Image.open(file).convert("RGB")
            uploaded_images.append((file.name, img))
    else:
        picture = st.camera_input("Capture from webcam")
        if picture:
            img = Image.open(picture).convert("RGB")
            uploaded_images.append(("webcam_image", img))

    if uploaded_images:
        st.subheader("Step 2: Enhance Images")
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)

        reports = []

        for filename, img in uploaded_images:
            enhanced_img = enhance_image(img, brightness, contrast)
            st.image(enhanced_img, caption=f"Enhanced: {filename}", use_column_width=True)

            img_array = preprocess_image(enhanced_img)
            prediction = model.predict(img_array, verbose=0)
            pred_index = np.argmax(prediction)
            pred_class = CLASS_LABELS[pred_index]
            confidence = prediction[0][pred_index] * 100

            st.success(f"Prediction for {filename}: {pred_class} ({confidence:.2f}% confidence)")
            st.pyplot(plot_confidence(prediction))

            heatmap = make_gradcam_heatmap(img_array, model)
            overlayed = overlay_heatmap(heatmap, enhanced_img)

            st.subheader("AI Explanation")
            col1, col2 = st.columns(2)
            with col1:
                st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True, clamp=True)
            with col2:
                st.image(overlayed, caption="Diagnostic Overlay", use_column_width=True)

            st.markdown(f"""
            **Interpretation for layer:**
            - **Red regions:** Most influential areas for the prediction
            - **Blue regions:** Minimal contribution to diagnosis
            - **Intensity:** Relative importance of cellular features

            **Clinical correlation:**
            - Compare highlighted regions with known morphological features of {pred_class}
            - Check for consistency with other diagnostic markers
            """)

            reports.append({"image": enhanced_img, "prediction": pred_class, "confidence": confidence,"heatmap":heatmap,"overlayed":overlayed})
            save_result_csv(filename, pred_class, confidence)

        if st.button("📄 Download Combined PDF Report"):
            pdf_path = save_bulk_pdf(reports)
            with open(pdf_path, "rb") as f:
                st.download_button("Download All Reports as PDF", f, file_name="combined_leukemia_report.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
