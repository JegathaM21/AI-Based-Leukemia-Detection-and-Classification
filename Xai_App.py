# Streamlit App with All Enhanced Features for Leukemia Detection

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import base64
import pandas as pd
from PIL import Image, ImageEnhance
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Constants
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leukemia_cnn_model.h5")
CLASS_LABELS = ['ALL', 'AML', 'CLL', 'CML', 'Healthy']
TARGET_SIZE = (224, 224)
CSV_LOG = "predictions_log.csv"

@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    dummy_input = np.zeros((1, *TARGET_SIZE, 3), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)
    return model

def process_image(img):
    img = img.resize(TARGET_SIZE)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

def predict(img_array, model):
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    return CLASS_LABELS[index], float(prediction[0][index]), prediction

def make_gradcam(img_array, model, layer_name='conv2d_2'):
    input_tensor = tf.keras.Input(shape=model.input_shape[1:], dtype=tf.float32)
    x = input_tensor
    conv_out, final_out = None, None
    for layer in model.layers:
        x = layer(x)
        if layer.name == layer_name:
            conv_out = x
        final_out = x
    grad_model = tf.keras.Model(inputs=input_tensor, outputs=[conv_out, final_out])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-10
    heatmap = cv2.resize(heatmap.numpy(), TARGET_SIZE)
    return heatmap

def overlay_heatmap(heatmap, img, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    bgr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(jet, alpha, bgr_img, 1 - alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

def save_pdf_report(img_pil, prediction, confidence):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "Leukemia Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 730, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, 710, f"Prediction: {prediction}")
    c.drawString(50, 690, f"Confidence: {confidence:.2f}%")
    
    image_path = "temp_image.jpg"
    img_pil.save(image_path)
    c.drawImage(image_path, 50, 400, width=300, height=300)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def enhance_image(image_pil, brightness=1.0, contrast=1.0):
    enhancer = ImageEnhance.Brightness(image_pil)
    image_pil = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(image_pil)
    return enhancer.enhance(contrast)

def main():
    st.set_page_config(page_title="Leukemia Detection AI", layout="wide")
    st.title("🧬 Leukemia Detection with Explainable AI")

    model = load_model()
    
    uploaded_files = st.file_uploader("Upload blood smear images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    use_webcam = st.checkbox("📷 Capture from webcam instead", value=False)

    if use_webcam:
        img_data = st.camera_input("Take a picture")
        if img_data:
            uploaded_files = [img_data]

    if uploaded_files:
        brightness = st.slider("🔆 Brightness", 0.5, 2.0, 1.0)
        contrast = st.slider("🎨 Contrast", 0.5, 2.0, 1.0)

        if not os.path.exists(CSV_LOG):
            df_log = pd.DataFrame(columns=['Timestamp', 'Filename', 'Prediction', 'Confidence'])
        else:
            df_log = pd.read_csv(CSV_LOG)

        for file in uploaded_files:
            img_pil = Image.open(file)
            img_pil = enhance_image(img_pil, brightness, contrast)
            img_array, display_img = process_image(img_pil)
            pred_class, confidence, raw_preds = predict(img_array, model)
            heatmap = make_gradcam(img_array, model)
            overlay = overlay_heatmap(heatmap, display_img)

            st.subheader(f"📌 Result for: {file.name if hasattr(file, 'name') else 'Webcam Capture'}")
            col1, col2 = st.columns(2)
            with col1:
                st.image(display_img, caption="Original Image", use_column_width=True)
            with col2:
                st.image(overlay, caption=f"Grad-CAM: {pred_class}", use_column_width=True)

            st.success(f"Prediction: {pred_class} ({confidence*100:.2f}% confidence)")

            # Save prediction to CSV log
            df_log = pd.concat([
                df_log,
                pd.DataFrame({
                    'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Filename': [file.name if hasattr(file, 'name') else 'webcam_capture'],
                    'Prediction': [pred_class],
                    'Confidence': [confidence]
                })
            ], ignore_index=True)

            # PDF download
            pdf_buf = save_pdf_report(display_img, pred_class, confidence * 100)
            b64 = base64.b64encode(pdf_buf.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="Leukemia_Report.pdf">🧾 Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

        df_log.to_csv(CSV_LOG, index=False)
        st.info(f"✅ Predictions logged to `{CSV_LOG}`.")

if __name__ == '__main__':
    main()
