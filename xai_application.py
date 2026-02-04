import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Load model
model = load_model(r"C:\Users\91787\Desktop\ProjectLeukemia\leukemia_cnn_model.h5")
last_conv_layer_name = "conv2d_2"
class_names = ['ALL', 'AML', 'CLL', 'CML', 'Healthy']

# Grad-CAM function
def get_gradcam(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model([model.inputs], 
                       [model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradients
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply weights with conv layer output
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Display heatmap on original image
def display_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap_color * alpha + img
    return np.uint8(superimposed)

# Streamlit UI
st.title("Leukemia Detection XAI App")
uploaded_file = st.file_uploader("Upload a leukemia cell image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array_exp)
    pred_class = class_names[np.argmax(preds)]

    st.image(img, caption=f"Predicted: {pred_class}", use_column_width=True)
    st.write(f"**Confidence**: {np.max(preds)*100:.2f}%")

    heatmap = get_gradcam(img_array_exp, model, last_conv_layer_name)
    img_path = f"temp.jpg"
    img.save(img_path)
    superimposed_img = display_gradcam(img_path, heatmap)

    st.image(superimposed_img, caption="Grad-CAM Explanation", use_column_width=True)
