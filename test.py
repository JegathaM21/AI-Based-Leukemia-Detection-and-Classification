import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import os

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the pre-trained model with proper initialization
@st.cache_resource
def load_leukemia_model():
    try:
        model_path = r"C:\Users\91787\Desktop\ProjectLeukemia\leukemia_cnn_model.h5"
        model = tf.keras.models.load_model(model_path)
        
        # Initialize model by running a dummy prediction
        dummy_input = np.zeros((1, 224, 224, 3))
        _ = model.predict(dummy_input)
        
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Function to generate Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, layer_name="conv2d_2"):
    try:
        # Get the convolutional layer
        conv_layer = model.get_layer(layer_name)
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Pool gradients spatially
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize between 0-1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize to match original image
        heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
        return heatmap
        
    except Exception as e:
        st.error(f"Heatmap generation failed: {str(e)}")
        return None

# Function to process the image
def load_and_process_image(uploaded_file):
    try:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize
        return img_array, img
    except Exception as e:
        st.error(f"Image processing failed: {str(e)}")
        return None, None

# Function to overlay heatmap on image
def overlay_heatmap(heatmap, original_img, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original_img = cv2.cvtColor(np.uint8(original_img), cv2.COLOR_RGB2BGR)
    overlayed = cv2.addWeighted(jet, alpha, original_img, 1-alpha, 0)
    return cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)

# Class labels
CLASS_LABELS = ['ALL', 'AML', 'CLL', 'CML', 'Healthy']

# Main Streamlit UI
def main():
    st.set_page_config(page_title="Leukemia Detection", layout="wide")
    st.title("Leukemia Detection with Explainable AI (Grad-CAM)")
    st.write("Upload a blood smear image for classification and visualization")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        model = load_leukemia_model()
        if model is None:
            return
            
        img_array, original_img = load_and_process_image(uploaded_file)
        if img_array is None:
            return
            
        # Display original image
        st.image(original_img, caption="Uploaded Image", use_column_width=True)
        
        # Model prediction
        prediction = model.predict(img_array)
        predicted_class = CLASS_LABELS[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Display prediction
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Prediction:** {predicted_class}")
            st.metric("Confidence", f"{confidence:.2f}%")
            
        # Generate Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, model)
        if heatmap is not None:
            # Display heatmap and overlay
            with col2:
                st.subheader("Grad-CAM Heatmap")
                fig, ax = plt.subplots()
                ax.imshow(heatmap, cmap='jet')
                ax.axis('off')
                st.pyplot(fig)
                
            st.subheader("Overlay Visualization")
            overlayed = overlay_heatmap(heatmap, original_img)
            st.image(overlayed, use_column_width=True)

if __name__ == "__main__":
    main()