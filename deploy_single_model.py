import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model(r"C:\Users\91787\Desktop\ProjectAiMl\leukemia_cnn_model.h5")

# Class labels used during training
class_labels = ['ALL', 'AML', 'CLL', 'CML', 'Healthy']

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(image_path):
    processed_img = preprocess_image(image_path)
    prediction = model.predict(processed_img)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return predicted_class, confidence

def upload_and_classify():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        predicted_class, confidence = predict_image(file_path)
        result_label.config(
            text=f"Prediction: {predicted_class} ({confidence * 100:.2f}% confidence)",
            fg="green" if confidence >= 0.80 else "orange"
        )

# Create GUI
root = tk.Tk()
root.title("Leukemia Detector")
root.geometry("400x500")

upload_button = Button(root, text="Upload Test Image", command=upload_and_classify, font=("Arial", 12))
upload_button.pack(pady=20)

panel = Label(root)
panel.pack(pady=10)

result_label = Label(root, text="Prediction Result", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
