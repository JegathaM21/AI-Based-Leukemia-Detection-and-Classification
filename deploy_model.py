import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = tf.keras.models.load_model(r"C:\Users\91787\Desktop\ProjectLeukemia\leukemia_cnn_model.h5")

# Class labels used in training (must match training order)
class_labels = ['ALL', 'AML', 'CLL', 'CML', 'Healthy']

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    r"C:\Users\91787\Desktop\ProjectLeukemia\DataSet\test",
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Make predictions
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Classification Report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Print predicted labels for each image
for i, pred_class_index in enumerate(predicted_classes):
    filename = test_generator.filenames[i]
    class_name = class_labels[pred_class_index]
    print(f"Image: {filename} --> Predicted Class: {class_name}")
