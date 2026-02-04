import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === USER INPUT ===
MODEL_PATH = r"C:\Users\91787\Desktop\ProjectLeukemia\leukemia_cnn_model.h5"
TEST_FOLDER_PATH = r"C:\Users\91787\Desktop\ProjectLeukemia\DataSet\test"

# === Load the model ===
model = load_model(MODEL_PATH)

# === Load test data ===
datagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_directory(
    TEST_FOLDER_PATH,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# === Predict and evaluate ===
predictions = model.predict(test_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# === Confusion matrix ===
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# === Classification report ===
report = classification_report(y_true, y_pred, target_names=class_labels)
print("\nClassification Report:\n")
print(report)
