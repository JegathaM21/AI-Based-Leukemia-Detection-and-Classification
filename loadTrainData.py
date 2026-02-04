import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow info/warnings

from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = r"C:\Users\91787\Desktop\ProjectLeukemia\DataSet\demopro"
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)
