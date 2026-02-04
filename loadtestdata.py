from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    r"C:\Users\91787\Desktop\ProjectLeukemia\DataSet\test",
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

print("Detected classes:", test_generator.class_indices)
print("Total test images:", test_generator.samples)
