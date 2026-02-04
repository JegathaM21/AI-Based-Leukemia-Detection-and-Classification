from tensorflow.keras.models import load_model

model = load_model(r'C:\Users\91787\Desktop\ProjectLeukemia\leukemia_cnn_model.h5')  # Replace with your actual model path

# Print all layer names and types
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} ({layer.__class__.__name__})")
