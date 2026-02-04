import tensorflow as tf
import numpy as np
import cv2 # Using OpenCV for image manipulation
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model # Still need Model class
import matplotlib.pyplot as plt # For displaying images (optional)
import os # To check if files exist

# --- Configuration ---
# **IMPORTANT:** Replace these with your actual paths
MODEL_PATH = r'C:\Users\91787\Desktop\ProjectLeukemia\leukemia_cnn_model.h5' # <-- Change this to the path of your saved model
IMAGE_PATH = r'C:\Users\91787\Desktop\ProjectLeukemia\DataSet\test\ALL_TEST-20230225T082325Z-001\ALL TEST\TEST 20 744_original_20190114_142815.jpg_b9c6424f-6d0a-431c-a9cf-2718e65e12a7.jpg' # <-- Change this to the path of your input image

# Target layer for Grad-CAM. Should be a Conv2D layer name from your model summary.
# 'conv2d_2' is the last Conv2D layer in your provided summary.
TARGET_LAYER_NAME = 'conv2d_2'

# Target class index for visualization.
# If None, the class with the highest prediction score will be used.
# If you want to see why the model predicted class 0 (assuming it's the first class), set TARGET_CLASS_INDEX = 0
TARGET_CLASS_INDEX = None

# Image dimensions your model expects
IMAGE_SIZE = (224, 224)

# Output path for the superimposed image
OUTPUT_PATH = 'grad_cam_output.jpg'

# --- 1. Validate File Paths ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image file not found at {IMAGE_PATH}")
    exit()

# --- 2. Load the Model ---
try:
    # Load the model.
    # WARNING:absl message about compiled metrics is normal when loading untrained/uncompiled models
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    # Optional: Print model summary to verify structure and layer names
    # model.summary()

    # --- IMPORTANT: Compile the model after loading ---
    # This helps finalize the state needed.
    if model.output_shape[-1] > 1:
         model.compile(optimizer='adam', loss='categorical_crossentropy')
    else: # Assuming binary classification with 1 output neuron (e.g., sigmoid)
         model.compile(optimizer='adam', loss='binary_crossentropy')
    print("Model compiled after loading.")

    # *** Perform a dummy forward pass after loading/compiling ***
    # This is intended to build the model's graph and potentially help,
    # although it hasn't fully resolved the .output access issue for Model() constructor yet.
    try:
        # Create a dummy input tensor matching the expected shape and dtype
        dummy_input = tf.zeros((1, *IMAGE_SIZE, 3), dtype=tf.float32) # Use correct dtype if needed

        # Perform a forward pass. The result is discarded.
        _ = model(dummy_input)
        print("Dummy forward pass completed after loading/compiling to build model fully.")

    except Exception as e:
        print(f"Error during dummy forward pass after loading/compiling: {e}")
        # If this fails, the model structure itself might be fundamentally broken upon loading.
        import traceback
        traceback.print_exc()
        # Note: We won't exit here, as we'll try a different method to get symbolic tensors.


except Exception as e:
    print(f"Error during model loading or initial compilation: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- 3. Prepare the Image (This is the actual image processing) ---
try:
    # Load the image
    img = image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
    # Convert image to a NumPy array
    img_array = image.img_to_array(img)
    # Add a batch dimension (required by model)
    img_array = np.expand_dims(img_array, axis=0)

    # --- IMPORTANT PREPROCESSING ---
    # Add any specific preprocessing your model requires here.
    # Based on your architecture summary, 0-255 input is likely okay.
    # If your model expected scaled input, add: img_array /= 255.0

    print(f"Image loaded and preprocessed: {IMAGE_PATH}")
except Exception as e:
    print(f"Error loading or preprocessing image from {IMAGE_PATH}: {e}")
    exit()


# --- 4. Define the Grad-CAM generation function (Manual Symbolic Tracing) ---
# This function will manually trace the symbolic tensors and define the intermediate model.
def generate_grad_cam(model, img_array, layer_name, target_class_index=None):
    """Generates a Grad-CAM heatmap for a given image and model using manual symbolic tracing.

    Args:
        model: The Keras model (already loaded, compiled, and dummy-called).
        img_array: The input image array (with batch dimension, shape (1, H, W, C)).
        layer_name: The name of the target convolutional layer.
        target_class_index: The index of the target class. If None, the
                            predicted class is used.

    Returns:
        A normalized 2D NumPy array representing the heatmap (0-1), or None if an error occurs.
    """
    print(f"Attempting Grad-CAM for layer: '{layer_name}'")
    try:
        # Get the target layer instance
        target_layer = model.get_layer(layer_name)
        if not isinstance(target_layer, tf.keras.layers.Conv2D):
            print(f"Error: Layer '{layer_name}' is not a Conv2D layer.")
            # List available Conv2D layers for debugging
            print("\nAvailable Conv2D layers in the model:")
            found_conv = False
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                     print(f"- {layer.name}")
                     found_conv = True
            if not found_conv:
                print("No Conv2D layers found in the model.")
            return None

        # --- Create an intermediate model using manual symbolic tracing ---
        # *** FIX Attempt 8: Manually build symbolic path up to target and final layers ***
        # This bypasses accessing model.inputs and model.output directly from the
        # loaded model object which seems to be problematic.
        try:
            # Create a *new* symbolic input tensor with the correct shape and dtype
            # Use the input shape/dtype from the loaded model, excluding the batch dimension
            if isinstance(model.inputs, list):
                if len(model.inputs) > 1:
                    print("Warning: Model has multiple inputs. Manual tracing might be complex and may not work.")
                    # This approach might not work well for multi-input models.
                input_shape_without_batch = model.inputs[0].shape[1:]
                input_dtype = model.inputs[0].dtype
            else: # Single input model
                input_shape_without_batch = model.inputs.shape[1:]
                input_dtype = model.inputs.dtype


            new_input_tensor = tf.keras.Input(shape=input_shape_without_batch, dtype=input_dtype)

            # Trace the symbolic tensor through the layers of the *loaded* model
            x = new_input_tensor # Start tracing from the new input tensor
            intermediate_output_tensor = None
            final_output_tensor = None

            # Iterate through the layers of the *loaded* model
            for i, layer in enumerate(model.layers):
                # Apply the layer from the *loaded* model to the current symbolic tensor 'x'
                # This creates a new symbolic tensor representing the output of this layer
                x = layer(x) # This links the layers together symbolically

                # Check if this is the target layer's output tensor
                if layer.name == layer_name:
                    intermediate_output_tensor = x # Capture the tensor *after* the target layer

                # Check if this is the last layer (the final output layer)
                if i == len(model.layers) - 1: # Check if it's the last layer by index
                     final_output_tensor = x # Capture the tensor *after* the last layer


            if intermediate_output_tensor is None:
                 print(f"Error: Could not find symbolic output tensor for layer '{layer_name}' during manual tracing.")
                 return None
            if final_output_tensor is None:
                 print("Error: Could not find symbolic output tensor for the final layer during manual tracing.")
                 return None


            # Now, define the intermediate model using the *new* symbolic input tensor
            # and the symbolic output tensors we manually traced.
            intermediate_model = Model(
                inputs=new_input_tensor, # Use the newly created symbolic input tensor
                outputs=[intermediate_output_tensor, final_output_tensor] # Use the traced symbolic outputs
            )
            print("Intermediate model created using manual symbolic tracing.")

        except Exception as e:
             print(f"Error during manual symbolic tracing or intermediate model creation: {e}")
             # Print traceback for intermediate model creation error
             import traceback
             traceback.print_exc()
             return None


        # Use tf.GradientTape to record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Perform the forward pass using the *intermediate* model
            # This call executes the graph defined by intermediate_model with the actual img_array.
            # It produces CONCRETE tensors for conv_layer_output and predictions that are tracked by the tape.
            results = intermediate_model(img_array)
            conv_layer_output = results[0] # First output is the traced target layer output
            predictions = results[1] # Second output is the traced final model output


            # Determine the target class index for visualization
            if target_class_index is None:
                # Use the class with the highest prediction score
                if predictions.shape[-1] <= 1: # Handle binary/single output case
                     print(f"Warning: Model output shape is {predictions.shape[-1]}. Cannot reliably determine class index for Grad-CAM from prediction alone.")
                     print("Please explicitly set TARGET_CLASS_INDEX for binary/single output models (0 or 1).")
                     return None

                target_class_index = tf.argmax(predictions[0])
                print(f"Grad-CAM: Using predicted class index: {target_class_index.numpy()}")
            else:
                 # Use the specified target class index
                 print(f"Grad-CAM: Using specified target class index: {target_class_index}")
                 # Ensure the index is within the valid range of the model's output classes
                 if not (0 <= target_class_index < predictions.shape[-1]):
                      print(f"Error: Specified TARGET_CLASS_INDEX ({target_class_index}) is out of valid range (0 to {predictions.shape[-1]-1}).")
                      return None

            # Get the loss/score for the target class from the predictions tensor
            target_class_score = predictions[:, target_class_index]


        # Compute the gradient of the target class score with respect to the output of the target layer tensor
        # This should now work because conv_layer_output is a concrete tensor from the intermediate_model call within the tape.
        grads = tape.gradient(target_class_score, conv_layer_output)

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Get the feature map for the specific image (batch size 1)
        feature_map = conv_layer_output[0] # Shape (height, width, channels)

        # Weighted combination of feature maps and gradients
        heatmap = feature_map @ pooled_grads[..., tf.newaxis] # Shape (height, width, 1)
        heatmap = tf.squeeze(heatmap, axis=-1) # Shape (height, width)


        # Apply ReLU
        heatmap = tf.maximum(heatmap, 0)

        # Normalize the heatmap
        max_val = tf.reduce_max(heatmap)
        if tf.equal(max_val, 0): # Use tf.equal for tensor comparison in TensorFlow
            print("Grad-CAM Warning: Max heatmap value is 0. Cannot normalize. Heatmap will be all zeros.")
            heatmap = heatmap * 0
        else:
            heatmap /= max_val

        print("Raw heatmap generated (0-1).")
        return heatmap.numpy() # Return as NumPy array

    except Exception as e:
        print(f"Error during Grad-CAM generation function: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Main execution flow ---

# ... (Image preparation is above) ...

# --- Generate Heatmap ---
# Call the function to generate the heatmap
# The dummy forward pass before this call, combined with manual symbolic tracing inside,
# is intended to resolve the issue.
heatmap = generate_grad_cam(model, img_array, TARGET_LAYER_NAME, TARGET_CLASS_INDEX)

# Check if heatmap generation was successful
if heatmap is None:
    print("Grad-CAM heatmap generation failed. Exiting.")
    exit()

# --- 5. Superimpose Heatmap on Original Image ---

# Resize the heatmap to the original image dimensions (IMAGE_SIZE)
# heatmap is already a NumPy array from the function
# OpenCV expects (width, height) for size
heatmap_resized = cv2.resize(heatmap, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

# Convert to color map
# Scale to 0-255, then apply JET colormap
heatmap_colored = np.uint8(255 * heatmap_resized)
colormap = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

# Load the original image again using OpenCV for easier superimposition
# Read the image in color (3 channels). Use IMREAD_COLOR for certainty.
original_img_cv = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
# Resize original image to match the target size used for the model
original_img_cv = cv2.resize(original_img_cv, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

# Superimpose the heatmap on the original image
# cv2.addWeighted(src1, alpha, src2, beta, gamma)
# src1: original image, alpha: weight of original image
# src2: colormap heatmap, beta: weight of heatmap
# gamma: scalar added to each sum (usually 0)
# Weights (0.6 and 0.4) can be adjusted for desired transparency
superimposed_img = cv2.addWeighted(original_img_cv, 0.6, colormap, 0.4, 0)

# --- 6. Display or Save the Result ---

# Save the superimposed image
try:
    cv2.imwrite(OUTPUT_PATH, superimposed_img)
    print(f"Superimposed Grad-CAM image saved to {OUTPUT_PATH}")
except Exception as e:
    print(f"Error saving output image to {OUTPUT_PATH}: {e}")

# Optional: Display the images using matplotlib (requires a GUI environment or notebook)
# try:
#     # Convert images to RGB for matplotlib if using OpenCV (BGR)
#     original_rgb = cv2.cvtColor(original_img_cv, cv2.COLOR_BGR2RGB)
#     superimposed_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 3, 1)
#     plt.imshow(original_rgb)
#     plt.title("Original Image")
#     plt.axis('off')

#     plt.subplot(1, 3, 2)
#     # Heatmap is grayscale (single channel); specify cmap
#     plt.imshow(heatmap_resized, cmap='jet') # Use resized heatmap
#     plt.title("Heatmap (Normalized)")
#     plt.axis('off')

#     plt.subplot(1, 3, 3)
#     plt.imshow(superimposed_rgb)
#     plt.title("Grad-CAM")
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()
# except Exception as e:
#     print(f"Could not display images with matplotlib: {e}")


# Optional: Display using OpenCV (requires a GUI environment)
# try:
#     cv2.imshow("Original", original_img_cv)
#     cv2.imshow("Grad-CAM", superimposed_img)
#     cv2.waitKey(0) # Wait indefinitely until a key is pressed
#     cv2.destroyAllWindows() # Close all OpenCV windows
# except Exception as e:
#      print(f"Could not display images with OpenCV: {e}")