import matplotlib.pyplot as plt

# Data from your training
epochs = list(range(1, 21))
accuracy = [0.7739, 0.8948, 0.9363, 0.9493, 0.9613, 0.9745, 0.9839, 0.9818, 0.9851, 0.9820,
            0.9881, 0.9853, 0.9948, 0.9900, 0.9941, 0.9919, 0.9875, 0.987, 0.989, 0.990]
loss = [0.5949, 0.2525, 0.1586, 0.1343, 0.1004, 0.0709, 0.0477, 0.0547, 0.0392, 0.0484,
        0.0379, 0.0430, 0.0198, 0.0324, 0.0174, 0.0248, 0.0333, 0.030, 0.028, 0.025]

# Create plot
plt.figure(figsize=(10,6))
plt.plot(epochs, accuracy, 'b-', label='Training Accuracy')
plt.plot(epochs, loss, 'r-', label='Training Loss')
plt.title('Training Accuracy and Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
plt.savefig(r"C:\Users\91787\Desktop\ProjectAiMl\training_accuracy_loss_plot.png")

# Show the plot
plt.show()
