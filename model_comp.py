import matplotlib.pyplot as plt

# Structure of the data:
# "model": (time to run (in minute), accuracy)
data = {
    "ResNet18": (8, 0.57),
    "Inception 32": (13, 0.61),
    "Inception 64": (15, 0.64),
    "Inception 128": (16, 0.64)
}

# Extracting the data for plotting
models = list(data.keys())
times = [data[model][0] for model in models]
accuracies = [data[model][1] for model in models]

import numpy as np

# Model vs training time per epoch
plt.figure(figsize=(10, 6))
bar_colors = ['#ff005f' if model == "ResNet18" else '#006fc4' for model in models]  # Different color for ResNet18
bars = plt.bar(models, times, color=bar_colors)

# Add a horizontal dotted line from the top of the ResNet18 bar
resnet_index = models.index("ResNet18")
resnet_height = times[resnet_index]
plt.axhline(y=resnet_height, color='#ff005f', linestyle='dotted')

plt.xlabel("Model")
plt.ylabel("Training Time per Epoch (minutes)")
plt.title("Model vs Training Time per Epoch")
plt.savefig("model_vs_time.pdf")
plt.show()

# Model vs accuracy
plt.figure(figsize=(10, 6))
bar_colors = ['#ff005f' if model == "ResNet18" else '#006fc4' for model in models]  # Different color for ResNet18
bars = plt.bar(models, accuracies, color=bar_colors)

# Add a horizontal dotted line from the top of the ResNet18 bar
resnet_height = accuracies[resnet_index]
plt.axhline(y=resnet_height, color='#ff005f', linestyle='dotted')

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model vs Accuracy")
plt.savefig("model_vs_accuracy.pdf")
plt.show()