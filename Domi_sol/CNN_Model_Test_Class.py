# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd

# Define the model (same as in the training script)
class CustomResNetWithFeatures(nn.Module):
    def __init__(self, num_types):
        super(CustomResNetWithFeatures, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # Using ResNet18 or ResNet50
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)  # Adjust the output size

        # Embedding for block type
        self.type_embedding = nn.Embedding(num_types, 10)  # Embedding for block types

        # Fully connected layers for additional features (classification)
        self.fc1 = nn.Linear(128 + 10 + 1, 64)  # Image features + type embedding + cam_angle (center of mass excluded)
        self.fc2 = nn.Linear(64, 6)  # Final output for stable height (6 classes for classification)

    def forward(self, x, type_idx, cam_angle):
        x = self.resnet(x)  # Pass through ResNet
        type_embed = self.type_embedding(type_idx)  # Embed block type

        # Ensure cam_angle has the correct shape
        cam_angle = cam_angle.unsqueeze(1)  # Shape should be (batch_size, 1)

        # Concatenate features
        combined = torch.cat((x, type_embed, cam_angle), dim=1)

        x = torch.relu(self.fc1(combined))  # Fully connected layer
        stable_height = self.fc2(x)  # Final output (classification logits for 6 classes)
        return stable_height


# Function to preprocess and load an image
def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 (or whatever size was used in training)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")  # Convert to RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Load the trained model from a checkpoint file
def load_model(pth_model_file_path, num_types):
    model = CustomResNetWithFeatures(num_types)  # Create the model
    checkpoint = torch.load(pth_model_file_path)
    model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights

    model.eval()  # Set the model to evaluation mode
    return model


# Function to generate predictions for test images and save to a CSV file
def generate_solution_csv(pth_model_file_path, test_images_dir, output_csv_file='sample_solution.csv', num_types=2):
    # Load the trained model
    model = load_model(pth_model_file_path, num_types)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    predictions = []

    # Iterate over test images
    for image_name in os.listdir(test_images_dir):
        img_id = os.path.splitext(image_name)[0]  # Extract image ID (without file extension)
        image_path = os.path.join(test_images_dir, image_name)

        # Preprocess the image
        image = load_and_preprocess_image(image_path).to(device)  # Move image to the correct device

        # Dummy input for block type and cam_angle (these need to be set appropriately)
        block_type = torch.tensor(1).to(device)  # Assuming block type 1
        cam_angle = torch.tensor(0.0).to(device)  # Assuming a default cam angle of 0.0

        with torch.no_grad():
            output = model(image, block_type.unsqueeze(0), cam_angle.unsqueeze(0))  # Forward pass
            predicted_class = output.argmax(dim=1).item() + 1  # Get the predicted class (1-6 for stable height)

        # Append prediction
        predictions.append({'id': img_id, 'stable height': predicted_class})

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(output_csv_file, index=False)
    print(f"Predictions saved to {output_csv_file}")


# Example usage
if __name__ == "__main__":
    pth_model_file_path = 'block_height_model_class2.pth'  # Path to your saved model file
    test_images_dir = 'test'  # Path to the test images directory
    generate_solution_csv(pth_model_file_path, test_images_dir, output_csv_file='sample_solution_classification2.csv', num_types=2)
