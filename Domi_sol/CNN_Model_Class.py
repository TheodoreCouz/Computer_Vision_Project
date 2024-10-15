# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import time

# 1. Dataset class to include images and physical features
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = str(self.data.iloc[idx]['id'])
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        stable_height = int(self.data.iloc[idx]['stable_height']) - 1  # Convert to zero-based class index
        block_type = self.data.iloc[idx]['type']
        cam_angle = self.data.iloc[idx]['cam_angle']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([block_type, cam_angle], dtype=torch.float32), torch.tensor(stable_height)

# 2. Custom ResNet with additional features for classification
class CustomResNetWithFeatures(nn.Module):
    def __init__(self, num_types, num_classes):
        super(CustomResNetWithFeatures, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)  # Extract visual features

        self.type_embedding = nn.Embedding(num_types, 10)  # Size of 10 for block type embedding
        self.fc1 = nn.Linear(128 + 10 + 1, 64)  # Combine image features, block type embedding, and cam_angle
        self.fc2 = nn.Linear(64, num_classes)  # Output for classification (logits for each class)

    def forward(self, x, type_idx, cam_angle):
        x = self.resnet(x)
        type_embed = self.type_embedding(type_idx)
        combined = torch.cat((x, type_embed, cam_angle.unsqueeze(1)), dim=1)  # Concatenate visual and physical features
        x = torch.relu(self.fc1(combined))
        logits = self.fc2(x)  # Output logits for each class (stable height)
        return logits

# 3. Training function
def train_model(csv_file_path, images_dir_path, pth_model_file_path, num_epochs=12, batch_size=16, num_classes=6):
    # Define image transformations
    transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
    ])

    # Initialize the dataset and dataloader
    dataset = CustomImageDataset(csv_file_path, images_dir_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get the number of block types for embedding
    unique_block_types = dataset.data['type'].unique()
    num_types = len(unique_block_types)

    # Initialize the model, loss function, and optimizer
    model = CustomResNetWithFeatures(num_types, num_classes)
    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        start_epoch_time = time.time()

        for images, physical_features, stable_heights in dataloader:
            images = images.to(device)

            block_type = (physical_features[:, 0] - 1).long().to(device)  # Adjust for zero-based indexing
            cam_angle = physical_features[:, 1].to(device)

            stable_heights = stable_heights.to(device)  # Already zero-based class labels

            optimizer.zero_grad()

            try:
                logits = model(images, block_type, cam_angle)  # Get logits from model

                # Calculate loss
                loss = criterion(logits, stable_heights)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Calculate accuracy
                _, predicted_classes = torch.max(logits, 1)  # Get predicted class (stable height)
                running_correct += (predicted_classes == stable_heights).sum().item()  # Count correct predictions

            except Exception as e:
                print(f"Error during model forward pass: {e}")
                print(f'Images shape: {images.shape}, Block type: {block_type}, Cam angle: {cam_angle}, Stable heights: {stable_heights}')
                raise

        end_epoch_time = time.time()
        accuracy = running_correct / len(dataloader.dataset)  # Calculate accuracy
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader.dataset):.4f}, Accuracy: {accuracy:.4f}, Time: {end_epoch_time - start_epoch_time:.2f}s')

        # Save the model after each epoch
        torch.save({'model_state_dict': model.state_dict()}, pth_model_file_path)
        print(f"Model saved after Epoch {epoch+1} to {pth_model_file_path}")

# Example Usage
if __name__ == "__main__":
    csv_file_path = 'train.csv'  # Path to your training CSV
    images_dir_path = 'train'  # Path to your training images
    model_dir_path = "block_height_model_class2.pth"
    train_model(csv_file_path, images_dir_path, model_dir_path, num_classes=6)  # Assume stable heights range from 1 to 6
