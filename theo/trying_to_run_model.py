# Load the model without loading the weights
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import pandas as pd
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

train_image_folder = 'COMP90086_2024_Project_train/train'
metadata_file = 'COMP90086_2024_Project_train/train.csv'
metadata = pd.read_csv(metadata_file)

def load_images_and_metadata(image_folder, metadata_df):
    images = []
    metadata_features = []
    labels = []
    
    for idx, row in metadata_df.iterrows():
        # Construct the image path
        img_path = os.path.join(image_folder, f'{row["id"]}.jpg')
        
        # Check if the image exists
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found.")
            continue
        
        # Load the image
        image = cv2.imread(img_path)
        
        # Check if the image is loaded correctly
        if image is None:
            print(f"Failed to load image {img_path}")
            continue
        
        # Resize the image
        image = cv2.resize(image, (224, 224))  # Resize to a fixed size
        image = img_to_array(image) / 255.0  # Normalize image
        
        # Append the image and metadata
        images.append(image)
        
        # Select relevant metadata features (e.g., shapeset, type, total_height, etc.)
        metadata_features.append([row['shapeset'], row['type'], row['total_height']])
        
        # Append the label (stable_height)
        labels.append(row['stable_height'])
    
    return np.array(images), np.array(metadata_features), np.array(labels)

# Let's assume the model requires input size 224x224x3
input_shape = (224, 224, 3)  # This must match the input size used during model training

# Create a new input layer
new_input = Input(shape=input_shape)
model_path = "/home/theo/Documents/Unif/Master/M2/Q1/Computer vision/Computer_Vision_Project/block_stable_height_predictor.h5"
# Try to load the model
try:
    model = load_model(model_path)
except ValueError as e:
    print(f"Error loading model: {e}")

# Load training data
images, metadata_features, labels = load_images_and_metadata(train_image_folder, metadata)

# Split the data into training and validation sets
X_train_img, X_val_img, X_train_meta, X_val_meta, y_train, y_val = train_test_split(
    images, metadata_features, labels, test_size=0.2, random_state=42)

# If you know the architecture of the model, you can redefine it
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Input
from tensorflow.keras.models import Model

# Define the same architecture as used in training
input_img = Input(shape=(224, 224, 3), name='image_input')
base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))

# Add layers as per the original model definition
x = base_model(input_img)
x = Flatten()(x)

# Define the metadata input branch
input_meta = Input(shape=(3,), name='metadata_input')
meta = Dense(32, activation='relu')(input_meta)

# Concatenate both branches
combined = Concatenate()([x, meta])
combined = Dense(128, activation='relu')(combined)
combined = Dense(64, activation='relu')(combined)
output = Dense(1, name='output')(combined)

# Create the final model
model = Model(inputs=[input_img, input_meta], outputs=output)

# Load the weights from the file
model.load_weights(model_path)

# Now you can evaluate the model or make predictions
val_loss, val_mae = model.evaluate([X_val_img, X_val_meta], y_val)
print(f'Validation Loss: {val_loss}, Validation MAE: {val_mae}')


