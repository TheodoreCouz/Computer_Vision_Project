import pandas as pd
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

# Paths to the dataset
train_image_folder = 'COMP90086_2024_Project_train/train'
metadata_file = 'COMP90086_2024_Project_train/train.csv'

# Load the metadata CSV file
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


# Load training data
images, metadata_features, labels = load_images_and_metadata(train_image_folder, metadata)

# Split the data into training and validation sets
X_train_img, X_val_img, X_train_meta, X_val_meta, y_train, y_val = train_test_split(
    images, metadata_features, labels, test_size=0.2, random_state=42)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow.keras.applications import ResNet50

# Image model branch
input_img = Input(shape=(224, 224, 3), name='image_input')
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model
x = base_model(input_img)
x = Flatten()(x)

# Metadata model branch
input_meta = Input(shape=(3,), name='metadata_input')
meta = Dense(32, activation='relu')(input_meta)

# Concatenate image and metadata branches
combined = Concatenate()([x, meta])
combined = Dense(128, activation='relu')(combined)
combined = Dense(64, activation='relu')(combined)
output = Dense(1, name='output')(combined)  # Regression output for stable height

# Compile the model
model = Model(inputs=[input_img, input_meta], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Display the model architecture
model.summary()

# Train the model
history = model.fit(
    [X_train_img, X_train_meta], y_train,
    validation_data=([X_val_img, X_val_meta], y_val),
    epochs=20,  # Adjust epochs as needed
    batch_size=32,
    verbose=1
)

# Save the trained model
model.save('block_stable_height_predictor.h5')

# Evaluate the model
val_loss, val_mae = model.evaluate([X_val_img, X_val_meta], y_val)
print(f'Validation Loss: {val_loss}, Validation MAE: {val_mae}')

# Load test images and metadata similarly to training
# Assuming test images and metadata are in similar format
test_image_folder = 'COMP90086_2024_Project_test/test'
test_metadata_file = 'COMP90086_2024_Project_test/test.csv'
test_metadata = pd.read_csv(test_metadata_file)

# Load the test data
test_images, test_metadata_features, _ = load_images_and_metadata(test_image_folder, test_metadata)

# Run predictions
predictions = model.predict([test_images, test_metadata_features])

# Save predictions in the required format
submission = pd.DataFrame({
    'id': test_metadata['id'],
    'stable_height': predictions.flatten().astype(int)
})
submission.to_csv('predictions.csv', index=False)
