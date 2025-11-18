import pandas as pd
import numpy as np
import cv2
import os

# Load CSV file (replace with your actual file)
csv_file = "C:\\Users\\habib\\Downloads\\SkinD Model\\skin_disease_dataset.csv"  # Make sure this file has 'filepath', 'split', 'label'
df = pd.read_csv(csv_file)

# Ensure column names are correct
df.columns = df.columns.str.strip()

# Define image size (adjust based on your model)
IMG_SIZE = (224, 224)  # Resize images to 224x224

# Function to load and preprocess images
def load_images(imagepath):
    images = []
    for path in imagepath:
        img = cv2.imread(path)  # Read image
        if img is None:
            print(f"Warning: Image at {path} could not be loaded!")
            continue
        img = cv2.resize(img, IMG_SIZE)  # Resize
        img = img / 255.0  # Normalize to [0,1]
        images.append(img)
    return np.array(images, dtype=np.float16)

# Function to extract data based on split type
def get_data(split_type):
    subset = df[df['split'] == split_type]  # Filter by 'test', 'train', or 'validation'
    return subset['imagepath'].values, subset['label'].values

# Function to process and save data separately
def save_data(split_type):
    # Load file paths and labels
    imagepath, labels = get_data(split_type)
    
    # Load and preprocess the data
    images = load_images(imagepath)
    
    # Save the data
    np.save(f"X_{split_type}.npy", images)
    np.save(f"y_{split_type}.npy", labels)
    print(f"✅ {split_type.capitalize()} data saved!")

# First, process and save the test data
save_data('test')

# Then, process and save the train data
save_data('train')

# Finally, process and save the validation data
save_data('validation')

print("✅ All data processed and saved successfully!")