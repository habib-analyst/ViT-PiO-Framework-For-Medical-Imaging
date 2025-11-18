import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tensorflow import keras
from keras import layers, models
from keras.models import load_model
import tensorflow as tf
from keras.models import load_model

# Define data augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
], name='data_augmentation')

# Custom layer for loading the model
class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = keras.layers.Dense(units=projection_dim, dtype='float16')
        self.position_embedding = keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim, dtype='float16'
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config

# Load test data
def load_test_data():
    X_test = np.load("E:\\Models\\SkinD Model\\NPYFile\\X_test.npy").astype(np.float32) 
    y_test = np.load("E:\\Models\\SkinD Model\\NPYFile\\y_test.npy").astype(int)

    # Ensure images have the correct shape (height, width, 3)
    if X_test.shape[-1] == 1:  # Convert grayscale to RGB if necessary
        X_test = np.repeat(X_test, 3, axis=-1)

    return X_test, y_test

# Load test data
X_test, y_test = load_test_data()

# Load the trained model with custom objects
model = load_model(
    "E:\\Models\\SkinD Model\\Model\\skin_disease_vit_perceiver_io_model.h5",
    custom_objects={'PatchEncoder': PatchEncoder,  'data_augmentation': data_augmentation}  # Pass custom layer
)

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels (0 = Melanoma, 1 = Tinea)

# Class names
class_names = ['Melanoma', 'Tinea']

# Function to save predicted images
def save_predicted_images(X, y_true, y_pred, class_names, num_images=10, output_dir="E:\\Models\\Predicted_Skin"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 6))

    for i, class_label in enumerate([0, 1]):  # 0 = Melanoma, 1 = Tinea
        class_indices = np.where(y_true == class_label)[0]
        selected_indices = random.sample(list(class_indices), min(num_images, len(class_indices)))

        for j, idx in enumerate(selected_indices):
            axes[i, j].imshow(X[idx])
            axes[i, j].axis('off')
            true_label = class_names[y_true[idx]]
            pred_label = class_names[y_pred[idx]]
            color = 'green' if true_label == pred_label else 'red'
            axes[i, j].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "predicted_skin_images.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved predicted images to {output_path}")

# Save predicted images
save_predicted_images(X_test, y_test, y_pred_classes, class_names)
