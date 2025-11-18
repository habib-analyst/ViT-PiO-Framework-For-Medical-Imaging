import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Load and preprocess data
def load_and_preprocess_data(data_path):
    # Load the .npy files
    X_train = np.load(os.path.join(data_path, "X_train.npy"))
    y_train = np.load(os.path.join(data_path, "y_train.npy"))
    X_val = np.load(os.path.join(data_path, "X_validation.npy"))
    y_val = np.load(os.path.join(data_path, "y_validation.npy"))
    X_test = np.load(os.path.join(data_path, "X_test.npy"))
    y_test = np.load(os.path.join(data_path, "y_test.npy"))

    # Ensure proper input shape and normalize
    for X in [X_train, X_val, X_test]:
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)
        if X.shape[-1] == 1:
            X = np.repeat(X, 3, axis=-1)
        X = X.astype('float16') / 255.0

    # Convert labels to integers if they're not already
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test

# Data Augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
])

# Patch encoding layer for ViT
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
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

# Perceiver IO block
def perceiver_io_block(x, num_heads=4, key_dim=64):
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    ffn = layers.Dense(key_dim * 4, activation="relu")(x)
    ffn = layers.Dense(x.shape[-1])(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x

# Build ViT + Perceiver IO Model
def build_model(input_shape=(224, 224, 3), patch_size=16, num_classes=2):
    inputs = layers.Input(shape=input_shape)
    
    # Apply data augmentation only during training
    x = layers.Lambda(lambda x: tf.keras.backend.in_train_phase(
        data_augmentation(x), x))(inputs)
    
    # Create patches
    patches = layers.Reshape((input_shape[0] // patch_size * input_shape[1] // patch_size, patch_size * patch_size * 3))(x)
    
    # ViT Encoder
    patch_encoder = PatchEncoder(num_patches=(input_shape[0] // patch_size) ** 2, projection_dim=64)
    encoded_patches = patch_encoder(patches)
    
    # Transformer Encoder
    for _ in range(4):
        encoded_patches = layers.MultiHeadAttention(num_heads=4, key_dim=64)(encoded_patches, encoded_patches)
        encoded_patches = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    # Perceiver IO blocks
    for _ in range(3):
        encoded_patches = perceiver_io_block(encoded_patches)
    
    # Classification head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    outputs = layers.Dense(num_classes, activation="softmax")(representation)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model



# Train the model
def train_model(X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
    model = build_model(input_shape=X_train.shape[1:])
    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks_list = [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7),
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return model, history

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_loss, test_acc

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('skin_disease_vit_perceiver_io_history.png')
    plt.close()

def plot_learning_curves(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Set the path to your data (relative path)
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "Skin Model", "NPYFile")

    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(data_path)

    # Print shapes to verify data loading
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Train the model
    model, history = train_model(X_train, y_train, X_val, y_val)

    # Plot learning curves
    plot_learning_curves(history)

    # Evaluate the model
    test_loss, test_acc = evaluate_model(model, X_test, y_test)

    # Plot training history
    plot_history(history)

    # Save the model
    model.save('skin_disease_vit_perceiver_io_model.h5')
    print("Model saved as 'skin_disease_vit_perceiver_io_model.h5'")

    # Print model summary
    model.summary()

