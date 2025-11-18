import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# Set global policy to mixed_float16
tf.keras.mixed_precision.set_global_policy('mixed_float16')

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

# Load and preprocess test data
def load_and_preprocess_data(data_path):
    X_test = np.load(f"{data_path}/X_test.npy")
    y_test = np.load(f"{data_path}/y_test.npy")

    # Ensure proper input shape
    if len(X_test.shape) == 3:
        X_test = np.expand_dims(X_test, axis=-1)
    if X_test.shape[-1] == 1:
        X_test = np.repeat(X_test, 3, axis=-1)

    # Convert labels to integers if they're not already
    y_test = y_test.astype(int)

    return X_test, y_test

# Load the saved model
def load_saved_model(model_path):
    return load_model(model_path, custom_objects={
        'PatchEncoder': PatchEncoder,
        'data_augmentation': data_augmentation
    })

# Make predictions
def make_predictions(model, X_test):
    return model.predict(X_test)

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

# Plot ROC curve
def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

    return roc_auc

# Main execution
if __name__ == "__main__":
    # Set the paths
    data_path = "C:\\Users\\habib\\Downloads\\Lung Model\\NPYF"
    model_path = "C:\\Users\\habib\\Downloads\\Lung Model\\Model\\vit_perceiver_io_model.h5"

    # Load and preprocess test data
    X_test, y_test = load_and_preprocess_data(data_path)

    # Load the saved model
    model = load_saved_model(model_path)

    # Make predictions
    y_pred_proba = make_predictions(model, X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Generate and plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # Generate and print classification report
    class_report = classification_report(y_test, y_pred, target_names=['Pneumonia', 'Lung Cancer'])
    print("Classification Report:")
    print(class_report)

    # Generate and plot ROC curve, calculate AUC
    roc_auc = plot_roc_curve(y_test, y_pred_proba)

    print(f"AUC Score: {roc_auc:.4f}")

    # Calculate and print additional metrics
    accuracy = np.mean(y_test == y_pred)
    precision = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_test == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")

    # Save classification report to a file
    with open('classification_report.txt', 'w') as f:
        f.write(class_report)

    print("Evaluation complete. Results saved as 'confusion_matrix.png', 'roc_curve.png', and 'classification_report.txt'.")

