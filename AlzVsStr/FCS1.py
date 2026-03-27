import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load dataset
X_train = np.load("C:\\Users\\habib\\Downloads\\SkinD Model\\NPYFile\\X_train.npy")
y_train = np.load("C:\\Users\\habib\\Downloads\\SkinD Model\\NPYFile\\y_train.npy")
X_test = np.load("C:\\Users\\habib\\Downloads\\SkinD Model\\NPYFile\\X_test.npy")
y_test = np.load("C:\\Users\\habib\\Downloads\\SkinD Model\\NPYFile\\y_test.npy")
X_val = np.load("C:\\Users\\habib\\Downloads\\SkinD Model\\NPYFile\\X_validation.npy")
y_val = np.load("C:\\Users\\habib\\Downloads\\SkinD Model\\NPYFile\\y_validation.npy")

### 1️⃣ Check Shapes ###
print("\n✅ Dataset Shapes:")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

### 2️⃣ Check Class Balance ###
train_Tinea = np.sum(y_train == 1)
train_Melanoma = np.sum(y_train == 0)
test_Tinea = np.sum(y_test == 1)
test_Melanoma = np.sum(y_test == 0)
val_Tinea = np.sum(y_val == 1)
val_Melanoma = np.sum(y_val == 0)

print("\n✅ Class Distribution:")
print(f"Train - Tinea: {train_Tinea}, Melanoma: {train_Melanoma}")
print(f"Test  - Tinea: {test_Tinea}, Melanoma: {test_Melanoma}")
print(f"Val   - Tinea: {val_Tinea}, Melanoma: {val_Melanoma}")

### 3️⃣ Check Pixel Value Range ###
print("\n✅ Pixel Normalization Check:")
print(f"Train Min/Max: {X_train.min()} / {X_train.max()}")
print(f"Test Min/Max: {X_test.min()} / {X_test.max()}")
print(f"Validation Min/Max: {X_val.min()} / {X_val.max()}")

### 4️⃣ Check Data Alignment ###
print("\n✅ First 5 Labels in y_train:")
print(y_train[:5])  # Should show 1 for stroke and 0 for Alzheimer's

### 5️⃣ Check for Corrupt Images ###
corrupt_count = 0
for i in range(len(X_train)):
    try:
        img = Image.fromarray((X_train[i] * 255).astype('uint8'))
    except Exception as e:
        print(f"⚠️ Corrupt image at index {i}: {e}")
        corrupt_count += 1
print(f"\n✅ Total corrupt images: {corrupt_count}")

### 6️⃣ Visualize Sample Images ###
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(X_train[i], cmap='gray')
    label = "Tinea" if y_train[i] == 1 else "Melanoma"
    ax.set_title(f"Label: {label}")
    ax.axis("off")
plt.show()

print("\n✅ Data check completed! Ready for model training! 🚀")
