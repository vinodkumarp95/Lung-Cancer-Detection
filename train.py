import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from PIL import Image

# Define paths to dataset
data_dir = 'lung_cancer_dataset'
normal_dir = os.path.join(data_dir, 'normal')
cancer_dir = os.path.join(data_dir, 'cancer')

# Load images and labels
def load_images_from_folder(folder, label, img_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('L')  # Ensure grayscale
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0  # Normalize to [0,1]
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    return images, labels

# Load normal and cancer images
normal_images, normal_labels = load_images_from_folder(normal_dir, 0)  # 0 = No Cancer
cancer_images, cancer_labels = load_images_from_folder(cancer_dir, 1)  # 1 = Cancer

# Combine data
X = np.array(normal_images + cancer_images)
y = np.array(normal_labels + cancer_labels)

# Reshape for CNN
X = X.reshape(-1, 64, 64, 1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          epochs=20,
          validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Save the model
if not os.path.exists('model'):
    os.makedirs('model')
model.save('model/lung_cancer_model.h5')
print("✅ Model saved successfully!")