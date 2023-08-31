# CodeClauseInternship_mask-detection
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Define paths to your dataset
dataset_dir = 'path_to_dataset'
with_mask_dir = os.path.join(dataset_dir, 'with_mask')
without_mask_dir = os.path.join(dataset_dir, 'without_mask')

# Load and preprocess images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img)
    return images

with_mask_images = load_images_from_folder(with_mask_dir)
without_mask_images = load_images_from_folder(without_mask_dir)

# Create labels
with_mask_labels = np.ones(len(with_mask_images))
without_mask_labels = np.zeros(len(without_mask_images))

# Combine images and labels
images = np.array(with_mask_images + without_mask_images)
labels = np.concatenate([with_mask_labels, without_mask_labels])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
num_epochs = 10
model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

# Generate classification report
predictions = model.predict(X_test)
predicted_labels = np.round(predictions)
report = classification_report(y_test, predicted_labels)
print(report)
