# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Dataset paths for Kaggle environment
part1_path = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1"
part2_path = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_2"
metadata_path = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv"

# Load metadata
metadata = pd.read_csv(metadata_path)

# Map image paths
def map_file_path(file_name):
    if os.path.exists(os.path.join(part1_path, file_name)):
        return os.path.join(part1_path, file_name)
    elif os.path.exists(os.path.join(part2_path, file_name)):
        return os.path.join(part2_path, file_name)
    else:
        return None

metadata['file_path'] = metadata['image_id'].apply(lambda x: map_file_path(f"{x}.jpg"))
metadata = metadata[metadata['file_path'].notnull()]

# Train/Val/Test Split
train_data, test_data = train_test_split(metadata, test_size=0.3, stratify=metadata['dx'], random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, stratify=test_data['dx'], random_state=42)

# Image Preprocessing (resize to 192x192)
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((192, 192))  # Image size
    return np.array(img) / 255.0

# Preprocess all images
train_images = np.array([preprocess_image(p) for p in train_data['file_path']])
val_images = np.array([preprocess_image(p) for p in val_data['file_path']])
test_images = np.array([preprocess_image(p) for p in test_data['file_path']])

# Encode labels
encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_data['dx'].values)
val_labels = encoder.transform(val_data['dx'].values)
test_labels = encoder.transform(test_data['dx'].values)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define DenseNet201 Model
input_tensor = Input(shape=(192, 192, 3))  # Adjust input shape
base_model = DenseNet201(weights='imagenet', include_top=False, input_tensor=input_tensor)

# Fine-tune last few layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Add classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output_tensor = Dense(len(encoder.classes_), activation='softmax')(x)

# Compile model
model = Model(inputs=base_model.input, outputs=output_tensor)
model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7, verbose=1)

# Train model
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    validation_data=(val_images, val_labels),
    epochs=50,
    callbacks=[early_stop, reduce_lr]
)

# Save model
model.save("skin_cancer_densenet201.h5")

# Predictions
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
print("Classification Report:\n", classification_report(test_labels, y_pred_classes, target_names=encoder.classes_))

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, y_pred_classes)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(np.arange(len(encoder.classes_)), encoder.classes_, rotation=45)
plt.yticks(np.arange(len(encoder.classes_)), encoder.classes_)
plt.tight_layout()
plt.show()

# Metrics
accuracy = accuracy_score(test_labels, y_pred_classes)
precision = precision_score(test_labels, y_pred_classes, average='weighted')
recall = recall_score(test_labels, y_pred_classes, average='weighted')
f1 = f1_score(test_labels, y_pred_classes, average='weighted')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Evaluate final model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)

# Plot Accuracy and Loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o', color='orange')
plt.axhline(y=test_accuracy, color='red', linestyle='--', label=f'Testing Accuracy: {test_accuracy:.2f}')
plt.title('Training, Validation, and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o', color='orange')
plt.axhline(y=test_loss, color='red', linestyle='--', label=f'Testing Loss: {test_loss:.2f}')
plt.title('Training, Validation, and Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
