# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.metrics import Precision, Recall

# Define constants
image_height, image_width = 240, 240  
num_channels = 1  
num_classes = 2  


positive_path = r"C:\Users\Ayman\Documents\GitHub\ML-ops\positive"
negative_path = r"C:\Users\Ayman\Documents\GitHub\ML-ops\negative"


# Preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = tf.keras.preprocessing.image.load_img(os.path.join(folder, filename), color_mode='grayscale', target_size=(image_height, image_width))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img /= 255.0  # Normalize
        images.append(img)
        if "positive" in folder:
            labels.append(1)
        elif "negative" in folder:
            labels.append(0)
    return np.array(images), np.array(labels)

X_positive, y_positive = load_images_from_folder(positive_path)
X_negative, y_negative = load_images_from_folder(negative_path)

X = np.concatenate((X_positive, X_negative), axis=0)
y = np.concatenate((y_positive, y_negative), axis=0)

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(image_height, image_width, num_channels)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=12)