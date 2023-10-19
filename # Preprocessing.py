# Preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    r"C:\Users\Ayman\Documents\GitHub\ML-ops",
    target_size=(image_height, image_width),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
     r"C:\Users\Ayman\Documents\GitHub\ML-ops",
    target_size=(image_height, image_width),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

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



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(train_generator, validation_data=val_generator, epochs=10)



model.save("brain_tumor_classification_model.h5")