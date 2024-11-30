import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 1. Load và chuẩn bị dữ liệu
def load_dataset(data_dir, img_size=(28, 28)):
    X = []
    y = []
    for label, folder in enumerate(sorted(os.listdir(data_dir))):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                img = load_img(img_path, target_size=img_size, color_mode="grayscale")
                img_array = img_to_array(img)
                X.append(img_array)
                y.append(label)
    return np.array(X), np.array(y)

data_dir = "dataset_characters"
img_size = (28, 28)

X, y = load_dataset(data_dir, img_size)
X = X / 255.0  # Normalize dữ liệu
y = to_categorical(y, num_classes=36)  # One-hot encode nhãn

# Chia dữ liệu thành train, validation, test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2. Tạo mô hình CNN
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = (28, 28, 1)
num_classes = 36
model = create_model(input_shape, num_classes)

# Compile mô hình
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

# 4. Huấn luyện mô hình
batch_size = 32
epochs = 20

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs,
    steps_per_epoch=len(X_train) // batch_size
)

# 5. Đánh giá mô hình
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# 6. Lưu mô hình
model.save('character_recognition_model.h5')