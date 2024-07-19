import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import cv2

# Load and preprocess the dataset as before
data_dir = pathlib.Path('training-images')
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Try to load the model
try:
    model = keras.models.load_model('object_detector1.keras')
    print("Model loaded successfully.")
except IOError:
    print("Model not found. Training a new one.")
    data_augmentation = keras.Sequential(
      [
        layers.RandomFlip("horizontal",
          input_shape=(img_height,
            img_width,
            3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
      ]
    )

    model = Sequential([
      layers.Input(shape=(img_height, img_width, 3)),
      data_augmentation,
      layers.Rescaling(1./255),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes, name="outputs")
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    epochs = 15
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save('object_detector1.keras')

def crop_largest_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return image[y:y+h, x:x+w]
    return image

# Capture image from webcam and process
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Crop the largest object
    cropped_frame = crop_largest_object(frame)
    if cropped_frame.shape[0] >= 100 and cropped_frame.shape[1] >= 100:
        # Resize to model input size
        resized_frame = cv2.resize(cropped_frame, (img_height, img_width))
        # Display the cropped and resized frame
        cv2.imshow('Cropped and Resized Object', resized_frame)
    else:
        cv2.imshow('Cropped and Resized Object', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
