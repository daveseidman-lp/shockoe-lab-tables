import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

img_height = 180
img_width = 180
video_path = 'test-images/test-vid1.mov'
model = keras.models.load_model('object_detector1.keras')
class_names = ['museum', 'noise', 'restaurants']

def crop_largest_object(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if contours:
      largest_contour = max(contours, key=cv2.contourArea)
      x, y, w, h = cv2.boundingRect(largest_contour)
      return image[y:y+h, x:x+w]
  return image

while True:
  cap = cv2.VideoCapture(video_path)
  frame_count = 0

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    cropped_image = crop_largest_object(frame)

    resized_image = cv2.resize(cropped_image, (img_height, img_width))

    img_array = tf.keras.utils.img_to_array(resized_image)
    # img_array = img_array / 255.0  # Rescale the image to [0, 1] range
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array, verbose=0)
    # score = tf.nn.softmax(predictions[0])

    scores = tf.nn.softmax(predictions[0])
      # Display the prediction scores on the image
    y_position = 30
    for i, class_name in enumerate(class_names):
      score = scores[i]
      label = f"{class_name}: {score:.2f}"
      cv2.putText(frame, label, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
      y_position += 30
    # confidence = 100 * np.max(score)
    # predicted_class = class_names[np.argmax(score)]

    # label = f"{predicted_class}: {confidence:.2f}%"
    # cv2.putText(resized_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('Cropped Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()
