import cv2
import numpy as np
import os

# Define the path to the video and output folder
video_path = 'etc/museum.mov'
output_folder = 'museum'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to find and crop the largest contour
def crop_largest_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return image[y:y+h, x:x+w]
    return image

# Open the video
cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    # Read three frames
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 3 == 0:
        cropped_image = crop_largest_object(frame)
        # Save the cropped image
        cv2.imwrite(f'{output_folder}/frame_{frame_count}.jpg', cropped_image)

# Release the video capture object
cap.release()
print("Processing complete.")
