import numpy as np
import cv2
import json
import os
from collections import deque

# File to save and load settings
settings_file = 'settings.json'

# Buffer to store the last 3 frames for temporal smoothing
frame_buffer = deque(maxlen=3)

def update(x):
    process_frame(current_frame.copy())
    save_settings()

def process_frame(frame):
    # Add the current frame to the buffer
    frame_buffer.append(frame)

    # Compute the weighted average frame if we have enough frames in the buffer
    if len(frame_buffer) == frame_buffer.maxlen:
        weights = np.array([0.1, 0.3, 0.6])
        avg_frame = np.average(frame_buffer, axis=0, weights=weights).astype(np.uint8)
    else:
        avg_frame = frame

    # Get current positions of trackbars
    blur_x = cv2.getTrackbarPos('Blur X', 'Settings')
    blur_y = cv2.getTrackbarPos('Blur Y', 'Settings')
    kernel_size = cv2.getTrackbarPos('Kernel Size', 'Settings')
    canny_lower = cv2.getTrackbarPos('Canny Lower', 'Settings')
    canny_upper = cv2.getTrackbarPos('Canny Upper', 'Settings')
    morph_type = cv2.getTrackbarPos('Morph Type', 'Settings')
    approx_epsilon = cv2.getTrackbarPos('Approx Epsilon', 'Settings') / 100.0  # scaled down for finer control
    brightness = cv2.getTrackbarPos('Brightness', 'Settings')
    contrast = cv2.getTrackbarPos('Contrast', 'Settings')

    # Apply brightness and contrast
    avg_frame = cv2.convertScaleAbs(avg_frame, alpha=contrast/50, beta=brightness-50)

    # Ensure kernel size and blur size are odd
    blur_x = blur_x * 2 + 1
    blur_y = blur_y * 2 + 1
    kernel_size = kernel_size * 2 + 1

    # Convert the frame to grayscale and blur it
    gray = cv2.cvtColor(avg_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_x, blur_y), 0)

    # Detect edges in the frame
    edged = cv2.Canny(gray, canny_lower, canny_upper)

    # Select the morphological operation
    morph_operations = [cv2.MORPH_ERODE, cv2.MORPH_DILATE, cv2.MORPH_CLOSE]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(edged, morph_operations[morph_type], kernel)

    # Create a blank color image
    closed_color = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

    # Find contours in the closed image
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours and label them
    for i, contour in enumerate(contours):
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, approx_epsilon * peri, True)

        # Draw the contour and label it on the color image
        if len(approx) == 4:
            cv2.drawContours(closed_color, [approx], -1, (0, 255, 0), 3)  # Green for rectangles

            # Calculate homography matrix
            src_pts = np.float32(approx).reshape(-1, 1, 2)
            dst_pts = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]]).reshape(-1, 1, 2)  # Reference rectangle
            H, _ = cv2.findHomography(src_pts, dst_pts)

            # Display the homography matrix
            # print(f"Homography matrix for rectangle #{i+1}:")
            # print(H)
        else:
            cv2.drawContours(closed_color, [approx], -1, (0, 0, 255), 1)  # Red for other shapes
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(closed_color, f"#{i+1}", (cX - 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Add white padding around the images
    avg_frame_padded = cv2.copyMakeBorder(avg_frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    closed_color_padded = cv2.copyMakeBorder(closed_color, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Concatenate original frame and closed_color side by side
    combined = np.hstack((avg_frame_padded, closed_color_padded))

    # Display the combined image with contours and labels
    cv2.imshow("Combined Output", combined)

def save_settings():
    settings = {
        'blur_x': cv2.getTrackbarPos('Blur X', 'Settings'),
        'blur_y': cv2.getTrackbarPos('Blur Y', 'Settings'),
        'kernel_size': cv2.getTrackbarPos('Kernel Size', 'Settings'),
        'canny_lower': cv2.getTrackbarPos('Canny Lower', 'Settings'),
        'canny_upper': cv2.getTrackbarPos('Canny Upper', 'Settings'),
        'morph_type': cv2.getTrackbarPos('Morph Type', 'Settings'),
        'approx_epsilon': cv2.getTrackbarPos('Approx Epsilon', 'Settings'),
        'brightness': cv2.getTrackbarPos('Brightness', 'Settings'),
        'contrast': cv2.getTrackbarPos('Contrast', 'Settings')
    }
    with open(settings_file, 'w') as f:
        json.dump(settings, f)

def load_settings():
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            cv2.setTrackbarPos('Blur X', 'Settings', settings.get('blur_x', 1))
            cv2.setTrackbarPos('Blur Y', 'Settings', settings.get('blur_y', 1))
            cv2.setTrackbarPos('Kernel Size', 'Settings', settings.get('kernel_size', 1))
            cv2.setTrackbarPos('Canny Lower', 'Settings', settings.get('canny_lower', 10))
            cv2.setTrackbarPos('Canny Upper', 'Settings', settings.get('canny_upper', 250))
            cv2.setTrackbarPos('Morph Type', 'Settings', settings.get('morph_type', 2))
            cv2.setTrackbarPos('Approx Epsilon', 'Settings', settings.get('approx_epsilon', 2))
            cv2.setTrackbarPos('Brightness', 'Settings', settings.get('brightness', 50))
            cv2.setTrackbarPos('Contrast', 'Settings', settings.get('contrast', 50))

# Load the video file
video_path = "test1.mp4"
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Create a window
cv2.namedWindow('Settings')
cv2.moveWindow('Settings', 0, 0)

# Create trackbars for GaussianBlur parameters
cv2.createTrackbar('Blur X', 'Settings', 1, 20, update)
cv2.createTrackbar('Blur Y', 'Settings', 1, 20, update)

# Create trackbars for getStructuringElement parameters
cv2.createTrackbar('Kernel Size', 'Settings', 1, 20, update)

# Create trackbars for Canny edge detection thresholds
cv2.createTrackbar('Canny Lower', 'Settings', 10, 100, update)
cv2.createTrackbar('Canny Upper', 'Settings', 250, 500, update)

# Create trackbar for morphological operation type
cv2.createTrackbar('Morph Type', 'Settings', 2, 2, update)  # 0: ERODE, 1: DILATE, 2: CLOSE

# Create trackbar for contour approximation accuracy
cv2.createTrackbar('Approx Epsilon', 'Settings', 2, 20, update)

# Create trackbars for brightness and contrast
cv2.createTrackbar('Brightness', 'Settings', 50, 100, update)
cv2.createTrackbar('Contrast', 'Settings', 50, 100, update)

# Load settings from file
load_settings()

# Move the main image window to the right
cv2.namedWindow('Combined Output')
cv2.moveWindow('Combined Output', 300, 0)

# Video control variables
paused = False
step = 0
current_frame = None

while True:
    if not paused or step != 0:
        ret, current_frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, current_frame = cap.read()

        process_frame(current_frame)

        step = 0

    # Handle key events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
    elif key == ord('n'):
        if paused:
            step = 1
    elif key == ord('b'):
        if paused:
            frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_pos - 2))
            ret, current_frame = cap.read()
            process_frame(current_frame)
            step = 1

# Release the video file and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
