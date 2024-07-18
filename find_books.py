import numpy as np
import cv2
import json
import os
from collections import deque
from scipy.spatial import distance
import random

# File to save and load settings
settings_file = 'settings.json'

# Tracker data structures
next_id = 0
trackers = {}
disappeared = {}
max_disappeared = 3
min_area = 3000  # Minimum area to consider a valid rectangle

def update(x):
    global max_disappeared, min_area
    if current_frame is not None:
        max_disappeared = cv2.getTrackbarPos('Max Disappeared', 'Settings')
        min_area = cv2.getTrackbarPos('Min Area', 'Settings')
        process_frame(current_frame.copy())
    save_settings()

def register(rect):
    global next_id
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    trackers[next_id] = (rect, color)
    disappeared[next_id] = 0
    next_id += 1

def deregister(rect_id):
    global next_id
    del trackers[rect_id]
    del disappeared[rect_id]
    if not trackers:
        next_id = 0

def update_trackers(rects):
    global trackers, disappeared

    if len(rects) == 0:
        for rect_id in list(disappeared.keys()):
            disappeared[rect_id] += 1
            if disappeared[rect_id] > max_disappeared:
                deregister(rect_id)
        return

    input_centroids = []
    for rect in rects:
        if len(rect) == 4:
            centroid_x = (rect[0][0][0] + rect[1][0][0] + rect[2][0][0] + rect[3][0][0]) // 4
            centroid_y = (rect[0][0][1] + rect[1][0][1] + rect[2][0][1] + rect[3][0][1]) // 4
            input_centroids.append((centroid_x, centroid_y))

    input_centroids = np.array(input_centroids)

    if len(trackers) == 0:
        for rect in rects:
            register(rect)
    else:
        object_ids = list(trackers.keys())
        object_centroids = []
        for rect, color in trackers.values():
            centroid_x = (rect[0][0][0] + rect[1][0][0] + rect[2][0][0] + rect[3][0][0]) // 4
            centroid_y = (rect[0][0][1] + rect[1][0][1] + rect[2][0][1] + rect[3][0][1]) // 4
            object_centroids.append((centroid_x, centroid_y))

        object_centroids = np.array(object_centroids)

        D = distance.cdist(object_centroids, input_centroids)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            object_id = object_ids[row]
            trackers[object_id] = (rects[col], trackers[object_id][1])
            disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        for row in unused_rows:
            object_id = object_ids[row]
            disappeared[object_id] += 1
            if disappeared[object_id] > max_disappeared:
                deregister(object_id)

        for col in unused_cols:
            register(rects[col])

def is_valid_rectangle(quad):
    if len(quad) != 4:
        return False

    def angle(p1, p2, p3):
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        ab = b - a
        bc = c - b
        cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    angles = [angle(quad[i][0], quad[(i + 1) % 4][0], quad[(i + 2) % 4][0]) for i in range(4)]
    return all(45 <= angle <= 135 for angle in angles)

def process_frame(frame):
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
    frame = cv2.convertScaleAbs(frame, alpha=contrast/50, beta=brightness-50)

    # Ensure kernel size and blur size are odd
    blur_x = blur_x * 2 + 1
    blur_y = blur_y * 2 + 1
    kernel_size = kernel_size * 2 + 1

    # Convert the frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

    rects = []

    # Loop over the contours and label them
    for i, contour in enumerate(contours):
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, approx_epsilon * peri, True)

        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Draw the contour and label it on the color image if it meets the area and angle criteria
        if len(approx) == 4 and area > min_area and is_valid_rectangle(approx):
            rects.append(approx)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                for rect_id, (tracked_rect, color) in trackers.items():
                    if np.array_equal(tracked_rect, approx):
                        cv2.drawContours(closed_color, [approx], -1, color, 3)
                        cv2.putText(closed_color, f"ID {rect_id}", (cX - 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.drawContours(closed_color, [approx], -1, (0, 0, 255), 1)  # Red for other shapes or small rectangles

    # Update trackers
    update_trackers(rects)

    # Add white padding around the images
    frame_padded = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    closed_color_padded = cv2.copyMakeBorder(closed_color, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Concatenate original frame and closed_color side by side
    combined = np.hstack((frame_padded, closed_color_padded))

    # Add text for tracked rectangles
    y_offset = 50
    for rect_id, (rect, color) in trackers.items():
        area = cv2.contourArea(rect)
        text = f"ID {rect_id}: Area {area:.2f}"
        cv2.putText(combined, text, (frame_padded.shape[1] + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        y_offset += 40

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
        'contrast': cv2.getTrackbarPos('Contrast', 'Settings'),
        'max_disappeared': cv2.getTrackbarPos('Max Disappeared', 'Settings'),
        'min_area': cv2.getTrackbarPos('Min Area', 'Settings')
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
            cv2.setTrackbarPos('Max Disappeared', 'Settings', settings.get('max_disappeared', 3))
            cv2.setTrackbarPos('Min Area', 'Settings', settings.get('min_area', 3000))

# Load the video file
video_path = "test1.mp4"
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

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

# Create trackbars for max disappeared and min area
cv2.createTrackbar('Max Disappeared', 'Settings', 3, 20, update)
cv2.createTrackbar('Min Area', 'Settings', 3000, 10000, update)

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
	ret, current_frame = cap.read()
	if not paused or step != 0:
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
