# import the necessary packages
import numpy as np
import cv2

# start capturing video from the webcam
video_path = "test1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video stream from webcam.")
    exit()

while True:
    # read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from webcam.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    # convert the frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # detect edges in the frame
    edged = cv2.Canny(gray, 10, 250)
    
    # construct and apply a closing kernel to 'close' gaps between 'white'
    # pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    # find contours (i.e. the 'outlines') in the frame and initialize the
    # total number of books found
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0
    
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # if the approximated contour has four points, then assume that the
        # contour is a book -- a book is a rectangle and thus has four vertices
        if len(approx) == 4:
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 4)
            total += 1
    
    # display the output
    print(f"I found {total} books in the frame.")
    cv2.imshow("Output", frame)
    
    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
