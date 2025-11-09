import os
import numpy as np
import cv2
import time

# Define path for saving images
output_dir = "output_images"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create the directory if it doesn't exist

cap = cv2.VideoCapture(0)

# Region of Interest coordinates
x = 50
y = 60
w = 200
h = 200

def imagePreprocess(frame):
    cv2.rectangle(frame, (x, y), (w + x, h + y), (0, 255, 0), 2)
    roi = frame[y:h + y, x:w + x]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Mask for thresholding the skin color
    mask = cv2.inRange(hsv, np.array([2, 20, 50]), np.array([30, 255, 255]))

    # Reducing noise in the image
    kernel = np.ones((5, 5))
    blur = cv2.GaussianBlur(mask, (5, 5), 1)

    # Applying morphological operations
    dilation = cv2.dilate(blur, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Thresholding the image
    ret, thresh = cv2.threshold(erosion, 127, 255, 0)

    return mask, thresh


count = 0
frameCount = 0

while True:
    # Capture frame
    ret, frame = cap.read()
    frameCount += 1

    # Preprocess the frame
    mask, thresh = imagePreprocess(frame)

    # Save images every 5th frame
    if frameCount % 5 == 0:
        if count < 1200:  # Capture up to 1200 images
            imgName = f"{output_dir}/{count}.jpg"
            cv2.imwrite(imgName, thresh)
            count += 1

    # Show the required frames
    cv2.imshow('frame', frame)
    cv2.imshow('roi', mask)
    cv2.imshow('thresh', thresh)

    # Exit if 'Esc' is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the webcam and destroy windows
cap.release()
cv2.destroyAllWindows()
