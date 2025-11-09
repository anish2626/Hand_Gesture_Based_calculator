import numpy as np
import cv2
from keras.models import load_model

# Start capturing the video
cap = cv2.VideoCapture(0)

# Dimensions of the region of interest
x = 50
y = 60
w = 200
h = 200

# Get weights from the trained model
model = load_model('D:\HCI_project\Gesture_Calculator\model.h5')

# This function is used to predict the image
def predictionImage(roi, thresh):
    img = np.zeros_like(roi, np.float32)

    # Converting 1 channel threshold image to 3 channel image for our model
    img[:, :, 0] = thresh
    img[:, :, 1] = thresh
    img[:, :, 2] = thresh
    img = img.reshape(1, 200, 200, 3)
    # Normalizing the image
    img /= 255.

    return img

# Preprocessing as done while creating the dataset
def imagePreprocess(frame):
    cv2.rectangle(frame, (x, y), (w + x, h + y), (0, 255, 0), 2)
    roi = frame[y:h + y, x:w + x]

    # Convert ROI to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([2, 20, 50]), np.array([30, 255, 255]))

    # Reduce noise using Gaussian blur
    blur = cv2.GaussianBlur(mask, (5, 5), 1)

    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(blur, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Add Canny edge detection
    edges = cv2.Canny(erosion, threshold1=100, threshold2=200)

    # Combine thresholding with edges
    ret, thresh = cv2.threshold(erosion, 127, 255, 0)
    combined = cv2.bitwise_or(thresh, edges)

    # Get the image to be used for prediction
    img = predictionImage(roi, combined)

    return mask, combined, img

# Write the predicted text
def writeTextToWindow(img, text, default_x_calc, default_y_calc):
    fontscale = 1.0
    color = (0, 0, 0)
    fontface = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(img, str(text), (default_x_calc, default_y_calc), fontface, fontscale, color)

    return img

# This array contains the first and the second operand that is to be used in calculation
predArray = [-1, -1]

# Dimensions used while writing the predicted text
default_y_calc = 80
default_x_calc = 25
 
predCount = 0  # For confirming the number displayed
predPrev = 0

# Space for writing the predicted text
result = np.zeros((300, 300, 3), np.uint8)
result.fill(255)  # Fill result window (make it white)
cv2.putText(result, "Calculator", (25, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 0))

# Create a scrollbar for scrolling
cv2.namedWindow("result")
cv2.createTrackbar("Scroll", "result", 0, 0, lambda x: None)

while True:
    ret, frame = cap.read()  # Read frame

    mask, thresh, img = imagePreprocess(frame)

    # If we get the same prediction for 15 times, we take it as the confirmed prediction
    if predCount > 15:
        print('Prediction: ' + str(predPrev))

        # Check whether it is the first operand or the second
        if predArray[0] == -1:
            predArray[0] = predPrev
            string = '{} + '.format(predArray[0])
            writeTextToWindow(result, string, default_x_calc, default_y_calc)
            default_x_calc += 20

        else:
            default_x_calc += 40
            predArray[1] = predPrev
            string = '{} = {}'.format(predArray[1], np.sum(predArray, axis=0))
            writeTextToWindow(result, string, default_x_calc, default_y_calc)

            default_x_calc = 25

            # Expand the result image if needed
            if default_y_calc + 30 > result.shape[0]:
                new_result = np.zeros((result.shape[0] + 300, result.shape[1], 3), np.uint8)
                new_result.fill(255)  # Fill the new space with white
                new_result[:result.shape[0], :, :] = result  # Copy old content into the new result
                result = new_result

            default_y_calc += 30

            print("Sum: {}".format(np.sum(predArray, axis=0)))
            predArray = [-1, -1]  # Reset the values of the operands
        predCount = 0  # Start counting again to get the next prediction

    predict = model.predict(img)  # Predict the number
    pred = predict.argmax()
    # Increase predCount only if the previous prediction matches with our current prediction
    if predPrev == pred:
        predCount += 1
    else:
        predCount = 0

    predPrev = pred

    # Scroll based on scrollbar position
    scroll_pos = cv2.getTrackbarPos("Scroll", "result")
    visible_result = result[scroll_pos:scroll_pos + 300, :, :]

    # Show the required windows
    cv2.imshow("result", visible_result)  # Scrolling result window
    cv2.imshow('frame', frame)  # Main webcam window
    cv2.imshow('thresh', thresh)  # Thresholded image used for prediction

    # Update scrollbar max value
    max_scroll = max(result.shape[0] - 300, 0)
    cv2.setTrackbarMax("Scroll", "result", max_scroll)

    k = cv2.waitKey(30) & 0xff  # Exit if Esc is pressed
    if k == 27:
        break

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Destroy all windows