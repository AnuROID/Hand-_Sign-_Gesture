import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import joblib

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the trained model
model = joblib.load("C:/yora/handsignrecoprjkt/hand_sign_model.pkl")

# Initialize the video capture and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# Hand sign categories
categories = ['A', 'B', 'C', 'D']  # Replace with your labels

while True:
    # Capture frame from webcam
    success, img = cap.read()
    if not success:
        print("Failed to capture image.")
        break

    # Detect hands
    hands, img = detector.findHands(img)
    if hands:
        # Extract hand bounding box
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background for processed hand image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Bounds checking for cropping
        h_img, w_img, _ = img.shape
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(w_img, x + w + offset), min(h_img, y + h + offset)
        imgCrop = img[y1:y2, x1:x2]

        # Check if imgCrop is valid
        if imgCrop.size == 0:
            print("Warning: imgCrop is empty!")
            continue

        # Aspect ratio adjustment
        aspectRatio = h / w
        if aspectRatio > 1:
            # Resize based on height
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
        else:
            # Resize based on width
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

        # Preprocess the image for prediction
        imgGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
        imgResized = cv2.resize(imgGray, (64, 64))  # Match model input size
        imgFlatten = imgResized.flatten().reshape(1, -1)

        # Debugging prediction input
        print(f"Input Shape: {imgFlatten.shape}")

        # Predict the hand sign
        prediction = model.predict(imgFlatten)
        print(f"Prediction: {prediction}")

        # Validate prediction and map to category
        if len(prediction) > 0 and 0 <= prediction[0] < len(categories):
            label = categories[prediction[0]]
        else:
            label = "Unknown"

        # Display prediction on the image
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display processed images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the original image
    cv2.imshow("Image", img)

    # Quit when 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        print("Program terminated manually.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
