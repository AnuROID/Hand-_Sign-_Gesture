import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize the video capture and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "C:/yora/handsignrecoprjkt/data/D"

counter = 0
save_interval = 0.3 # Save every 30 milliseconds
start_time = time.time()  # Track the time for saving images

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image.")
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background for the processed hand image
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
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

        # Display the processed images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        # Save the image every 30 milliseconds
        current_time = time.time()
        if current_time - start_time >= save_interval:  # Check if 30ms have passed
            counter += 1
            file_path = f'{folder}/Image_{time.time()}.jpg'
            cv2.imwrite(file_path, imgWhite)
            print(f"Image saved at {file_path}. Total images saved: {counter}")
            start_time = current_time  # Reset the timer

            # Stop the program automatically after 100 images
            if counter >= 2:
                print("1 images have been saved. Program is terminating.")
                break

    # Display the original image
    cv2.imshow("Image", img)
    
    # Quit when 'q' is pressed (optional, if you want a manual exit too)
    key = cv2.waitKey(1)
    if key == ord("q"):
        print(f"Program terminated manually. Total images saved: {counter}")
        break

cap.release()
cv2.destroyAllWindows()
