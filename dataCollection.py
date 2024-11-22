import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os

# Set up directories
gesture_name = input("Enter gesture label (e.g., '1', 'add', 'equals'): ")
save_dir = os.path.join('data', gesture_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Get the starting counter based on existing images
existing_images = [img for img in os.listdir(save_dir) if img.endswith('.jpg')]
counter = len(existing_images)

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
capture = False  # Flag to start/stop capturing

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    if hands and capture:  # Start capturing only when the flag is True
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]
        imgCropResized = cv2.resize(imgCrop, (imgSize, imgSize))
        imgWhite[:imgCropResized.shape[0], :imgCropResized.shape[1]] = imgCropResized

        # Save the cropped image with the correct numbering
        cv2.imwrite(f'{save_dir}/Image_{counter + 1}.jpg', imgWhite)
        counter += 1
        print(f"Captured {counter} images for gesture '{gesture_name}'")

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):  # Press 's' to start capturing
        capture = not capture
        if capture:
            print("Started capturing images...")
        else:
            print("Stopped capturing images.")

cap.release()
cv2.destroyAllWindows()
