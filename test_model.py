import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

model = load_model('model/gesture_model.h5')

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          'add', 'subtract', 'multiply', 'divide', 'modulo',
          'open_bracket', 'close_bracket', 'equals', 'decimal', 'delete', 'ac']

gesture_to_operator = {
    'add': '+',
    'subtract': '-',
    'multiply': '*',
    'divide': '/',
}

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
imgSize = 300
offset = 20

expression = ""
result = ""

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    # Updated text color to red (BGR format: (0, 0, 255))
    cv2.putText(img, f'Expression: {expression}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, f'Result: {result}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    key = cv2.waitKey(1)
    if key & 0xFF == ord(' '):
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]
            imgCropResized = cv2.resize(imgCrop, (imgSize, imgSize))
            imgWhite[:imgCropResized.shape[0], :imgCropResized.shape[1]] = imgCropResized

            img_input = imgWhite.astype('float32') / 255.0
            img_input = np.expand_dims(img_input, axis=0)

            prediction = model.predict(img_input)
            index = np.argmax(prediction)
            label = labels[index]

            if label == 'delete':
                expression = expression[:-1]
            elif label == 'ac':
                expression = ""
                result = ""
            elif label == 'equals':  
                try:
                    result = str(eval(expression))  
                except Exception as e:
                    result = "Error"  
            else:
                if label in gesture_to_operator:
                    expression += gesture_to_operator[label]
                else:
                    expression += label

    if key == 13:  
        try:
            result = str(eval(expression))  
        except Exception as e:
            result = "Error"  

    cv2.imshow("MudraCalc", img)

    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
