import os
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np

model = load_model('mnist_asl_model2.keras')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

img_vals = []
label_map  = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

while True:
  ret, frame = cap.read()
  H, W, _ = frame.shape
  x_vals = []
  y_vals = []

  framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  result = hands.process(framergb)
  landmarks = result.multi_hand_landmarks

  if landmarks:
    for lm in landmarks:
      for point in lm.landmark:
        x = int(point.x * W)
        y = int(point.y * H)
        x_vals.append(x)
        y_vals.append(y)
      
    min_x = min(x_vals) - 20 if min(x_vals) - 20 >= 0 else 0
    max_x = max(x_vals) + 20 
    min_y = min(y_vals) - 20 if min(y_vals) - 20 >= 0 else 0
    max_y = max(y_vals) + 20


    hand_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[min_y:max_y, min_x:max_x],(28,28))
       
    img = (hand_frame/255).reshape(-1, 28, 28, 1)
    prediction = model.predict(img, verbose = 0)
    predicted_letters =  np.flip(np.argsort(prediction[0]))

    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 0), 4)
    cv2.putText(frame, label_map[predicted_letters[0]], (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label_map[predicted_letters[1]], (min_x + 50, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label_map[predicted_letters[2]], (min_x + 100, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    
    # if key%256 == 32: # Space
    #     print("Save frame")
    #     print(f"x max = {max_x}\nx min = {min_x}\ny max = {max_y}\ny min = {min_y}")
    #     cv2.imwrite(('frame.jpg'), frame)

  cv2.imshow('frame', frame)
  key = cv2.waitKey(1)
  if key%256 == 27: # ESC
        print("Quit app")
        break

cap.release()
cv2.destroyAllWindows()
