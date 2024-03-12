
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
import pandas as pd

model = load_model('mnist_asl_model2.keras')

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
_, frame = cap.read()
h, w, c = frame.shape

analysisframe = ''
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']



while True:
    _, frame = cap.read()

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
    
        if k%256 == 32:
            # SPACE pressed
            print("Space pressed")
            hand_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y_min:y_max, x_min:x_max],(28,28))
            pixel_list = []
            n, m = hand_frame.shape
            for i in range(n): #Enumerate?
                for j in range(m):
                    pixel_list.append(frame[i,j])
            
            img = (np.array(pixel_list)/255).reshape(-1, 28, 28, 1)

            prediction = model.predict(img)
            predarray = np.array(prediction[0])
            letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
            predarrayordered = sorted(predarray, reverse=True)
            high1 = predarrayordered[0]
            high2 = predarrayordered[1]
            high3 = predarrayordered[2]
            for key,value in letter_prediction_dict.items():
                if value==high1:
                    print("Predicted Character 1: ", key)
                    print('Confidence 1: ', 100*value)
                elif value==high2:
                    print("Predicted Character 2: ", key)
                    print('Confidence 2: ', 100*value)
                elif value==high3:
                    print("Predicted Character 3: ", key)
                    print('Confidence 3: ', 100*value)
            time.sleep(5)
    
    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()