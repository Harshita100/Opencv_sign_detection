import cv2
import pickle
import mediapipe as mp
import os
import matplotlib.pyplot as plt

DATA_DIR = '.\data'

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR,dir_,img_path))
        data_t = []
        color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_ = []
        y_ = []

        result = hands.process(color_img)
        for lm in result.multi_hand_landmarks:
            for i in range(len(lm.landmark)):
                x = lm.landmark[i].x
                y = lm.landmark[i].y

                x_.append(x)
                y_.append(y)
        
            for i in range(len(lm.landmark)):
                x = lm.landmark[i].x
                y = lm.landmark[i].y

                data_t.append(x-min(x_))
                data_t.append(x-min(x_))
            
        data.append(data_t)
        labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data':data, 'labels':labels},f)

f.close()