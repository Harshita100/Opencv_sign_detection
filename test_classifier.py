import cv2
import pickle
import mediapipe as mp
import numpy as np

cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing_hands = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence =0.3)

labels = {0:'back',1:'next',2:'first' }

data_t = []

model_dic = pickle.load(open('./model.pickle', 'rb'))
model = model_dic['model']

while True:
    rect, frame = cam.read()


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    data_t = []
    x_=[]
    y_=[]

    h, w ,z = frame.shape

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_drawing_hands.draw_landmarks(frame, hl,mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style() )

        for lm in results.multi_hand_landmarks:
            for i in range(len(lm.landmark)):
                x = lm.landmark[i].x
                y = lm.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(lm.landmark)):
                x = lm.landmark[i].x
                y = lm.landmark[i].y

                data_t.append(x-min(x_))
                data_t.append(y-min(y_))

        predicts = model.predict([np.asarray(data_t)])
        charac = labels[int(predicts[0])]

        x1 = int(min(x_) *w) -10
        y1 = int(min(y_) *h) -10

        x2 = int(max(x_) *w) -10
        y2 = int(max(y_) *h) -10

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 4)
        cv2.putText(frame, charac, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3, cv2.LINE_AA)
        

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

cam.release()
cv2.distroyAllWindows()

        
        


