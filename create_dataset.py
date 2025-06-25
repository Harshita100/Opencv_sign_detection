import os
import cv2

DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

classes = 3
dataset = 100

cam = cv2.VideoCapture(0)
for j in range(classes):
    if not os.path.exists(os.path.join(DATA_DIR,str(j))):
        os.makedirs(os.path.join(DATA_DIR,str(j)))

        print(f"collecting for {j+1} class")

        # done = False
        while True:
            ret, frame = cam.read()
            cv2.putText(frame,"Press s to start!: ", (100,50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,255,255), 3, cv2.LINE_AA)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('d'):
                break
        
        counter = 0
        while counter < dataset:
            ret, frame = cam.read()
            cv2.imshow('frame',frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame )

            counter +=1
        
cam.release()
cv2.destroyAllWindows()