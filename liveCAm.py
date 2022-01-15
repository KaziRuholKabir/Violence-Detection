import os
import cv2
from flask.scaffold import F
from tensorflow.keras.models import load_model
import numpy as np

cap= cv2.VideoCapture(0)
data = []
model = load_model("Violence_detection-CNN-BiLSTM.h5")

text = ""
while True: 
    ret,frame = cap.read()
    image = cv2.resize(frame,(100,100))
    data.append(np.array(image))
    
    if len(data) == 10:
        
        X = np.array(data).reshape(1, 10, 100, 100, 3)
        Y = model.predict(X)[0][0]
        print(Y)
        data = []
        

        if Y > 0.5:
            text = "This is Violence"
            print(text)
        else:
            text = "This is not Violence"
            print(text)



    cv2.putText(frame,text,(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    cv2.imshow("Anything", frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113 :
        break
    


