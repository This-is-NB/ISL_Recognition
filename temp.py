import cv2
import numpy as np 
import pandas as pd
from tensorflow import keras
model = keras.models.load_model('MNIST_AmericanSignLanguage.h5')

cap = cv2.VideoCapture(0)
while True:
    ret, photo = cap.read()
    photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    
    # photo = cv2.flip(photo,1)
    photo = cv2.rectangle(photo,(0,0),(200,200),(0,0,255),thickness=3,)
    hand = photo[0:200,0:200]
    hand = cv2.resize(hand,(28,28))
    
    # r = model.predict(hand)
    # print(np.argmax(r))
    # cphoto = photo[100:400, 50:350]
    cv2.imshow('hi', photo)
    cv2.imshow('hand', hand)
    # hand = cv2.resize(hand,(28,28))
    hand = hand.reshape(1,28,28,1)
    # print(hand.shape)
    r = model.predict(hand)
    print(np.argmax(r))
    if cv2.waitKey(1) == 13:
        break
    
    
cv2.destroyAllWindows()
cap.release()