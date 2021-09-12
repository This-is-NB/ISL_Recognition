import cv2
import numpy as np
import os,time
# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
# def face_extractor(img):
#     return img
#     # Function detects faces and returns the cropped face
#     # If no face detected, it returns the input image
    
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
#     if faces is ():
#         return None
    
#     # Crop all faces found
#     for (x,y,w,h) in faces:
#         cropped_face = img[y:y+h, x:x+w]

#     return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
flag = False
alph = input().upper()
os.system(f"mkdir ISL_dataset\{alph}")
x = 0
while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    sign = frame[0:200,200:400]
    frame = cv2.rectangle(frame,(200,0),(400,200),(250,0,0))
    cv2.imshow("i",frame)
    
    if flag:
        count += 1
        # face = cv2.resize(faceimg, (200, 200))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = f"ISL_dataset/{alph}/" + str(count) + '.jpg'
        cv2.imwrite(file_name_path,sign)
        sign = cv2.putText(sign, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        # print(sign.shape) 
        cv2.imshow(f'{alph}', sign)
        # Put count on images and display live count
    
    key = cv2.waitKey(1)
        # print("Face not found")
    if key == 32 : #32 is the SPACE Key
        flag = True
        time.sleep(5)
        print("START")
    if key == 27: # 27 is ESC Key
        flag = False
    if count%500==0 and count>x : 
        flag = False
        x += 500
    if key == 13 or count == 2500: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")