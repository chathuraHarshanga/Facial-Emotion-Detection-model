
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing import image
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

model = model_from_json(open("fer.json", "r").read())
model.load_weights('fer.h5') 
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    

cap=cv2.VideoCapture(0)  

if not cap.isOpened():  
    print("Cannot open camera")
    exit()

while True:
   
    ret, frame=cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        
    gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces_detected = face_haar_cascade.detectMultiScale(gray_image,1.32,5)
    
    #Draw Triangles around the faces detected
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0), thickness=7)
        roi_gray=gray_image[y:y+w,x:x+h]
        roi_gray=cv2.resize(roi_gray,(48,48))
         
        image_pixels = tf.keras.preprocessing.image.img_to_array(roi_gray)
        # plt.imshow(image_pixels)
        #plt.show()
        image_pixels = np.expand_dims(image_pixels, axis = 0)
        image_pixels /= 255

        predictions = model.predict(image_pixels)
        print(predictions)
        max_index = np.argmax(predictions[0])
        emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        emotion_prediction = emotion_detection[max_index]
        
        print(emotion_prediction)
        
        cv2.putText(frame,emotion_prediction,(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    
    resize_image = cv2.resize(frame, (1000, 700))
    cv2.imshow('Emotion',resize_image)
    if cv2.waitKey(10) == ord('b'):
            break


cap.release()
cv2.destroyAllWindows
