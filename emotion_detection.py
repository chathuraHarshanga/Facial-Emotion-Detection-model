from fer import FER
import cv2
import face_recognition

img = cv2.imread("images/justin2.jpg")
imgEmotion = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
faceLoc = face_recognition.face_locations(img);

detector = FER()
detector.detect_emotions(img)
print(detector.detect_emotions(img))

cv2.imshow('original image',img)
cv2.waitKey()
cv2.destroyAllWindows()


