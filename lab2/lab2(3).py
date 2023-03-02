import cv2
import numpy

cap = cv2.VideoCapture("video_2.mp4")

#Завантаження класифікаторів для виявлення людей
pedestrian_detector = cv2.HOGDescriptor()
pedestrian_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Виявлення пішоходів
    pedestrians, _ = pedestrian_detector.detectMultiScale(gray, winStride=(8,8), padding=(32,32), scale=1.05)

    #Малювання прямокутників
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow("Pedestrians Detected", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
