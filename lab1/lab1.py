import imutils
import numpy as np
import cv2

img = cv2.imread('photo.jpg', 1);

#Відображення у вікні
cv2.imshow('Input img', img)

#Вирізання частини зображення 
roi = img[60:160, 250:400]
cv2.imshow('Input img', roi)

#Зміна розмірів зображення
h, w = img.shape[0:2]
h_new = 300
ratio = w / h
w_new = int(h_new * ratio)
resized = cv2.resize(img, (w_new, h_new))
cv2.imshow('Input img', resized)

#Зміна розміру за допомогою пакету imutils
resized = imutils.resize(img, width=400)
cv2.imshow('Input img', resized)

#Поворот зображення
resized = imutils.resize(img, width=550)
h, w = img.shape[0:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(resized, M, (w,h))
cv2.imshow('Input img', rotated)

#Розмивання зображення
resized = imutils.resize(img, width=300)
blurred = cv2.GaussianBlur(resized, (11, 11), 0)
cv2.imshow('Input img', blurred)

#Склеювання нормального та розмитого зображень
resized = imutils.resize(img, width=300)
blurred = cv2.GaussianBlur(resized, (11, 11), 0)
suming = np.hstack((resized, blurred))
cv2.imshow('Input img', suming)

#Малювання прямокутника та лінії
resized = imutils.resize(img, width=300)
cv2.rectangle(resized, (80, 170), (140, 220), (0, 0, 255), 2)
cv2.line(resized, (0, 0), (200, 200), (255, 0, 0), 5)

#Малювання ліній за набором точок та кола
img = np.zeros((200, 200, 3), np.uint8)
points = np.array([[0, 0], [100, 50], [50, 100], [0, 0]])
cv2.polylines(img, np.int32([points]), 1, (255, 255, 255))
cv2.circle(img, (140, 100), 50, (0, 0, 255), 2)
cv2.imshow('Input img', img)

#Розміщення тексту
img = np.zeros((200, 550, 3), np.uint8)
font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
cv2.putText(
    img, 'OpenCV', (0, 100), font, 4, (255, 255, 255), 4, cv2.LINE_4
)
cv2.imshow('Input img', img)

cv2.waitKey()
cv2.closeAllWindows()
