import cv2

image = cv2.imread("photo_2.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Завантаження класифікаторів обличчя, очей та усмішок
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

#Виявлення обличчів
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    #Малювання навколо обличчя
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray_image[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    #Виявлення очей
    eyes = eye_cascade.detectMultiScale(roi_gray)
    #Малювання навколо очей
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
    #Виявлення усміщок
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
    #Малювання навколо усмішок
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 225, 0), 2)

#Рахуємо кількість обличь
num_faces = len(faces)
print("Number of people in the photo:", num_faces)

new_size = (800, 600)
resized_image = cv2.resize(image, new_size)

cv2.imshow(f'Detected people: {num_faces}', resized_image)
cv2.waitKey(0)