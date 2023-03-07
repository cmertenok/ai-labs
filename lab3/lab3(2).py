import cv2
import numpy as np

cap = cv2.VideoCapture('road.mp4')

# rho = 3
# theta = np.pi / 180
# threshold = 15
# min_line_len = 150
# max_line_gap = 60

rho = 6
theta = np.pi / 60
threshold = 100
min_line_len = 40
max_line_gap = 25

while True:
    ret, frame = cap.read()

    if not ret:
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)

    low_t = 50
    high_t = 150
    edges = cv2.Canny(blur, low_t, high_t)

    lines = cv2.HoughLinesP(
        edges, rho, theta, threshold,
        min_line_len, max_line_gap
    )

    if lines is not None:

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

    cv2.imshow('Result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
