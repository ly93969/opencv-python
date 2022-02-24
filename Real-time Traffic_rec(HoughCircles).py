import cv2
import numpy as np
import os


def imgdetect(image):
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([124, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerb=blue_lower, upperb=blue_upper)
    Canny = cv2.Canny(mask, 9, 9)
    circle = cv2.HoughCircles(Canny, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=100, maxRadius=200)
    return circle


myList = os.listdir('ImagesQuery')

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    imgOriginal = image.copy()
    cir = imgdetect(image)
    if not cir is None:
        cir = np.uint16(np.around(cir))
        max_r, max_i = 0, 0
        for i in range(len(cir[:, :, 2][0])):
            if cir[:, :, 2][0][i] > 50 and cir[:, :, 2][0][i] > max_r:
                max_i = i
                max_r = cir[:, :, 2][0][i]
        x, y, r = cir[:, :, :][0][max_i]
        if y > r and x > r:
            square = imgOriginal[y - r:y + r, x - r:x + r]
            img_gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
            cv2.imshow('img_gray', img_gray)
            for i in range(4):
                template = cv2.imread('ImagesQuery/' + myList[i])
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCORR_NORMED)
                (_, score, _, _) = cv2.minMaxLoc(res)
                threshold = 0.8
                h, w = template.shape[:2]
                if score > 0.93:
                    cv2.rectangle(imgOriginal, (x - r - 5, y - r - 5), (x + r + 5, y + r + 5), (0, 255, 0), 2)
                    if i == 0:
                        cv2.putText(imgOriginal, 'Back', (x, y + r + 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                    if i == 1:
                        cv2.putText(imgOriginal, 'Forward', (x, y + r + 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                    if i == 2:
                        cv2.putText(imgOriginal, 'Left', (x, y + r + 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                    if i == 3:
                        cv2.putText(imgOriginal, 'Right', (x, y + r + 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('img', imgOriginal)
    cv2.waitKey(1)
