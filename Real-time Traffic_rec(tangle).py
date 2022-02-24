import cv2
import numpy as np
import os


def imgdetect(image):
    a = 0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([100, 43, 46])
    high_hsv = np.array([124, 255, 255])
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    cv2.imshow('red_blue', mask)
    # 高斯模糊
    mohu = cv2.GaussianBlur(mask, (5, 5), 0)
    # 二值处理
    thresh = cv2.threshold(mohu, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # 闭运算
    ker = np.ones((5, 5), np.uint8)
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, ker)
    # cv.imshow('close', close)
    contours, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

a = 0
path = 'ImagesQuery'
myList = os.listdir(path)

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    imgOriginal = image.copy()
    contours = imgdetect(image)
    for i in contours:
        # 获取轮廓外接矩形左上角坐标(x,y)和矩形的宽高
        x, y, w, h = cv2.boundingRect(i)
        # 然后提取宽高比例在0.8到1.3和面积大于200的轮廓
        if 0.8 <= w / h <= 1.2:
            if w * h < 400:
                pass
            else:
                # 裁剪矩形并保存图片
                a += 1
                img = image[y:y + h, x:x + w]
                # 放大到指定尺寸
                img = cv2.resize(img, (380, 380))
                # cv2.imshow('text, img')
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                for i in range(4):
                    template = cv2.imread('ImagesQuery/' + myList[i])
                    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCORR_NORMED)
                    (_, score, _, _) = cv2.minMaxLoc(res)
                    threshold = 0.8
                    h, w = template.shape[:2]
                    if score > 0.90:
                        print(myList[i])
        else:
            print('None')
    cv2.imshow('img', imgOriginal)
    cv2.waitKey(1)


