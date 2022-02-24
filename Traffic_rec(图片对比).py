import cv2
import numpy as np
import os

path = 'ImagesQuery'
images = []
className = []
myList = os.listdir(path)
print('Total Classes Detected', len(myList))
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgOriginal = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scores = []
    final = []
    for i in range(4):
        template = cv2.imread('ImagesQuery/' + myList[i])
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCORR_NORMED)
        (_, score, _, _) = cv2.minMaxLoc(res)
        threshold = 0.8
        h, w = template.shape[:2]
        if score > 0.90:
            print(myList[i])
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                bottom_right = (pt[0] + w, pt[1] + h)
                cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 1)
    # cv2.namedWindow('img', 0)
    cv2.imshow('img', img)
    cv2.waitKey(1)









# template = cv2.imread('ImagesQuery/forward.png', 0)
# h, w = template.shape[:2]
# print("", h, w)
# cap = cv2.VideoCapture(0)
# while True:
#     success, img = cap.read()
#     imgOriginal = img.copy()
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
#     print(res)
#     threshold = 0.8
#     # 取匹配程度大于%80的坐标
#     loc = np.where(res >= threshold)
#     # np.where返回的坐标值(x,y)是(h,w)，注意h,w的顺序
#     for pt in zip(*loc[::-1]):
#         bottom_right = (pt[0] + w, pt[1] + h)
#         cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 2)
#
#     cv2.namedWindow('img', 0)
#     cv2.imshow('img', img)
#     cv2.waitKey(1)


