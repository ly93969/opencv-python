import cv2
import numpy as np
import os

def imgdetect(image):
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([124, 255, 255])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.inRange(image, blue_lower, blue_upper)
    # image = cv2.blur(image, (9, 9))
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    image = cv2.Canny(image, 9, 9)
    return image

def findDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList


def findID(img, desList, thres=25):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    print(matchList)
    if len(matchList) != 0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal


path = 'ImagesQuery'
orb = cv2.ORB_create(nfeatures=1000)

images = []
className = []
myList = os.listdir(path)
print('Total Classes Detected', len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}')
    imgCur = imgdetect(imgCur)
    images.append(imgCur)
    className.append(os.path.splitext(cl)[0])
print(className)




desList = findDes(images)
print(len(desList))

cap = cv2.VideoCapture(0)

while True:

    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = imgdetect(img2)
    id = findID(img2, desList)
    print('ID',id)
    if id != -1:
        cv2.putText(imgOriginal, className[id], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    cv2.imshow('img2', img2)
    cv2.waitKey(1)
