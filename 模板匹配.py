import cv2
import cv2 as cv
import numpy as np
import os

a = 0
src = cv.imread('ImageText/forward.png')
hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
low_hsv = np.array([100, 43, 46])
high_hsv = np.array([124, 255, 255])
mask = cv.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
cv.imshow('blue', mask)
# 高斯模糊
mohu = cv.GaussianBlur(mask, (5, 5), 0)
# 二值处理
thresh = cv.threshold(mohu, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
# 闭运算
ker = np.ones((5, 5), np.uint8)
close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, ker)
# cv.imshow('close', close)
contours, hierarchy = cv.findContours(close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
print('总轮廓=', len(contours))
for i in contours:
    # print(cv.contourArea(i),cv.arcLength(i,True))#计算轮廓面积和周长

    # 获取轮廓外接矩形左上角坐标(x,y)和矩形的宽高
    # 然后提取宽高比例在0.8到1.3和面积大于200的轮廓
    x, y, w, h = cv.boundingRect(i)
    if 0.8 <= w / h <= 1.3:
        if w * h < 300:
            pass
        else:
            # 裁剪矩形并保存图片
            a += 1
            img = src[y:y + h, x:x + w]

            # 放大到指定尺寸
            img = cv.resize(img, (500, 460))
            cv2.imshow('12345', img)
            cv.rectangle(src, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
            # 对截取矩形图片处理
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            cv2.imshow("gray", gray)
            ret, thresh = cv.threshold(gray, 130, 255, cv.THRESH_BINARY_INV)
            cv2.imshow("thresh", thresh)
            # 开闭运算
            ker = np.ones((6, 6), np.uint8)
            close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, ker)
            # 掩码
            h, w = gray.shape[0], gray.shape[1]
            point1 = [0.15 * w, h / 4]
            point2 = [0.15 * w, 4 * h / 5]
            point3 = [0.83 * w, 4 * h / 5]
            point4 = [0.83 * w, h / 4]
            list1 = np.array([[point1, point2, point3, point4]], dtype=np.int32)
            mask = np.zeros_like(gray)
            mask = cv.fillConvexPoly(mask, list1, 255)
            mask1 = cv.bitwise_and(mask, thresh)
            cv2.imshow("mask1", mask1)

            # 开运算
            ker = np.ones((6, 6), np.uint8)
            mask1 = cv.morphologyEx(mask1, cv.MORPH_OPEN, ker)

            # 闭运算
            ker = np.ones((5, 5), np.uint8)
            mask1 = cv.morphologyEx(mask1, cv.MORPH_CLOSE, ker)

            # 找外轮廓,
            contours1, hierarchy1 = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            a = len(contours1)

            # cv.drawContours(src, contours1, -1, (0, 255, 0), 3)
            if 0 < a <= 3:
                print('单个矩形内的轮廓', a)
                list3 = []
                for i, element in enumerate(contours1):
                    x2, y2, w2, h2 = cv.boundingRect(element)
                    # print('x2,y2,w2,h2=', x2, y2, w2, h2)
                    # dist1[str(i)]=x2
                    list3.append(x2)
                # print('list3=',min(list3))
                # print(type(element))

                # 轮廓外接矩形
                # boundingboxes = [cv.boundingRect(c) for c in close]  # 返回外接矩形的四个值x,y,h,w
                list2 = []  # 存放轮廓列表
                for lk in contours1:
                    x1, y1, w1, h1 = cv.boundingRect(lk)
                    roi = mask1[y1:y1 + h1, x1:x1 + w1]
                    roi = cv.resize(roi, (60, 90))
                    # 把roi变成三通道图像
                    roi = cv.cvtColor(roi, cv.COLOR_GRAY2BGR)

                    list2.append(roi)
                    # cv.imshow('a4',list2[0])

                    # 遍历模板,进行模板匹配
                    filename = os.listdir(r'ImagesQuery1')
                    scores = []
                    for i in range(4):
                        src1 = cv.imread('ImagesQuery1/' + filename[i])
                        # gray = cv.cvtColor(src1, cv.COLOR_BGR2GRAY)
                        # ret, thresh = cv.threshold(gray, 70, 255, cv.THRESH_BINARY)
                        result = cv.matchTemplate(src1, roi, cv2.TM_CCORR_NORMED)
                        (_, score, _, _) = cv.minMaxLoc(result)

                        scores.append(score)
                    print('得分列表:', scores)
                    x3 = np.argmax(scores)  # x是列表最大值所对应的下标
                    y3 = scores[x3]
                    print('最可能取值:', x3, '分数=', scores[x3])
                    # if y3 > 70000000:
                    #
                    #     print('x1=', x1, list3)
                    #     if x1 == min(list3):
                    #
                    #         cv.putText(src, 'limt:' + str(x3), (x, y + 20), cv.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                    #
                    #     elif x1 == max(list3):
                    #
                    #         cv.putText(src, 'limt:' + '  ' + str(x3), (x, y + 20), cv.FONT_HERSHEY_DUPLEX, 1,
                    #                    (255, 0, 0), 2)
                    #     else:
                    #         cv.putText(src, 'limt:' + ' ' + str(x3), (x, y + 20), cv.FONT_HERSHEY_DUPLEX, 1,
                    #                    (255, 0, 0), 2)
                    # else:
                    #     pass
            else:
                pass

    else:
        pass

# cv.imshow('img',img)
# cv.imshow('mask1',mask1)
cv.imshow('src', src)

cv.waitKey(0)
cv.destroyAllWindows()
