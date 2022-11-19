import cv2
import numpy as np

Img = cv2.imread("D:/Pictures/ColorRec/all.png")

font = cv2.FONT_HERSHEY_SIMPLEX
lower_apple = np.array([0, 130, 130])
higher_apple = np.array([10, 230, 230])
lower_orange = np.array([11, 180, 180])
higher_orange = np.array([20, 255, 255])
lower_banana = np.array([21, 140, 140])
higher_banana = np.array([28, 230, 255])

img_hsv = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
kernel = np.ones(shape=[3, 3], dtype=np.uint8)

# 苹果检测二值，膨胀和高斯滤波处理
mask_apple = cv2.inRange(img_hsv, lower_apple, higher_apple)
mask_apple = cv2.dilate(mask_apple, kernel, iterations=1)
mask_apple = cv2.GaussianBlur(mask_apple, (3, 3), 0.8, 0.8)

# 橙子检测预处理
mask_orange = cv2.inRange(img_hsv, lower_orange, higher_orange)
mask_orange = cv2.dilate(mask_orange, kernel, iterations=1)
mask_orange = cv2.GaussianBlur(mask_orange, (3, 3), 0.8, 0.8)

# 香蕉检测预处理
mask_banana = cv2.inRange(img_hsv, lower_banana, higher_banana)
mask_banana = cv2.dilate(mask_banana, kernel, iterations=1)
mask_banana = cv2.GaussianBlur(mask_banana, (3, 3), 0.8, 0.8)

cnts1, hierarchy1 = cv2.findContours(mask_apple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts2, hierarchy2 = cv2.findContours(mask_banana, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts3, hierarchy3 = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(cnts1) > 0:  # 如果至少存在一个轮廓
    max_contour = max(cnts1, key=cv2.contourArea)  # 找到面积最大的轮廓
    (x1, y1, w1, h1) = cv2.boundingRect(max_contour)
    cv2.rectangle(Img, (x1, y1 - 20), (x1 + w1, y1 + h1), (0, 0, 255), 2)
    cv2.putText(Img, 'apple', (x1, y1 - 20), font, 0.7, (0, 0, 255), 2)
else:
    pass

if len(cnts2) > 0:  # 如果至少存在一个轮廓
    max_contour = max(cnts2, key=cv2.contourArea)  # 找到面积最大的轮廓
    (x2, y2, w2, h2) = cv2.boundingRect(max_contour)
    cv2.rectangle(Img, (x2, y2), (x2 + w2, y2 + h2), (18, 157, 190), 2)
    cv2.putText(Img, 'banana', (x2, y2 - 1), font, 0.7, (18, 157, 190), 2)
else:
    pass

if len(cnts3) > 0:  # 如果至少存在一个轮廓
    max_contour = max(cnts3, key=cv2.contourArea)  # 找到面积最大的轮廓
    (x3, y3, w3, h3) = cv2.boundingRect(max_contour)
    cv2.rectangle(Img, (x3, y3 - 50), (x3 + w3, y3 + h3), (2, 133, 250), 2)
    cv2.putText(Img, 'orange', (x3, y3 - 50), font, 0.7, (2, 133, 250), 2)
else:
    pass

cv2.imshow("image", Img)
cv2.waitKey(0)
cv2.destroyAllWindows()
