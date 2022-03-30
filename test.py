import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


image = './experiment/15.jpg'
savefile = './mark1'
# image = os.listdir(image_file)
save_image = os.path.join(savefile, image)

#设定颜色HSV范围，假定为红色
redLower = np.array([0, 0, 0])
redUpper = np.array([180, 255, 46])

#读取图像
img = cv2.imread(image)

#将图像转化为HSV格式
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#去除颜色范围外的其余颜色
mask = cv2.inRange(hsv, redLower, redUpper)

# 二值化操作
ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

#膨胀操作，因为是对线条进行提取定位，所以腐蚀可能会造成更大间隔的断点，将线条切断，因此仅做膨胀操作
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(binary, kernel, iterations=1)

#获取图像轮廓坐标，其中contours为坐标值，此处只检测外形轮廓
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

plt.subplot(2, 2, 1), plt.imshow(img)
plt.title("org"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(hsv)
plt.title("hsv"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(mask)
plt.title("mask"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(binary)
plt.title("binary"), plt.xticks([]), plt.yticks([])
plt.show()
"""
if len(contours):
    #cv2.boundingRect()返回轮廓矩阵的坐标值，四个值为x, y, w, h， 其中x, y为左上角坐标，w,h为矩阵的宽和高
  boxes = [cv2.boundingRect(c) for c in contours]
  for box in boxes:
    x, y, w, h = box
    #绘制矩形框对轮廓进行定位
    cv2.rectangle(img, (x, y), (x+w, y+h), (153, 153, 0), 2)
    #将绘制的图像保存并展示
    cv2.imwrite(save_image, img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """