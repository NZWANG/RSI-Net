# _*_ coding:   utf-8 _*_
# Author    :   Kyro
# @Time     :    8:28 PM
import cv2
import numpy as np

def change_3_channel_to_gray(part_mask):
    temp = cv2.cvtColor(part_mask, cv2.COLOR_BGR2GRAY)
    map_list = [i for i in np.unique(temp)]
    for i,j in zip(map_list,[x for x in range(len(map_list))]):
        temp[temp == i] = j
    return temp

label = cv2.imread("./Potsdams/cut/top_potsdam_5_13_RGB_1_4_label.tif")

label_cropped = change_3_channel_to_gray(label)

cv2.imwrite("./Potsdams/cut/top_potsdam_5_13_RGB_1_4_label_gray.tif", label_cropped)