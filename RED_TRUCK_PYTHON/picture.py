import cv2
import sys
import numpy as np
original_stdout = sys.stdout

f = open('banner.txt', 'w')
sys.stdout = f
img =  cv2.resize(cv2.cvtColor(cv2.imread("./banner.png"), cv2.COLOR_BGR2GRAY),(50,50) , interpolation = cv2.INTER_AREA)
for j in img.tolist():
    for i in j:
        if(type(i)== "int"):
            if(i < 100):
                i = str(i) + " "
            if(i<10):
                i = str(i) + "  "
    print(j)


