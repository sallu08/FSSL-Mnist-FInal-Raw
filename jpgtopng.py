# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 19:05:05 2022

@author: my pc
"""

import os
import cv2

trainingSamplePath = "C:/Users/my pc/Downloads/MNIST Dataset JPG format/MNIST Dataset JPG format/MNIST - JPG - training/dataD/"

for _file in os.listdir(trainingSamplePath):
    # lab=np.append(lab,int(_file[1]))
    img = cv2.imread(trainingSamplePath+_file)
    cv2.imwrite("C:/Users/my pc/Downloads/MNIST Dataset JPG format/MNIST Dataset JPG format/MNIST - JPG - training/dataDp/"+_file.split(".")[0]+".png", img)
