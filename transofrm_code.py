# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 18:22:15 2022

@author: my pc
"""

import cv2
import numpy as np
import os

trainingSamplePath = "C:/cp/Mnist_Png/Train1/dataD/9/"

for _file in os.listdir(trainingSamplePath):
    # lab=np.append(lab,int(_file[1]))
    src = cv2.imread(trainingSamplePath+_file)
    # kernel = np.ones((3, 3), np.uint8)
    # warp_dst = cv2.dilate(src, kernel, iterations=1)
    # warp_dst = cv2.blur(src,(5,5))
    srcTri = np.array( [[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]] ).astype(np.float32)
    dstTri = np.array( [[0, src.shape[1]*0.10], [src.shape[1]*0.90, src.shape[0]*0.10], [src.shape[1]*0.10, src.shape[0]*0.90]] ).astype(np.float32)
    warp_mat = cv2.getAffineTransform(srcTri, dstTri)
    warp_dst = cv2.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))
    center = (src.shape[1]//2, src.shape[0]//2)
    angle =-10
    scale = 0.92
    rot_mat = cv2.getRotationMatrix2D( center, angle, scale )
    warp_rotate_dst = cv2.warpAffine(warp_dst, rot_mat, (src.shape[1], src.shape[0]))
    cv2.imwrite(trainingSamplePath+_file,warp_rotate_dst)
  