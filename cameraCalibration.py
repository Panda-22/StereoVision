#!/usr/bin/env python

'''
# https://learnopencv.com/camera-calibration-using-opencv/
# https://blog.csdn.net/qingfengxd1/article/details/108539722
'''
import cv2
import numpy as np
import os
import glob

###########
# Calculate the camera matrics
###########

# Defining the dimensions of checkerboard
CHECKERBOARD = (5, 8) #根据具体棋盘格数指定，6*9的格在这里是（5，8），可能是指黑白交叉点
# 每行棋和每列的内角数（patternSize = cvSize (points_per_row, points_per_colum) = cvSize(columns,rows)）
#CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #终止条件

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = [] # 真实世界中的3d点
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] # 图像中的2d点

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

'''
上一段也可写作：
# 准备对象点， 如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*8,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:8].T.reshape(-1,2)
'''

# Extracting path of individual image stored in a given directory
images = glob.glob('./calib-right/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners 寻找棋盘中的点
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
    	cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
    
    cv2.imshow('img',img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

h,w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,None)

print("Camera matrix : \n")
print(cameraMatrix)
print("dist Coefficient: \n")
print(distCoeffs)
# 透镜失真系数
print("rvecs : \n")
print(rvecs)
# rvecs：3×1 Rotationl旋转向量。矢量的方向指定了旋转轴，矢量的大小指定了旋转角度。
print("tvecs : \n")
print(tvecs)
# tvecs：3×1 translation平移向量。


###########
# Save Camera Matrics to file
###########

def matrix2txt(filename, matrix_name, matrix):
    with open(filename, 'a') as f:
        f.write(matrix_name + ' : ')
        for row in matrix:
            for element in row:
                f.write(str(element) + ' ')
        f.write('\n')
    f.close()

filename = 'cali.txt'

matrix2txt(filename, 'cameraMatrix', cameraMatrix)
matrix2txt(filename, 'distCoeffs', distCoeffs)

'''with open(filename, 'a') as f:
    f.write('rvecs : ' + str(rvecs) + '\n')
    f.write('tvecs : ' + str(tvecs) + '\n')
f.close()'''


###########
# Using the derived camera parameters to undistort the image
###########

def undstImg(i):
    img = cv2.imread(images[i])
    # Refining the camera matrix using parameters obtained by calibration
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h), 1, (w,h))

    # Method 1 to undistort the image: 使用上面获得的ROI裁剪结果
    dst = cv2.undistort(img, cameraMatrix, distCoeffs, None, newcameramtx)

    # Method 2 to undistort the image: 找到从扭曲图像到未扭曲图像的映射函数。然后使用重映射功能。
    #mapx,mapy=cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None,newcameramtx,(w,h),5)
    #dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    
    # Displaying the undistorted image
    cv2.imshow("undistorted image",dst)
    cv2.waitKey(0)

'''for i in range(15): 
    undstImg(i)'''



