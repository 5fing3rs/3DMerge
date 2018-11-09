import numpy as np
import cv2
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="path to the first image")
# ap.add_argument("-s", "--second", required=True,
	# help="path to the second image")
args = vars(ap.parse_args())

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
imageA = cv2.imread(args["first"],0)
imageACopy = np.float32(imageA)
# print(imageA)
# imageB = cv2.imread(args["second"])
cv2.imshow("Image A", imageA)
cv2.waitKey(0)

camera_mat1 = [[475.847198,0,314.711304],[0,475.847229,245.507904],[0,0,1]]
camera_img1 = np.array(camera_mat1, dtype=np.float32)

points3d = cv2.rgbd.depthTo3d(imageACopy,camera_img1)

for _i in range(400,450):
	for _j in range(550,600):
		for _k in range(0,3):
			print(points3d[_i][_j][_k],end = " ")
		print("_",end = "")
	print()
	print()
	print()