import numpy as np
import cv2

def find_depth(imageA):
	imageACopy = np.float32(imageA)
	#  print(imageA)
	# imageB = cv2.imread(args["second"])

	camera_mat1 = [[475.847198,0,314.711304],[0,475.847229,245.507904],[0,0,1]]
	camera_img1 = np.array(camera_mat1, dtype=np.float32)

	points3d = cv2.rgbd.depthTo3d(imageACopy,camera_img1)

	return points3d