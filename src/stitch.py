# USAGE
# python stitch.py --first images/bryce_left_01.png --second images/bryce_right_01.png 

# import the necessary packages
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2
from depth import find_depth

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-fc", "--firstcol", required=True,
	help="path to the first colour image")
ap.add_argument("-sc", "--secondcol", required=True,
	help="path to the second colour image")
ap.add_argument("-fd", "--firstdepth", required=True,
	help="path to the first depth image")
ap.add_argument("-sd", "--seconddepth", required=True,
	help="path to the second depth image")
args = vars(ap.parse_args())

# load the two images and resize them to have a width of 500 pixels
# (for faster processing)
imageA = cv2.imread(args["firstcol"])
imageB = cv2.imread(args["secondcol"])

imageADepth = cv2.imread(args["firstdepth"],0)
imageBDepth = cv2.imread(args["seconddepth"],0)

imageA = imutils.resize(imageA, width=500)
imageB = imutils.resize(imageB, width=500)

imageADepth = imutils.resize(imageADepth, width=500)
imageBDepth = imutils.resize(imageBDepth, width=500)
# imageA = imageA[-250:,250:]
# imageB = imageB[:250,:250]
# stitch the images together to create a panorama

depth = find_depth(imageBDepth)

stitcher = Stitcher()
(result, vis, imageB1) = stitcher.stitch(depth, [imageA, imageB], showMatches=True)

# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B1", imageB1)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)