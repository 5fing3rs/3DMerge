# import the necessary packages
import numpy as np
import imutils
import cv2
from ess_stitch import Helper

class Stitcher(Helper):
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()
		self.right_intrinsic = np.array([[475.847198,0,314.711304],[0,475.847229,245.507904],[0,0,1]], dtype = np.float32)
		self.left_inv = np.linalg.inv(self.right_intrinsic)


	def stitch(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(imageB, imageA) = images
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)
		# print(kpsA)
		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh)

		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			return None

		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M
		result = cv2.warpPerspective(imageA, H,
			(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

		imageB1 = cv2.warpPerspective(imageB, H[:2][:2],(imageB.shape[1],imageB.shape[0]))
		# check to see if the keypoint matches should be visualized
		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
				status)

			# return a tuple of the stitched image and the
			# visualization
			return (result, vis, imageB1)

		# return the stitched image
		return (result, imageB1)

	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# check to see if we are using OpenCV 3.X
		if self.isv3:
			# detect and extract features from the image
			descriptor = cv2.xfeatures2d.SIFT_create()
			# descriptor = cv2.ORB_create(nfeatures=10000)
			(kps, features) = descriptor.detectAndCompute(image, None)

		# otherwise, we are using OpenCV 2.4.X
		else:
			# detect keypoints in the image
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)

			# extract features from the image
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)

		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create("FlannBased")
		# matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# print(matches)

			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)
			E = self.calculate_essential_matrix(matches, kpsA, kpsB)
			# return the matches along with the homograpy matrix
			# and status of each matched point
			print("E",E)
			print("H",H)

			p1 ,p2 ,p3 ,p4 = \
			self.calculate_possible_solutions(E,np.array([[475.847198,0,314.711304],[0,475.847229,245.507904],[0,0,1]], dtype = np.float32)) 

			return (matches, p1, status)

		# otherwise, no homograpy could be computed
		return None
	def calculate_essential_matrix(self,matches, kp1, kp2):
		
		img_points1, img_points2 = [], []
		camera_mat1 = [[475.847198,0,314.711304],[0,475.847229,245.507904],[0,0,1]]
		camera_mat2 = [[475.847198,0,314.711304],[0,475.847229,245.507904],[0,0,1]]

		camera_img1 = np.array(camera_mat1, dtype=np.float32)
		camera_img2 = np.array(camera_mat2, dtype=np.float32)

		for match in range(0,len(matches)):
			img_points1.append(kp1[match])
			img_points2.append(kp2[match])
			
		img_points1 = np.array(img_points1, dtype=np.int32)
		img_points2 = np.array(img_points2, dtype=np.int32)
		for _i in range(0,len(matches)):
			print(img_points1[_i])
			
		fundamental_mat , mask = cv2.findFundamentalMat(points1 = img_points1, 
												points2 = img_points2,
												method = cv2.FM_RANSAC,ransacReprojThreshold = 1.,confidence  = 0.99,mask = None)
	    
		# print(fundamental_mat)
		# return fundamental_mat.astype(np.float64)
		# The essential matrix E, is calculated as:
		#         E = K_2^T * F * K_1
		# where K_2 is the camera of the second image,
		# K_1 is the camera of the first image and F
		# the fundamental matrix
		return camera_img2.astype(np.float64).T.dot(fundamental_mat.astype(np.float64)).dot(camera_img1.astype(np.float64))
	
	def calculate_possible_solutions(self, essential_matrix, camera):
		E = essential_matrix
		print("E: \n", E)
		W = np.array([0, -1, 0, 1,  0, 0, 0,  0, 1 ], dtype = np.float64).reshape(3,3)
		print("\n\nW:", W)
		U,D,V = np.linalg.svd(np.array(E, dtype=np.float64))
			# We calculated two ways you can have
			# the rotation matrix from the decomposition
			# in singular values ​​of the essential matrix
			#    * R = U·W·V^T
		R_uwvt = U.dot(W).dot(V.T)
			#    * R = U·W^T·V^T
		R_uwtvt = U.dot(W.T).dot(V.T)
			# The matrix "t" can be obtained from the last
			# U column, but the sign is unknown, so
			# four possible solutions appear depending on the value
			# of R and the sign of t
		T = U[:,-1].reshape(1,3).T
		newrow = [0,0,0,1]
		P_uwvt = camera.dot(np.hstack((R_uwvt, T)))
		P_uwvt = np.vstack([P_uwvt,newrow])
		self.extrinsic = P_uwvt
		self.extrinsic_inv = np.linalg.inv(self.extrinsic)

		P_neg_uwvt = camera.dot(np.hstack((R_uwvt, -T)))
		P_neg_uwvt = np.vstack([P_neg_uwvt,newrow])

		P_uwtvt = camera.dot(np.hstack((R_uwtvt, T)))
		P_uwtvt = np.vstack([P_uwtvt,newrow])

		P_neg_uwtvt = camera.dot(np.hstack((R_uwtvt, -T)))
		P_neg_uwtvt = np.vstack([P_neg_uwtvt,newrow])

		print("\nEstimated Cameras:")
		print("P = UWV^T:\n", P_uwvt, "\n")
		print("P = -UWV^T:\n", P_neg_uwvt, "\n")
		print("P = UW^TV^T:\n", P_uwtvt, "\n")
		print("P = -UW^TV^T:\n", P_neg_uwtvt, "\n")
		return P_uwvt, P_neg_uwvt, P_uwtvt, P_neg_uwtvt
	
	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB

		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		# return the visualization
		return vis
		
	def apply_disparity(self, img, disp):
		batch_size, _, height, width = img.size()
    	# Original coordinates of pixels
    	#DIr is always left
    	# x_base = np.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
    	# y_base = np.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)    
        # Apply shift in X direction
        # x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel

        # flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

        # In grid_sample coordinates are assumed to be between -1 and 1

        # output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')
		grid = meshgrid_abs(height, width)
		grid = tile(grid.unsqueeze(0), 0, batch_size)
		# camera = self._camera
		
		intrinsic_mat_inv = self.left_inv
		intrinsic_mat = self.right_intrinsic
		
		cam_coords = img2cam(disp, grid, intrinsic_mat_inv)  # [B x H * W x 3]
		cam_coords = np.concatenate((cam_coords,np.ones((batch_size, height * width, 1))),-1)  # [B x H * W x 4]
		
		extrinsic_mat = self.extrinsic
		world_coords = cam2world(cam_coords, extrinsic_mat)  # [B x H * W x 4]
		
		extrinsic_inv = self.extrinsic_inv
		other_cam_coords = world2cam(world_coords, extrinsic_inv)  # [B x H * W x 4]
		
		other_cam_coords = other_cam_coords[:, :, :3]  # [B x H * W x 3]
		
		other_image_coords = cam2img(other_cam_coords, intrinsic_mat)  # [B x H * W x 2]
		
		projected_img = self.spatial_transformer(img, other_image_coords)
		return projected_img

	def spatial_tranformer(self,img,coords):
		"""A wrapper over binlinear_sampler(), taking absolute coords as input."""
		batch,ch,height,width = img.shape
		px = coords[:, :, 0]
		py = coords[:, :, 1]
		# Normalize coordinates to [-1, 1] to send to _bilinear_sampler.
		px = px/(width-1)* 2.0- 1.0
		py = py / (height - 1) * 2.0 - 1.0
		output_img = self.bilinear_sampler(self.device, img, px, py)
		return output_img
	
	def bilinear_sampler(self,device, im, x, y):
		"""
		Perform bilinear sampling on im given list of x, y coordinates.
       	Implements the differentiable sampling mechanism with bilinear kernel
       	in https://arxiv.org/abs/1506.02025.
       	x,y are tensors specifying normalized coordinates [-1, 1] to be sampled on im.
       	For example, (-1, -1) in (x, y) corresponds to pixel location (0, 0) in im,
       	and (1, 1) in (x, y) corresponds to the bottom right pixel in im.

       	:param im: Batch of images with shape [B, h, w, channels].
       	:param x: Matrix of normalized x coordinates in [-1, 1], with shape [B, h, w].
       	:param y: Matrix of normalized y coordinates in [-1, 1], with shape [B, h, w].
       	:return: Sampled image with shape [B, C, H, W].
       	"""
       	x = np.reshape(x,-1)
       	y = np.reshape(y,-1)
       	# Constants.
       	batch_size, channels, height, width = im.shape

       	max_y = int(height - 1)
       	max_x = int(width - 1)

       	# Scale indices from [-1, 1] to [0, width - 1] or [0, height - 1].
       	x = (x + 1.0) * (width - 1.0) / 2.0
       	y = (y + 1.0) * (height - 1.0) / 2.0

       	# Compute the coordinates of the 4 pixels to sample from.
       	x0 = np.floor(x)
       	x1 = x0 + 1
       	y0 = np.floor(y)
       	y1 = y0 + 1

       	x0 = np.clip(x0, 0.0, max_x)
       	x1 = np.clip(x1, 0.0, max_x)
       	y0 = np.clip(y0, 0.0, max_y)
       	y1 = np.clip(y1, 0.0, max_y)
       	dim2 = width
       	dim1 = width * height

       	# Create base index.
       	base = np.arange(batch_size) * dim1
       	base = np.reshape(base, [-1, 1])
       

       	base_y0 = base + y0 * dim2
       	base_y1 = base + y1 * dim2
       	idx_a = base_y0 + x0
       	idx_b = base_y1 + x0
       	idx_c = base_y0 + x1
       	idx_d = base_y1 + x1
       	idx_a, idx_b, idx_c, idx_d = [_.type(np.float) for _ in [idx_a, idx_b, idx_c, idx_d]]

       # Use indices to lookup pixels in the flat image and restore channels dim.
       	im_flat = np.reshape(im, (-1, channels))
       	pixel_a = im_flat[idx_a]
       	pixel_b = im_flat[idx_b]
       	pixel_c = im_flat[idx_c]
       	pixel_d = im_flat[idx_d]

       # And finally calculate interpolated values.
       	wa = np.expand_dims(((x1 - x) * (y1 - y)), axis = 1)
       	wb = np.expand_dims((x1 - x) * (1.0 - (y1 - y)), axis = 1)
       	wc = np.expand_dims(((1.0 - (x1 - x)) * (y1 - y)), axis = 1)
       	wd = np.expand_dims(((1.0 - (x1 - x)) * (1.0 - (y1 - y))), axis = 1)

       	output = wa * pixel_a + wb * pixel_b + wc * pixel_c + wd * pixel_d
       	output = np.reshape(output, (batch_size, channels, height, width))
		return output