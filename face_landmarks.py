import numpy as np
import argparse
import os
import imutils
import face_utils
from collections import OrderedDict
import dlib
import cv2
import matplotlib.pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")

group1 = ap.add_mutually_exclusive_group()
group1.add_argument("-i", "--image", action = "store", help = "path to input image")
group1.add_argument("-v", "--video", action = "store_true", help = " video stream")
group1.add_argument("-l", "--visualize", action = "store_true", help = "visualizing facial landmarks")

group2 = ap.add_mutually_exclusive_group()
group2.add_argument("-m", "--mouth", action = "store_true", help = "visualizing mouth")
group2.add_argument("-reb", "--right_eyebrow", action = "store_true", help = "visualizing right-eyebrow")
group2.add_argument("-leb", "--left_eyebrow", action = "store_true", help = "visualizing left-eyebrow")
group2.add_argument("-re", "--right_eye", action = "store_true", help = "visualizing right eye")
group2.add_argument("-le", "--left_eye", action = "store_true", help = "visualizing left eye")
group2.add_argument("-n", "--nose", action = "store_true", help = "visualizing nose")
group2.add_argument("-j", "--jaw", action = "store_true", help = "visualizing jaw")

args = ap.parse_args()

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.shape_predictor)

specific_facial_landmark_index = OrderedDict()

if args.mouth:
	specific_facial_landmark_index["mouth"] = (48, 68)
elif args.right_eyebrow:
	specific_facial_landmark_index["right_eyebrow"] = (17, 22)
elif args.left_eyebrow:
	specific_facial_landmark_index["left_eyebrow"] = (22, 27)
elif args.right_eye:
	specific_facial_landmark_index["right_eye"] = (36, 42)
elif args.left_eye:
	specific_facial_landmark_index["left_eye"] = (42, 48)
elif args.nose:
	specific_facial_landmark_index["nose"] = (27, 35)
elif args.jaw:
	specific_facial_landmark_index["jaw"] = (0, 17)
else:
	specific_facial_landmark_index = face_utils.facial_landmarks_indexes

# load the input image, resize it, and convert it to grayscale
def face_detection(image, resize, video = False):
	if resize:
		image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the face number
		# cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		if video:
			for (x, y) in shape:
				cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
		else:
			image = visualize_facial_landmarks(image, shape)
		
	return image

# show the output image with facial landmarks
def displaying_and_saving_result_image(orig, image):
	plt.subplot(121)
	plt.imshow(orig)
	plt.xticks([])
	plt.yticks([])
	plt.title("Input")

	plt.subplot(122)
	plt.imshow(image)
	plt.xticks([])
	plt.yticks([])
	plt.title("Output")

	fname = name
	plt.savefig(fname)
	plt.show()

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	# create two copies of the input image -- one for the
	# overlay and one for the final output image
	overlay = image.copy()
	output = image.copy()
 
	# if the colors list is None, initialize it with a unique
	# color for each facial landmark region
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220)]

	# loop over the facial landmark regions individually
	for (i, name) in enumerate(specific_facial_landmark_index.keys()):
		# grab the (x, y)-coordinates associated with the
		# face landmark
		(j, k) = face_utils.facial_landmarks_indexes[name]
		pts = shape[j:k]
 
		# check if are supposed to draw the jawline
		if name == "jaw":
			# since the jawline is a non-enclosed facial region,
			# just draw lines between the (x, y)-coordinates
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv2.line(overlay, ptA, ptB, colors[i], 2)
 
		# otherwise, compute the convex hull of the facial
		# landmark coordinates points and display it
		else:
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay, [hull], -1, colors[i], -1)
	
	# apply the transparent overlay
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
 
	# return the output image
	return output

if args.video is False and args.visualize is False:
	name = "result_" + args.image[1]
	image = plt.imread(args.image)
	orig = image
	image = face_detection(orig, resize = True)
	displaying_and_saving_result_image(orig, image)
elif args.video:
	cap = cv2.VideoCapture(0)
	while True:
		ret, frame = cap.read()
		image = face_detection(frame, resize = False, video = True)
		cv2.imshow("video", image)
		if cv2.waitKey(1) == ord("q"):
			break
	cap.release()
	cv2.destroyAllWindows()
elif args.visualize:
	cap = cv2.VideoCapture(0)
	while True:
		ret, frame = cap.read()
		image = face_detection(frame, resize = False)
		cv2.imshow("video", image)
		if cv2.waitKey(1) == ord("q"):
			break
	cap.release()
	cv2.destroyAllWindows()






