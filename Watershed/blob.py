import cv2

import sys

import numpy as np

src = cv2.imread(sys.argv[1],0)

src = cv2.GaussianBlur(src, sigmaX=5, ksize=(3,3))
thresh = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
# Show blobs
cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL)
cv2.imshow("Keypoints", thresh)
cv2.waitKey(0)