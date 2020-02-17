import cv2 

import sys

import numpy as np

import random as rng
from matplotlib import pyplot as plt

src = cv2.imread(sys.argv[1], 0)


kernel = np.asarray([[1,1,1],[1,1,1],[1,1,1]])
output = cv2.filter2D(src, -1, kernel)

# kernel2 = np.asarray([[8,8,8],[1,1,1],[1,1,1]])
# output += cv2.filter2D(src, -1, kernel2)

#TL
kernel2 = np.asarray([[8,8,1],[8,1,1],[1,1,1]])
output += cv2.filter2D(src, -1, kernel2)

#TR
kernel2 = np.asarray([[1,8,8],[1,1,8],[1,1,1]])
output += cv2.filter2D(src, -1, kernel2)

#BL
kernel2 = np.asarray([[1,1,1],[8,1,1],[8,8,1]])
output += cv2.filter2D(src, -1, kernel2)

#BR
kernel2 = np.asarray([[1,1,1],[1,1,8],[1,8,8]])
output += cv2.filter2D(src, -1, kernel2)

kernel2 = np.asarray([[1,1,1],[1,8,1],[1,1,1]])
output += cv2.filter2D(src, -1, kernel2)


# kernel = np.asarray([[1,1,1],[1,1,1],[8,8,8]])
# output = cv2.filter2D(src, -1, kernel)

# kernel2 = np.asarray([[8,8,8],[1,1,1],[1,1,1]])
# output += cv2.filter2D(src, -1, kernel2)

# kernel3 = np.asarray([[8,1,1],[8,1,1],[1,8,1]])
# output += cv2.filter2D(src, -1, kernel2)

# kernel4 = np.asarray([[1,1,8],[1,8,1],[1,1,8]])
# output += cv2.filter2D(src, -1, kernel2)

# kernel2 = np.asarray([[1,1,1],[8,1,1],[8,8,1]])
# output += cv2.filter2D(src, -1, kernel2)

# kernel2 = np.asarray([[1,1,1],[1,1,8],[1,8,8]])

# kernel2 = np.asarray([[1,8,8],[1,1,8],[1,1,1]])
# output += cv2.filter2D(src, -1, kernel2)

# kernel2 = np.asarray([[1,1,1],[1,8,1],[1,1,1]])
# output += cv2.filter2D(output, -1, kernel2)

#output = cv2.filter2D(output, -1, kernel)

#output = cv2.Laplacian(output, -1)

#output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, np.ones((3,3)), iterations=1)

#output = cv2.GaussianBlur(output, ksize=(3,3), sigmaX=10)

#contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#print(len(contours))

#cv2.drawContours(src, contours, -1, (255,255,255), 1)

cv2.imwrite('sobel.jpg', output)