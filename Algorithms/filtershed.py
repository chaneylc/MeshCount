import cv2
import numpy as np
import random as rng
import sys

src = cv2.imread(sys.argv[1])

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

_, output = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.asarray([[2,1,2],[1,2,1],[2,1,2]])
#kernel = np.asarray([[0,1,0],[1,-4,1],[0,1,0]])
#kernel = np.asarray([[2,2,2],[2,-1,2],[2,2,2]])
#kernel = np.asarray([[-1,2,2],[2,-1,2],[2,2,-1]])
#kernel = np.asarray([[-1,2,2],[2,-1,2],[2,2,-1]])

#kernel = 
#kernel = np.asarray([[2,-1,-1],[2,2,-1],[2,2,2]])
#kernel2 = np.asarray([[-1,-1,2],[-1,2,2],[2,2,2]])

cv2.namedWindow('test', cv2.WINDOW_NORMAL)
for i in range(0,100):

    #g = cv2.GaussianBlur(output, ksize=(3,3), sigmaX=3)

    output = cv2.filter2D(output, -1, kernel, anchor=(0,0))+cv2.filter2D(output, -1, kernel, anchor=(0,1))+cv2.filter2D(output, -1, kernel, anchor=(0,2))+cv2.filter2D(output, -1, kernel, anchor=(1,0))+cv2.filter2D(output, -1, kernel, anchor=(1,1))+cv2.filter2D(output, -1, kernel, anchor=(1,2))+cv2.filter2D(output, -1, kernel, anchor=(2,0))+cv2.filter2D(output, -1, kernel, anchor=(2,1))+cv2.filter2D(output, -1, kernel, anchor=(2,2))

    cv2.imshow('test', output)
    cv2.waitKey(5000)

#output = cv2.adaptiveThreshold(output, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)
# for i in range(0,30):

#     output = cv2.GaussianBlur(output, ksize=(3,3), sigmaX=3)
   
#     #prev = cv2.GaussianBlur(prev, ksize=(3,3), sigmaX=3)
    
#     #cv2.imshow('test', output)
#     #cv2.waitKey(0)

# output = np.bitwise_not(output)
ouput = gray-output
contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
for contour in contours:

    r = rng.randint(0,255)
    g = rng.randint(0,255)
    b = rng.randint(0,255)

    cv2.drawContours(src, [cv2.approxPolyDP(contour, 0.05*cv2.arcLength(contour,True),True)], -1, (r,g,b), 2)

#_, output = cv2.threshold(output, 128, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('test', src)
cv2.waitKey(5000)
cv2.imwrite("test.jpg", src) #cv2.addWeighted(output, 0.5, src, 0.9, 1))