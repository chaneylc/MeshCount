import sys
import cv2
import numpy
from scipy.ndimage import label
import random as rng
def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=3)
    border = border - cv2.erode(img, None, iterations=1)

    dt = cv2.distanceTransform(img, 2, 5)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)

    cv2.imwrite('dt3.jpg', dt)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now. 
    lbl[border == 255] = 255

    lbl = lbl.astype(numpy.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(numpy.uint8)
    return 255 - lbl


img = cv2.imread(sys.argv[1])

# Pre-processing.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
_, img_bin = cv2.threshold(img_gray, 0, 255,
        cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imwrite('thresh.jpg', img_bin)

img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
        numpy.ones((3, 3), dtype=int))

cv2.imwrite('thresh.jpg', img_bin)

result = segment_on_dt(img, img_bin)
cv2.imwrite(sys.argv[2], result)

result[result != 255] = 0
result = cv2.dilate(result, None)

contours, _ = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
count = 0

cv2.namedWindow("test", cv2.WINDOW_NORMAL)
for index, c in enumerate(contours):

    mask = numpy.zeros_like(img)
    r = rng.randint(0,128)
    g = rng.randint(0,128)
    b = rng.randint(0,128)

    color = (r, g, b)
    print(cv2.contourArea(c))
    if cv2.contourArea(c) >= 500 and cv2.contourArea(c) < 10000:

        hull = cv2.convexHull(c, True)

        if len(c)/len(hull) <= 5:

            M = cv2.moments(c)

            if M["m00"] > 0:

                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                cv2.drawContours(img, contours, index, color, 1)
                count += 1
                cv2.putText(img, "{}".format(count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1, cv2.LINE_AA)     
                #cv2.imshow("test", mask)
                #cv2.waitKey(0)
#img[result == 255] = (0, 0, 255)
print(count)
cv2.imwrite(sys.argv[3], img)