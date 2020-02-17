import cv2 

import sys

import numpy as np

import random as rng

from math import ceil, floor, pi

SCREEN_WIDTH=1920
SCREEN_HEIGHT=1080

# Defined seed mapping of average areas to seed types
# values scaled by 10 because whiterice/sorghum are so close
def getSeedMapping():
    return {'2506': 'canola',       #'3720':'canola',
            '36980':'soybean',
            '30100':'roughrice',
            '11469':'wheat',        #
            '12290':'whiterice',
            '5344':'sorghum'}

def getDistanceTransformMapping():
    return {'canola': {"dtThreshScale": 0.2, "numErode": 2},
            'soybean': {"dtThreshScale": 0.75, "numErode": 1},
            'roughrice': {"dtThreshScale": 30, "numErode": 1},
            'wheat': {"dtThreshScale": 0.1, "numErode": 0},
            'whiterice': {"dtThreshScale": 0.4, "numErode": 3},
            'sorghum': {"dtThreshScale": 0.2, "numErode": 2}}

#uses the static dictionaries with saved parameters
#parameter area is the average ground truth area, returns a dictionary with algorithm parameters
#relative to the seed classified in the first step
def getDistanceTransformThresh(area):

    mapping = getSeedMapping()
    dtMapping = getDistanceTransformMapping()

    for avgArea,seedType in mapping.items():

        k = int(avgArea)
        scaledArea = area*10.0

        print(scaledArea)        
        if scaledArea >= k-(k*0.1) and scaledArea <= k+(k*0.1):

            return dtMapping[seedType]

def enclosingBoundingRect(contour):
    min_x = 900 
    min_y = 1800

    max_h = -1
    max_w = -1

    #print(contour)
    for point in contour:
        point = point[0]
        max_h = max(max_h, point[1])
        max_w = max(max_w, point[0])

        min_x = min(min_x, point[0])
        min_y = min(min_y, point[1])

        #print(point)

   # print("done")
    return min_x, min_y, max_w, max_h

def stdDev(values):

    mean = float(sum(values))/len(values)
    variance = 0

    for v in values:
        variance += (v-mean)**2

    variance /= len(values)-1

    return variance ** (0.5)

#find the median value
#based on the median value find Q1 which is the median of the lower values
#similarly, find Q3 which is the median of the higher values
#Q2 is the overall median
#the interquartal range is Q3 - Q1 
#returns lo and hi thresholds defined as:
#lo = Q1 - 1.5*IQR
#hi = Q3 + 1.5*IQR
def quartileRange(values):

    size = len(values)
    sortedList = sorted(values, key=lambda x: x[0])
    Q2 = sortedList[size//2]

    lowerArray = sortedList[:size//2]
    upperArray = sortedList[size//2:]

    Q1 = lowerArray[len(lowerArray)//2][0]
    Q3 = upperArray[len(upperArray)//2][0]

    IQR = Q3-Q1

    return Q1-1.5*IQR, Q3+1.5*IQR

def interquartileReduce(pairs):

    lo, hi = quartileRange(pairs)

    output = []

    for area,contour in pairs:

        if area >= lo and area <= hi:

            output.append((area,contour))

    return output

def makeOdd(val):

    if val % 2 == 0:

        return val+1

    return val

#function to return contours that are non-clusters
#uses canny edge detection and an iterative interquartile range filtering algorithm to mine ground truths
def findGroundTruths(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    blur = cv2.medianBlur(thresh, 9)

    edges = cv2.Canny(blur, threshold1=200, threshold2=255)

    cv2.imwrite("canny.jpg", edges)

    #smoothed = cv2.GaussianBlur(edges, ksize=(5,5), sigmaX=10)

    #cv2.imwrite('smoothed.jpg', smoothed)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(img, contours, -1, (255,255,0), -1)

    #cv2.imwrite("output.jpg", img)

    areas = []

    for contour in contours:

        area = cv2.contourArea(contour)

        areas.append((area, contour))
    
    import pylab as pl
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    data = sorted(list(map(lambda x: x[0], areas)))
    fit = stats.norm.pdf(data, np.mean(data), np.std(data))
    pl.plot(data, fit, '-o')
    pl.hist(data,normed=True)
   # pl.show()

    print("Average Area: {}".format(mean(map(lambda x: x[0], areas))))

    for i in range(0,1):

        areas = interquartileReduce(areas)

    groundTruths = []

    gtAreas = []
    for area, contour in areas:

        gtAreas.append(area)
        groundTruths.append(contour)


    data = sorted(list(gtAreas))
    fit = stats.norm.pdf(data, np.mean(data), np.std(data))
    pl.plot(data, fit, '-o')
    pl.hist(data,normed=True)
    #pl.show()

    gtImg = img.copy()

    cv2.drawContours(gtImg, groundTruths, -1, (255,255,0), -1)

    for contour in groundTruths:
        
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv2.putText(gtImg, "{}".format(cv2.contourArea(contour)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1, cv2.LINE_AA)     


    cv2.imwrite("gt.jpg", gtImg)

    return groundTruths

#finding the mean
def calcMajorMinorAvgs(contours):
    
    majors = []
    minors = []

    for contour in contours:

        rect = cv2.minAreaRect(contour)
        majors.append(max(rect[1][0], rect[1][1]))
        minors.append(min(rect[1][0], rect[1][1]))

    return mean(majors), mean(minors)

def generateScales(groundTruths, erodedGroundTruths):
    sorted_gt = sorted(groundTruths, key = lambda x: cv2.contourArea(x))
    eroded_gt = sorted(erodedGroundTruths, key = lambda x: cv2.contourArea(x))

    scale_widths = []
    scale_heights = []

    for gt, egt in zip(sorted_gt, eroded_gt):
        rect_gt = cv2.minAreaRect(gt)
        rect_egt = cv2.minAreaRect(egt)

        scale_widths.append(rect_gt[1][0]/ rect_egt[1][0])
        scale_heights.append(rect_gt[1][1]/ rect_egt[1][1])

    return scale_widths, scale_heights


def ellipseArea(w, h):
    
    return pi*w*h


def groundTruthScales(image, contours):

    majors = []
    minors = []

    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, 0, (255,255,0), 1)

    cv2.imwrite("Masks/gt.png", mask)

    print("Ground truths for scales: {}".format(len(contours)))
    for index in range(0, len(contours)):

        gt_rect = cv2.minAreaRect(contours[index])
        gt_major = max(gt_rect[1][0], gt_rect[1][1])
        gt_minor = min(gt_rect[1][1], gt_rect[1][0])

        mask = np.zeros_like(image)

        cv2.drawContours(mask, contours, index, (255,255,0), -1)
        #cv2.imwrite("Masks/masks_index_{}.png".format(index), mask)

        mask = cv2.erode(mask, np.ones((3,3)), iterations=7)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        masked_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(mask, masked_contours, 0, 255, -1)

        #cv2.imwrite("Masks/masks_index_{}.png".format(index), mask)
        #print(len(masked_contours))

        eroded_gt = masked_contours.pop()

        eroded_rect = cv2.minAreaRect(eroded_gt)
        e_major = max(eroded_rect[1][0], eroded_rect[1][1])
        e_minor = min(eroded_rect[1][1], eroded_rect[1][0])

        if e_major > 0 and e_minor > 0:
            majors.append(gt_major/e_major)
            minors.append(gt_minor/e_minor)

    print(majors, minors)
    return mean(majors), mean(minors)

def interpolateValues(scales):
    
    widths = list(map(lambda x: scales[0], scales))
    heights = list(map(lambda x: scales[1], scales))

    minWidth = min(widths)
    maxWidth = max(widths)
    minHeight = min(heights)
    maxHeight = max(heights)



def mean(values):

    return float(sum(values))/len(values)

if __name__ == "__main__":

    path = sys.argv[1]

    outputPath = sys.argv[2]

    img = cv2.imread(path)

    groundTruths = findGroundTruths(img)

    # [(eroded_l, eroded_w, scale), ..., numGt]
    #scales = groundTruthScales(img, groundTruths)

    #gtMajorAvg, gtMinorAvg = calcMajorMinorAvgs(groundTruths)

    gtMajorScale, gtMinorScale = groundTruthScales(img, groundTruths)

    print(gtMajorScale, gtMinorScale)

    areas = map(lambda x: cv2.contourArea(x), groundTruths)

    print("GT Avg Area: {}".format(mean(areas)))
    #get parameters based on the identified seed
    parameters = getDistanceTransformThresh(mean(areas))

    print(parameters)
    #adaptiveRegion =float(sum(areas))/len(areas)

    #print(adaptiveRegion)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    cv2.imwrite('threshed.jpg', thresh)

    # noise removal
    kernel = np.ones((3,3),np.uint8)

    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel , iterations = 3)
    thresh = cv2.erode(thresh, kernel, iterations = 7)
    
    gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel, iterations=1)

    #gradient = cv2.dilate(gradient, kernel, iterations=1)
   # gradient = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel, iterations=1)

    cv2.imwrite('gradient.jpg', gradient)

    sure_bg = cv2.morphologyEx(thresh, cv2.MORPH_ELLIPSE, kernel, iterations=parameters['numErode'])
    #sure_bg = cv2.erode(thresh, kernel, iterations=parameters["numErode"])

    #sure_bg = cv2.medianBlur(sure_bg, 9)
    #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)

    #cv2.imwrite('noiseremoval.jpg', opening)

    # sure background area
    #sure_bg = cv2.dilate(opening,kernel,iterations=3)

   # cv2.imwrite('surebg.jpg', sure_bg)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,cv2.DIST_MASK_5)
    
    #print(dist_transform.max())

    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    #dist_transform = cv2.erode(dist_transform, kernel,iterations=1)

    cv2.imwrite('dt3.jpg', dist_transform*10)

    ret, sure_fg = cv2.threshold(dist_transform,0.1,255,0)

    cv2.imwrite('sure_fg.jpg', sure_fg)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(thresh,sure_fg)


    cv2.imwrite('unknown.jpg', unknown)

    sure_fg = cv2.subtract(thresh, gradient)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    cv2.imwrite('markers.jpg', markers)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[gradient==255] = 0

    markers = cv2.watershed(img,markers)

    img[markers == -1] = [0,0,255]

    #img[markers > 0] = [255,255,255]

    cv2.imwrite("outputPath.jpg", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('gray.jpg', gray)

    markers = markers.astype(np.uint8)

    cv2.imwrite('markersA.jpg', markers*1000)

    #ret, markers2 = cv2.threshold(markers, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    #adaptiveRegion = makeOdd(int(adaptiveRegion)/4)

    #markers2 = cv2.adaptiveThreshold(markers, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3, 1)
    ret, markers2 = cv2.threshold(markers, 1, 255, cv2.THRESH_BINARY)

    cv2.imwrite('markers1.jpg', markers2)

    contours, hierarchy = cv2.findContours(markers2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(img, contours, -1, (0, 0, 0))

    #print(hierarchy)

    #print(len(contours))
    actual_contours = []

    #print(hierarchy)
    for contour, hier in zip(contours, hierarchy[0]):
        # if hier[2] == -1 and not (hier[3] == -1):
        #     actual_contours.append(contours[hier[3]]) 
        if hier[2] != -1 or hierarchy[0][hier[2]][2] == -1:
            actual_contours.append(contour)
        elif heir[2] != -1 and cv2.contourArea(contour) > 300:
            actual_contours.append(contour)

    count = 0
    floorCount = 0
    ceilCount = 0


    #mask = cv2.CreateImage(img.shape, 8, 3)

    kernel = np.ones((3,3),np.uint8)


    for index, contour in enumerate(sorted(contours, key= lambda x: cv2.minAreaRect(x)[1][1])):
                
        if cv2.contourArea(contour) < 10000: 

            r = rng.randint(0,128)
            g = rng.randint(0,128)
            b = rng.randint(0,128)

            color = (r, g, b)

            #hull = cv2.convexHull(contour, returnPoints=False)
            points = cv2.convexHull(contour, returnPoints=True)
            #defects = cv2.convexityDefects(contour, hull)
            
            if len(points) >= 5:

                #print(points)

                #cv2.drawContours(img, points, -1, color, 10)

                # try:
                #     for d in range(defects.shape[0]):
                #         s,e,f,d = defects[d,0]
                #         start = tuple(contour[s][0])
                #         end = tuple(contour[e][0])
                #         far = tuple(contour[f][0])
                #         #cv2.line(img,start,end,(0,0,0),5)

                #         if d > 1000:
                #             #print(d)
                #             cv2.circle(img,far,5,(0,0,255),-1)


                # except:
                #     pass
                # identify the co-oridinates to draw a bounding rectangle around the identified contours

                rect = cv2.minAreaRect(contour)
                w = rect[1][0]
                h = rect[1][1]

                major = max(w,h)
                minor = min(w,h)

                #print(major, minor)

                #diffMajor = gtMajorAvg/major
                #diffMinor = gtMinorAvg/minor
                # diffMajor = abs(gtMajorAvg-major)
                # diffMinor = abs(gtMinorAvg-minor)

                #print(major+diffMajor, minor+diffMinor)

                if w == major:
                    rect = (rect[0], (major*gtMajorScale, minor*gtMinorScale), rect[2])
                else:
                    rect = (rect[0], (minor*gtMinorScale, major*gtMajorScale), rect[2])

                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (0,0,255), 1)

                cv2.putText(img, "{}".format(1.4*(((major*gtMajorScale)*(minor*gtMinorScale))/2)), (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1, cv2.LINE_AA)     

                #cv2.putText(img, "{}".format(((major*gtMajorScale)*(minor*gtMinorScale))/2), (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1, cv2.LINE_AA)     
                #cv2.putText(img, "{}".format(((major+diffMajor)*(minor+diffMinor))/2), (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1, cv2.LINE_AA)     
                #cv2.putText(img, "{}".format((major+diffMajor)*(minor+diffMinor)), (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1, cv2.LINE_AA)     


                #scale_w = scale_widths[len(scale_widths)// 2]
                #scale_h = scale_heights[len(scale_heights)// 2]



                #widths = sorted(widths)
                #heights = sorted(heights)
                #numGt = len(widths)



                # diff = abs(gtArea - w*h)
                # diffW = abs(gtWidthAvg-w)
                # diffH = abs(gtHeightAvg-h)

                #correctedW = rect[1][0] * scale_w
                #correctedH = rect[1][1] * scale_h
                #rect = (rect[0], (w, correctedH), rect[2])
                #print(diff)




                # cv2.drawContours(mask, contour, -1, [0, 255, 0], -1)

                # mask = cv2.dilate(mask, kernel, iterations = 7)

                

                # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                # dilated_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # cv2.drawContours(mask, dilated_contours, -1, 255, -1)
                # cv2.drawContours(img, dilated_contours, -1, 255, -1)

                # dilated_rect = cv2.minAreaRect(dilated_contours.pop())
                # dilated_w = rect[1][0]
                # dilated_h = rect[1][1]

                # box = cv2.boxPoints(dilated_rect)
                # box = np.int0(box)
                # cv2.drawContours(img, [box], -1, 255, 2)


                # cv2.imwrite("Masks/mask{}.jpg".format(count), mask)



                #dilated_rect = cv2.minAreaRect()

                #ellipse = cv2.fitEllipse(contour)
                #cv2.ellipse(img, ellipse, (0,255,0), 2)

                #x1, y1, x2, y2 = cv2.boundingRect(contour)
                
                # Draw a bounding rectangle around the identified contours
                #cv2.rectangle(img, (x1, y1), (x2 + x1, y1 + y2), (0, 255, 0), 1) 
                # #estimate = abs(correctedW*correctedH/gtArea)
                # print("{}-{}={}".format(correctedW*correctedH, gtArea, estimate))

                count += 1             
                # floorCount += floor(estimate)
                # ceilCount += round(estimate)

            # Plot the area of the contour (seed) next to the rectangle
                #cv2.putText(img, "{}".format(count), (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1, cv2.LINE_AA)     
        
    #print(count)

    #cv2.drawContours(img, contours, -1, (0,0,255), -1)

    # print("Seed Count is between: {}-{}".format(count, ceilCount))
    cv2.imwrite('contours.jpg', img)
