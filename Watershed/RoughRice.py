import cv2 

import sys

import numpy as np

import random as rng

from math import ceil, floor, pi

import pylab as pl
import scipy.stats as stats
import matplotlib.pyplot as plt

SCREEN_WIDTH=1920
SCREEN_HEIGHT=1080

# Defined seed mapping of average areas to seed types
# values scaled by 10 because whiterice/sorghum are so close
def getSeedMapping():
    return {'2506': 'canola',       #'3720':'canola',
            '36980':'soybean',
            '30100':'roughrice',
            '11469':'wheat',        #
            '8712':'whiterice',
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
    
    gray = cv2.GaussianBlur(gray, sigmaX=5, ksize=(3,3))

    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    mask = np.zeros_like(img)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for index in range(0,len(contours)):
        hull = cv2.convexHull(contours[index], True)

        if len(contours[index])/len(hull) <= 3:

        #print(len(c)/len(hull))
            cv2.drawContours(mask, contours, index, (128,128,128), -1)

    cv2.imwrite("gt_thresh.png", mask)

    blur = cv2.medianBlur(mask, 9)

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

        M = cv2.moments(contour)

        if M["m00"] > 0:

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            #print(img[cY][cX])

            avgVal = mean(img[cY][cX])
            if avgVal < 170:
            
                areas.append((area, contour))
    
    

    data = sorted(list(map(lambda x: x[0], areas)))
    fit = stats.norm.pdf(data, np.mean(data), np.std(data))
    pl.plot(data, fit, '-o')
    pl.hist(data,normed=True)
    #pl.show()

    #print("Average Area: {}".format(mean(map(lambda x: x[0], areas))))

    for i in range(0,1):

        areas = interquartileReduce(areas)

    groundTruths = []

    gtAreas = []
    groundTruthConvexity = []
    for area, contour in areas:

        gtAreas.append(area)
        groundTruths.append(contour)

        hull = cv2.convexHull(contour, True)
        polyDP = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        groundTruthConvexity.append(len(polyDP)/ len(hull))

    groundTruthConvexity = mean(groundTruthConvexity)
    data = sorted(list(gtAreas))
    fit = stats.norm.pdf(data, np.mean(data), np.std(data))
    pl.plot(data, fit, '-o')
    pl.hist(data,normed=True)
    #pl.show()

    gtImg = img.copy()

    cv2.drawContours(gtImg, groundTruths, -1, (255,255,0), -1)
    cv2.imwrite("gt.jpg", gtImg)

    groundTruthDefectsDist = 0.0
    maxContour = None
    defectVals = []
    for index, contour in enumerate(groundTruths):
        
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull):
            defects = cv2.convexityDefects(contour, hull)
            
            #maxD = max(defects, key=lambda x: x[0][3])[0][3]
            #avg = mean(list(map(lambda x: x[0][3], defects)))
            #print(maxD-avg)
            for d in defects:
                defectVals.append(d[0][3])
                #print(d[0][3])
                #prev = groundTruthDefectsDist
                #groundTruthDefectsDist = max(groundTruthDefectsDist, d[0][3])
                #if prev != groundTruthDefectsDist:
                 #   maxContour = index


        #cv2.putText(gtImg, "{}".format(cv2.contourArea(contour)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1, cv2.LINE_AA)     

    # M = cv2.moments(maxContour)
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])
    # cv2.drawContours(gtImg, groundTruths, maxContour, 255, -1)
    cv2.imwrite("gt.jpg", gtImg)

    median = sorted(defectVals)
    median = median[len(median)//2]
    return groundTruths, groundTruthConvexity, stdDev(defectVals), median

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

        mask = cv2.erode(mask, np.ones((3,3)), iterations=3)

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

def threshold(src):

    dst = src.copy()
    lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
    R = [(0, 153),(0, 255),(0, 255)]
    red_range = np.logical_and(R[0][0] < lab[:,:,0], lab[:,:,0] < R[0][1])
    green_range = np.logical_and(R[1][0] < lab[:,:,1], lab[:,:,1] < R[1][1])
    blue_range = np.logical_and(R[2][0] < lab[:,:,2], lab[:,:,2] < R[2][1])
    valid_range = np.logical_and(red_range, green_range, blue_range)

    lab[valid_range] = 200
    lab[np.logical_not(valid_range)] = 0

    return lab



def mean(values):

    return float(sum(values))/len(values)

if __name__ == "__main__":

    path = sys.argv[1]

    outputPath = sys.argv[2]

    img = cv2.imread(path)

    groundTruths, groundTruthConvexity, defectStdDev, defectMean = findGroundTruths(img)

    print(defectStdDev)

    # [(eroded_l, eroded_w, scale), ..., numGt]
    #scales = groundTruthScales(img, groundTruths)

    #gtMajorAvg, gtMinorAvg = calcMajorMinorAvgs(groundTruths)

    gtMajorScale, gtMinorScale = groundTruthScales(img, groundTruths)

    print(gtMajorScale, gtMinorScale)

    gtAvgArea = mean(list(map(lambda x: cv2.contourArea(x), groundTruths)))

    print("GT Avg Area: {}".format(gtAvgArea))
    #get parameters based on the identified seed
    parameters = getDistanceTransformThresh(gtAvgArea)

    print(parameters)
    #adaptiveRegion =float(sum(areas))/len(areas)
    #print(adaptiveRegion)
    kernel = np.ones((3,3),np.uint8)


    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #gray = cv2.GaussianBlur(gray, sigmaX = 5, ksize = (3, 3))
    
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    cv2.imwrite('threshed.jpg', thresh)


    # noise removal

    thresh = cv2.erode(thresh, kernel, iterations = 4)
    
    #gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel, iterations=1)

    #gradient = cv2.dilate(gradient, kernel, iterations=1)
   # gradient = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel, iterations=1)

    #cv2.imwrite('gradient.jpg', gradient)

    sure_bg = cv2.dilate(thresh, kernel, iterations=5)

    #sure_bg = cv2.morphologyEx(thresh, cv2.MORPH_ELLIPSE, kernel, iterations=parameters['numErode'])
    #sure_bg = cv2.erode(thresh, kernel, iterations=parameters["numErode"])

    #sure_bg = cv2.medianBlur(sure_bg, 9)
    #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)

    #cv2.imwrite('noiseremoval.jpg', opening)

    # sure background area
    #sure_bg = cv2.dilate(opening,kernel,iterations=3)

    cv2.imwrite('surebg.jpg', sure_bg)

    # Finding sure foreground area
    dt = cv2.distanceTransform(thresh,cv2.DIST_L2,cv2.DIST_MASK_5)
    
    #print(dt.max())
    #dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)

    #cv2.imwrite('dt3.jpg', dt)
    #_, sure_fg = cv2.threshold(dt, 75, 255, cv2.THRESH_BINARY)
    cv2.normalize(dt, dt, 0, 1.0, cv2.NORM_MINMAX)

    #dist_transform = cv2.erode(dist_transform, kernel,iterations=1)

    cv2.imwrite('dt3.jpg', dt*1000)

    ret, sure_fg = cv2.threshold(dt,0.2,255,0)

    cv2.imwrite('sure_fg.jpg', sure_fg)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)


    cv2.imwrite('unknown.jpg', unknown)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    cv2.imwrite('markers.jpg', markers)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)

    markers[markers==-1]=0
    markers = markers.astype('uint8')

    markers = 255-markers
    #img[markers == -1] = [0,0,255]
    markers[markers!=255] = 0

    #img[markers == 255] = [0, 255, 0]
    #cv2.dilate(markers,None, iterations=2)
    #img[markers > 0] = [255,255,255]

    #markers = markers.astype(np.uint8)

    cv2.imwrite('markersA.jpg', markers*1000)

    #ret, markers2 = cv2.threshold(markers, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    #adaptiveRegion = makeOdd(int(adaptiveRegion)/4)

    #markers2 = cv2.adaptiveThreshold(markers, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3, 1)
    #ret, markers2 = cv2.threshold(markers, 1, 255, cv2.THRESH_BINARY)

    #cv2.imwrite('markers1.jpg', markers2)

    contours, hierarchy = cv2.findContours(markers, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    #cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    mask = np.zeros_like(img)
    contour_list = []
    floorCount = 0
    roundCount = 0
    areaFloorCount = 0
    areaRoundCount = 0
    for index, c in enumerate(contours):

        area = cv2.contourArea(c)
        mask = np.zeros_like(img)
             
        r = rng.randint(0,128)
        g = rng.randint(0,128)
        b = rng.randint(0,128)

        color = (r, g, b)

        if hierarchy[0][index][2] == -1 and area < 500000 and area > 500:
            #print(groundTruthDefectDist)
            #print(area)
            #cv2.drawContours(img, contours, hierarchy[0][index][2], 255, 5)
            cv2.drawContours(img, contours, index, 255, 2)

            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #cv2.drawContours(img, [box], 0, (0,0,255), 1)

            hull = cv2.convexHull(c, True)
            hull2 = cv2.convexHull(c, returnPoints=False)

            areaPercent = cv2.contourArea(c)/gtAvgArea

            minX = int(rect[0][0])
            minY = int(rect[0][1])
            maxX = minX + int(rect[1][0])
            maxY = minY + int(rect[1][1])

            pixels = []

            found = False
            for x in range(minX, maxX):
                if found:
                    break
                for y in range(minY, maxY):
                    if(cv2.pointPolygonTest(c, (x, y), False) > 0):
                        pixels.append(mean(img[y][x]))
                        #found = True
                        #break
            if mean(pixels) < 100:

                print(mean(pixels))
                if len(hull2):
                    defVals = []
                    defects = cv2.convexityDefects(c, hull2)
                    for d in defects:
                        defVals.append(d[0][3])




                    #print(stdDev(defVals))
                    if mean(defVals) > defectMean + 4*defectStdDev:
                        roundCount += round(areaPercent)
                        cv2.putText(img, "X", (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1, cv2.LINE_AA)         
                        cv2.drawContours(mask, contours, index, 255, -1)
                        cv2.imwrite("defectmask.jpg", mask)
                        data = sorted(defVals)
                        fit = stats.norm.pdf(data, np.mean(data), np.std(data))
                        print(fit)
                        pl.plot(data, fit, '-o')
                        pl.hist(data,normed=True)
                        pl.show()
                    else:
                        roundCount += 1
                        #print(d)
                        #defVals.append(d[0][3])

                    #stddev = stdDev(defVals)
                    #if stddev > 500:
                        #cv2.putText(img, "{}".format(stddev)[:4], (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1, cv2.LINE_AA)         

               # polyDP = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
               
                areaRoundCount += round(areaPercent)
                floorCount += 1

            #if convexity <= .9: #groundTruthConvexity*0.8 and convexity <= groundTruthConvexity * 1.5 :
            
            #print(cv2.contourArea(c)/gtAvgArea)
            #rect = cv2.minAreaRect(c)
            #cv2.drawContours(img, contours, index, [255, 0, 0], 3)
            #cv2.putText(img, "{}".format(convexity)[:4], (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1, cv2.LINE_AA)         
    print("count: {}-{}".format(floorCount, roundCount))
    print("area based count: {}-{}".format(floorCount, areaRoundCount))
    print(set(sorted(contour_list)))


    cv2.imwrite('contours.jpg', img)
