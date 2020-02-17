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


#function to calculate the mean given an array of values
def mean(values):

    return float(sum(values))/len(values)

#function to calculate the standard deviation given an array of values
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

#filter outlayers from the pairs parameter
def interquartileReduce(pairs):

    lo, hi = quartileRange(pairs)

    output = []

    for area,contour in pairs:

        if area >= lo and area <= hi:

            output.append((area,contour))

    return output

#function to return contours that are non-clusters
#uses canny edge detection and an iterative interquartile range filtering algorithm to mine ground truths
def findGroundTruths(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray, sigmaX=5, ksize=(3,3))

    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #use a mask to draw all contours that have a certain convexity, attempts to eliminate contours
    #which typically have a very high #contour points / # of convex hull points ratio
    mask = np.zeros_like(img)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for index in range(0,len(contours)):
        hull = cv2.convexHull(contours[index], True)

        if len(contours[index])/len(hull) <= 3:

            cv2.drawContours(mask, contours, index, (128,128,128), -1)

    cv2.imwrite("gt_thresh.png", mask)

    blur = cv2.medianBlur(mask, 9)

    edges = cv2.Canny(blur, threshold1=200, threshold2=255)

    cv2.imwrite("canny.jpg", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    areas = []
    
    #filter any inner contours where we assume the background is white of intensity 170
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
    
    # vals = []
    # for x in range(0,img.shape[0]):
    #     for y in range(0,img.shape[1]):
    #         vals.append(mean(img[x][y]))

    #debug data to plot histogram of areas
    #data = sorted(list(map(lambda x: x[0], vals)))
    # data = sorted(vals)
    # fit = stats.norm.pdf(data, np.mean(data), np.std(data))
    # pl.plot(data, fit, '-o')
    # pl.hist(data,normed=True)
    # pl.show()

    #print("Average Area: {}".format(mean(map(lambda x: x[0], areas))))


    #use interquartile reduce to remove outliers
    areas = interquartileReduce(areas)

    groundTruths = []

    #gtAreas = []
    for area, contour in areas:

        #gtAreas.append(area)
        groundTruths.append(contour)

    #debug histogram plot of groundtruth areas
    # data = sorted(list(gtAreas))
    # fit = stats.norm.pdf(data, np.mean(data), np.std(data))
    # pl.plot(data, fit, '-o')
    # pl.hist(data,normed=True)
    #pl.show()


    #output image to visualize mined ground truths
    gtImg = img.copy()

    cv2.drawContours(gtImg, groundTruths, -1, (255,255,0), -1)
    cv2.imwrite("gt.jpg", gtImg)

    defectVals = []
    
    for index, contour in enumerate(groundTruths):
        
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        hull = cv2.convexHull(contour, returnPoints=False)
        
        if len(hull):
            
            defects = cv2.convexityDefects(contour, hull)
            
            for d in defects:
                
                defectVals.append(d[0][3])
                
        #cv2.putText(gtImg, "{}".format(cv2.contourArea(contour)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1, cv2.LINE_AA)     

    #cv2.imwrite("gt.jpg", gtImg)

    defectMean = sorted(defectVals)
    defectMean = defectMean[len(defectMean)//2]
    return groundTruths, stdDev(defectVals), defectMean


#driver function
if __name__ == "__main__":

    path = sys.argv[1]

    #outputPath = sys.argv[2]

    img = cv2.imread(path)

    groundTruths, defectStdDev, defectMean = findGroundTruths(img)

    #print(defectStdDev)

    gtAvgArea = mean(list(map(lambda x: cv2.contourArea(x), groundTruths)))

    #print("GT Avg Area: {}".format(gtAvgArea))

    kernel = np.ones((3,3),np.uint8)


    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    cv2.imwrite('threshed.jpg', thresh)

    thresh = cv2.erode(thresh, kernel, iterations = 4)

    sure_bg = cv2.dilate(thresh, kernel, iterations=5)

    cv2.imwrite('surebg.jpg', sure_bg)

    dt = cv2.distanceTransform(thresh,cv2.DIST_L2,cv2.DIST_MASK_5)
    
    cv2.normalize(dt, dt, 0, 1.0, cv2.NORM_MINMAX)

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


    #run watershed on blurred img to reduce inner contour noise and 
    #avoid over segmentation
    markers = cv2.watershed(cv2.GaussianBlur(img, sigmaX=5, ksize=(3,3)),markers)
    cv2.imwrite('markers.jpg', markers)



    #draw boundaries of watershed segments
    markers[markers==-1]=0
    markers = markers.astype('uint8')

    cv2.imwrite('markers_1.jpg', markers)

    markers = 255-markers

    cv2.imwrite('markers_3.jpg', markers)
    #img[markers == -1] = [0,0,255]
    markers[markers!=255] = 0

    cv2.imwrite('markers_2.jpg', markers)


    #use find contours to accumulate the clusters/single seeds
    contours, hierarchy = cv2.findContours(markers, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    mask = np.zeros_like(img)
    contour_list = []
    floorCount = 0
    roundCount = 0
    areaFloorCount = 0
    areaRoundCount = 0
    


    #use find contours hierarchy to find all child contours that are not considered noise
    new_parents = []
    for index, c in enumerate(contours):
        
        children = []
        area = cv2.contourArea(c)
        
        if area < 500000:
            
            for i, hier in enumerate(hierarchy[0]):
               
                if hier[3] == index:
                  
                    childArea = cv2.contourArea(contours[i])

                    if childArea > gtAvgArea*.25:
                     
                        new_parents.append(contours[i])
                        

    #countingmethod to estimate total count based on total area of children
    childSum = float(sum(map(lambda x: cv2.contourArea(x), new_parents)))
    print("CHILD SUM")
    print(childSum/gtAvgArea)

    #iterate over all current contours
    #perform area count and defects-based count
    for index, c in enumerate(new_parents):

        area = cv2.contourArea(c)
       # mask = np.zeros_like(img)
             
        r = rng.randint(0,128)
        g = rng.randint(0,128)
        b = rng.randint(0,128)
        color = (r, g, b)


        if area < 500000: 

            #segment to draw rotated bounding boxes
            rect = cv2.minAreaRect(c)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            #cv2.drawContours(img, [box], 0, (0,0,255), 1)

            hull2 = cv2.convexHull(c, returnPoints=False)

            #estimate the contour based on average ground truth area
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
                        
            if len(pixels) and mean(pixels) < 100:

                if len(hull2):
                    
                    defVals = []
                    defects = cv2.convexityDefects(c, hull2)
                    
                    if isinstance(defects, np.ndarray):
                        
                        for d in defects:
                           
                            defVals.append(d[0][3])

                        #print(stdDev(defVals))

                        #classify the contour as a cluster if it is greater than the 4th standard deviation of the mean
                        if mean(defVals) > defectMean + 4*defectStdDev:
                            
                            roundCount += round(areaPercent)
                            #cv2.putText(img, "X", (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1, cv2.LINE_AA)         
                            

                            #debug statement to view distribution of defects
                            #cv2.drawContours(mask, new_parents, index, 255, -1)
                            #cv2.imwrite("defectmask.jpg", mask)
                            #data = sorted(defVals)
                            #fit = stats.norm.pdf(data, np.mean(data), np.std(data))
                            #print(fit)
                            #pl.plot(data, fit, '-o')
                            #pl.hist(data,normed=True)
                            #pl.show()
                        else:

                            #if it is not considered a defects cluster, count it as one
                            roundCount += 1
               
                #area count always estimates contour count by ground truth area
                areaRoundCount += round(areaPercent)

                floorCount += 1

           
                cv2.drawContours(img, new_parents, index, [r,g,b], -1)
                cv2.putText(img, "X", (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 1, cv2.LINE_AA)   




    #find which count was the max/min to output correct range (might not be consistent due to over segmentations and cluster estimation rounding)
    maxCount = max(floorCount, roundCount)
    minCount = min(floorCount, roundCount)

    print("count: {}-{}".format(minCount, maxCount))


    maxCount = max(floorCount, areaRoundCount)
    minCount = min(floorCount, areaRoundCount)
    print("area based count: {}-{}".format(minCount, maxCount))

    cv2.imwrite('contours.jpg', img)
