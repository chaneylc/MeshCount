import cv2 as cv
import numpy as np
import math
from collections import defaultdict
from os import listdir

"""
Average soybean diameter: 5-7mm

Soybean sampled for skewness error: ~6mm in diameter

pi*r^2 = area in mm

area = ~28.2743338823mm

diameter of penny = 18.9mm

area of penny = 280.552077947mm

area of soybean mm / area of soybean pix = 0.00148839701

area of penny mm / area of penny pix = 0.00203227919
"""

OUTPUT_DIR = "/home/chaneylc/Desktop/MeshCount/"
#OUTPUT_DIR = r"C:\Users\marven\Documents\Spring-2020\CIS690\MeshCount"

def writeImg(name, img):
	cv.imwrite("{}/{}".format(OUTPUT_DIR, name), img)

#input is image using first-version soybeans mesh
#threshold function to find 3d printed mesh from an RGB image 1980x2592
#return is a blue background with red-masked mesh
def threshold(src):

    dst = src.copy()
    lab = cv.cvtColor(dst, cv.COLOR_RGB2LAB)

    #LAB threshold ranges that was found manually to work with
    #the lightbox
    L = [(0,33),(0,255),(0,255)]
    

    #use numpy logical and to find all pixels that satify the LAB range for each channel
    L_range = np.logical_and(L[0][0] < lab[:,:,0], lab[:,:,0] < L[0][1])
    a_range = np.logical_and(L[1][0] < lab[:,:,1], lab[:,:,1] < L[1][1])
    b_range = np.logical_and(L[2][0] < lab[:,:,2], lab[:,:,2] < L[2][1])

    valid_range = np.logical_and(L_range, a_range, b_range)

    #set all pixels
    lab[valid_range] = 0
    lab[np.logical_not(valid_range)] = 255

    #convert the image back to RGB
    rgb = cv.cvtColor(lab, cv.COLOR_LAB2RGB)

    #Debug statements to visualize the image outpu
    writeImg("mesh.jpg", rgb)

    return rgb


#function to replace a certain channel with value between a range
def replaceChannelValue(src, channelIndex, lo, hi, value):

	#src is the input image
	dst = src.copy()

	#all channel values specified from parameter
	values = dst[:,:,channelIndex]

	#indices that satisfy the given range from lo to hi
	indices = np.where((values>lo) & (values<=hi))

	#set the output image where the indices are true to the value given s.a black [0,0,0]
	dst[indices] = value

	return dst

def calcVariance(values):

	print(values)

	mean = sum(values) / float(len(values))

	print(mean)

	variance = 0
    
	for value in values:

		variance += (value-mean)**2

	variance /= len(values)

	print(variance)

	print(variance ** (0.5))

	return variance ** (0.5)

def  calcCircularity(area, perimeter):

	return 4 * (math.pi) * (area / (perimeter**2))

#pi * r^2 = x
#r = sqrt(x/pi)

#Cpx is the area in pxs of the ground truth coin
#Cmm is the known area in mm of the ground truth coin
#Kpx is the opencv contour area of the to-be measured contour
# def measureArea(Cpx, Cmm, Kpx):

# 	rmm = (Cmm / math.pi) ** (0.5)
# 	rpx = (Cpx / math.pi) ** (0.5)
# 	kpx = (Kpx / math.pi) ** (0.5)

# 	kmm = (float(kpx) * float(rmm)) / float(rpx)

# 	print(math.pi * (kmm ** 2))

# 	return math.pi * (kmm ** 2)


#check variance of area and circularities are within some threshold
def coinRecognition(coins):

	areas = []
	circus = []

	print(coins)
	for area in coins:

		peri = cv.arcLength(areaMap[area], True)

		areas.append(area)
		circus.append(calcCircularity(area, peri))

	areaVar = calcVariance(areas)
	circVar = calcVariance(circus)

	print(circus)

	print("Variances:")
	print(areaVar)
	print(circVar)

	areaThreshold = (696.019755467 * 0.9)
	areaLow = 696.019755467 - areaThreshold
	areaHigh = 696.019755467 + areaThreshold

	circuThreshold = 0.00163288510892 * 0.9
	circuLow = 0.00163288510892 - circuThreshold
	circuHigh = 0.00163288510892 + circuThreshold

	return True, ""

	if areaVar >= areaLow and areaVar <= areaHigh:

		if circVar >= circuLow and circVar <= circuHigh:

			return True, ""

		else:

			return False, "Circularity variance threshold failed."


	return False, "Area variance threshold failed."


def measureArea(Cpx, Cmm, Kpx):

	return (float(Kpx)*float(Cmm)) / float(Cpx)

#driver code
if __name__ == "__main__":

	for index, image in enumerate(listdir("/home/chaneylc/Desktop/MeshCount/Test"), 1):
		
		print(image)

		imagePath = "/home/chaneylc/Desktop/MeshCount/Test/{}".format(image)

		#read the input file
		src = cv.imread(imagePath)

		writeImg("gray.png", src)

		output = src.copy()

		src = threshold(src)

		src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)

		src = cv.medianBlur(src, 9)

		edges = cv.Canny(src, threshold1=200, threshold2=255)

		writeImg("edges.png", edges)

		edges_smoothed = cv.GaussianBlur(edges, ksize=(5,5), sigmaX=10)

		contours, _ = cv.findContours(edges_smoothed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

		print(len(contours))

		#assume that there will be four equally sized coins that are bigger than the measured seeds
		#first create an mapping of all areas
		areaMap = defaultdict(int)

		for contour in contours:
			
			area = cv.contourArea(contour)
			
			areaMap[area] = contour

		#sort the keys by the area values, top 4 contours will be the coins
		sortedKeys = sorted(areaMap, reverse=True)

		if sortedKeys and len(sortedKeys) >= 4:

			result, message = coinRecognition(sortedKeys[:4])
			
			if not result:

				print(message)
				
			else:

				#find contours, area threshold, count
				coinGroundTruth = sortedKeys[0]

			    # Ignore the first 4 sorted keys since we assume that the first 4 keys correspond to the coins    
				seedKeys = sortedKeys[4:]
			    
			    # Create a copy of the source since drawContours() modifies the source image
				seedOutput = src.copy()
			    
				seedCount = 0    
			    
				for area in seedKeys:
			        
					contour = areaMap[area]
			        	        
			        # Grab all contours whose area is over 5000 to ignore any noise that might have been identified as seeds
					if(area > 5000):
			            
						seedCount += 1

						cv.drawContours(seedOutput, [contour], -1, [255, 0, 0], 2)
			            
			            # identify the co-oridinates to draw a bounding rectangle around the identified contours
						x, y, w, h = cv.boundingRect(contour)
			            
			            # Draw a bounding rectangle around the identified contours
						cv.rectangle(seedOutput, (x, y), (x + w, y + h), (0, 255, 0), 2)          
			            
			            # Plot the area of the contour (seed) next to the rectangle
						cv.putText(seedOutput, "{}".format(str(measureArea(coinGroundTruth, 280.552077, area))[:5]), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 4, cv.LINE_AA)                 
			            
				writeImg("output{}.png".format(seedCount), seedOutput)