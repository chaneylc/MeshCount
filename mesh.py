import cv2 as cv
import numpy as np
from collections import defaultdict
import itertools
import math


OUTPUT_DIR = r"C:\Users\marven\Documents\Spring-2020\CIS690"

#input is image using first-version soybeans mesh
#threshold function to find 3d printed mesh from an RGB image 1980x2592
#return is a blue background with red-masked mesh
def threshold(src):

    dst = src.copy()
    lab = cv.cvtColor(dst, cv.COLOR_RGB2LAB)

    #LAB threshold ranges that was found manually to work with
    #the lightbox
    L = [(0,90),(0,255),(0,255)]

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
    
    #gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    
    #eroded_rgb = cv.erode(gray, (3,3))
    
    #_, bin_img = cv.threshold(eroded_rgb, 100, 255, cv.THRESH_BINARY)

    #Debug statements to visualize the image outpu
    cv.imwrite("{}\Mesh.jpg".format(OUTPUT_DIR), rgb)

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

#driver code
if __name__ == "__main__":

	#read the input file
	src = cv.imread("{}\IMG.jpg".format(OUTPUT_DIR))
    
	src_copy = src.copy()
    

	#create a mask image 'dst' of the mesh
	dst = threshold(src)    
    
	#subtract the mesh from the original using opencv saturation subtract
	sub = cv.subtract(dst, src)

	cv.imwrite("{}\Subtracted.jpg".format(OUTPUT_DIR), sub)
    
	rep_sub = replaceChannelValue(sub, 2, 0, 255, 255)    

	#convert the image to grayscale
	gray = cv.cvtColor(rep_sub, cv.COLOR_RGB2GRAY)
    
	cv.imwrite("{}\gray_scale.jpg".format(OUTPUT_DIR), gray)
    

	#final threshold on gray image to segment seeds from background and mesh mask
	#ret, gray = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

	#cv.imwrite("{}\Threshed.jpg".format(OUTPUT_DIR), gray)
    
	#color = cv.cvtColor(gray,cv.COLOR_GRAY2RGB)    
        
	#find contours, area threshold, count
	edges = cv.Canny(gray, threshold1=240, threshold2=255)
    
	edges_smoothed = cv.GaussianBlur(edges, ksize=(5,5), sigmaX=10)
    
	kernel = np.ones((5,5), np.uint8) 
    
	#edges_smoothed = cv.erode(edges_smoothed, kernel) 
    
	edges_smoothed = cv.morphologyEx(edges_smoothed, cv.MORPH_CLOSE, kernel)
    
	orig_edges_smoothed = edges_smoothed.copy()  
    
    
    
	contours, hierarchy = cv.findContours(edges_smoothed,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
	contourCount = 0
    
	centerMap = defaultdict(int)
        
    
	for contour in contours:
        
		perimeter = cv.arcLength(contour, True)
                
		if(cv.contourArea(contour) > 1000):   
                    
			cv.drawContours(edges_smoothed, [contour], -1, (255, 255, 255), 3)
            
			(x,y),radius = cv.minEnclosingCircle(contour)

			center = (int(x), int(y))
                        
			centerMap[center] = contour
            
			contourCount += 1                                
    
	#cv.imwrite("{}\init_contours.jpg".format(OUTPUT_DIR), edges_smoothed)     
    

    
	res = cv.subtract(edges_smoothed, orig_edges_smoothed)
    
	res_copy = res.copy()    
    
	cv.imwrite("{}\subtract_res.jpg".format(OUTPUT_DIR), res)     
    
	for a, b in itertools.combinations(centerMap.keys(), 2):
		dist = math.sqrt( (a[0] - a[1])**2 + (b[0] - b[1])**2 )
		print(dist)
        if(dist < 1000 and a in centerMap.keys()):
			del centerMap[a]            
    
	#for cnt in centerMap.keys():    
		#cv.drawContours(orig_edges_smoothed, [centerMap[cnt]], -1, (255, 255, 255), 3) 
        
		#orig_edges_smoothed = cv.dilate(orig_edges_smoothed, (3, 3), iterations = 2)    
        
        
	final = cv.subtract(res, orig_edges_smoothed)        
                  
    #res = cv.dilate(res, (5, 5), iterations = 5)     

	cv.imwrite("{}\Threshed_Contours.jpg".format(OUTPUT_DIR), final)
	cv.imwrite("{}\orig_edges_smoothed.jpg".format(OUTPUT_DIR), res_copy)       
