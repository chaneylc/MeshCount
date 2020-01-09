import cv2 as cv
import numpy as np

OUTPUT_DIR = "/home/chaneylc/Desktop/Test"

#input is image using first-version soybeans mesh
#threshold function to find 3d printed mesh from an RGB image 1980x2592
#return is a blue background with red-masked mesh
def threshold(src):

    dst = src.copy()
    lab = cv.cvtColor(dst, cv.COLOR_RGB2LAB)

    #LAB threshold ranges that was found manually to work with
    #the lightbox
    L = [(108,138),(0,255),(0,255)]

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
    cv.imwrite("{}/Mesh.jpg".format(OUTPUT_DIR), rgb)

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
	src = cv.imread("{}/IMG.jpg".format(OUTPUT_DIR))

	#create a mask image 'dst' of the mesh
	dst = threshold(src)

	#subtract the mesh from the original using opencv saturation subtract
	sub = cv.subtract(dst, src)

	cv.imwrite("{}/Subtracted.jpg".format(OUTPUT_DIR), sub)

	#convert the image to grayscale
	gray = cv.cvtColor(sub, cv.COLOR_RGB2GRAY)

	#final threshold on gray image to segment seeds from background and mesh mask
	ret, gray = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

	cv.imwrite("{}/Threshed.jpg".format(OUTPUT_DIR), gray)

	#find contours, area threshold, count
