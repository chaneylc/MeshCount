import cv2 

import sys

import numpy as np

import random as rng



def trackbar_callback(val):

    global global_dt
    global_dt = val

def analysis(path, dt):

    img = cv2.imread(path)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    cv2.imwrite('threshed.jpg', thresh)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    cv2.imwrite('noiseremoval.jpg', opening)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    cv2.imwrite('surebg.jpg', sure_bg)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    
    cv2.imwrite('dt.jpg', dist_transform)

    ret, sure_fg = cv2.threshold(dist_transform,dt*dist_transform.max(),255,0)

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

    img[markers == -1] = [255,0,0]

    img[markers > 0] = [255,255,255]

    cv2.imwrite(outputPath, img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('gray.jpg', gray)


    markers = markers.astype(np.uint8)

    ret, markers2 = cv2.threshold(markers, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(markers2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(hierarchy)

    print(len(contours))

    count = 0

    for index, pair in enumerate(zip(contours, hierarchy[0]),1):
                
        contour = pair[0]
        node = pair[1]

        r = rng.randint(0,128)
        g = rng.randint(0,128)
        b = rng.randint(0,128)

        color = (r, g, b)
        if node[2] == -1:

            hull = cv2.convexHull(contour, returnPoints=False)
            points = cv2.convexHull(contour, returnPoints=True)
            defects = cv2.convexityDefects(contour, hull)
            
            try:
                if len(defects) > 25:

                    print(points)

                    count += 1

                    cv2.drawContours(img, points, -1, color, 5)

                    for d in range(defects.shape[0]):
                        s,e,f,d = defects[d,0]
                        start = tuple(contour[s][0])
                        end = tuple(contour[e][0])
                        far = tuple(contour[f][0])
                        cv2.line(img,start,end,(0,0,0),2)
                        cv2.circle(img,far,2,(0,0,0),-1)


                    # identify the co-oridinates to draw a bounding rectangle around the identified contours
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Draw a bounding rectangle around the identified contours
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)          
                    
                    # Plot the area of the contour (seed) next to the rectangle
                    cv2.putText(img, "{}".format(count), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 4, cv2.LINE_AA) 
            except:
                pass    

    print(count)

    cv2.drawContours(img, contours, -1, (0,0,0), -1)

    return img

if __name__ == "__main__":
    
    global_dt = 1

    path = sys.argv[1]

    outputPath = sys.argv[2]

    cv2.namedWindow("Analysis", cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("Analysis", 1024,768)
    #cv.setMouseCallback("Variance", variance_mouse_callback)

    cv2.createTrackbar('DT', 'Analysis', 1, 100, trackbar_callback)

    while True:

        dst = analysis(path, global_dt/100.0)

        cv2.imshow("Analysis", dst)

        key = cv2.waitKey(3000) & 0xFF

        if key == ord("q"): 
            break

    cv2.destroyAllWindows()

    cv2.imwrite('contours.jpg', dst)