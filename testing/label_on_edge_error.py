import cv2 
import numpy as np
from matplotlib import pyplot as plt
import os
import math

#funtion to add the mask to the originial image for correct dimentions
def compare_to_original_image(img_gray):
    
    # Replace 'your_folder_path' with the actual path to your folder containing images
    original_folder_path = 'testing/orientation_testing'

    # List all files in the folder
    original_file_list = os.listdir(original_folder_path)
   
    for file in original_file_list:
        if file.lower().endswith(('.png')):
            # Create the full file path by joining the folder path and file name
            original_img_file_path = os.path.join(original_folder_path, file)
    
            originial_image = cv2.imread(original_img_file_path, 0)
            img_gray = cv2.resize(img_gray,(originial_image.shape[1], originial_image.shape[0]), interpolation=cv2.INTER_CUBIC)
            return img_gray, originial_image    
        



def labels_on_edge(cnt):
        # draw the contour on the original grayscale image
        cv2.drawContours(color, [cnt], 0, (0, 255, 0), 3)
        moments = cv2.moments(cnt)
        cx, cy = int(moments['m10']/moments['m00']),int(moments['m01']/moments['m00'])
        print([cx,cy])
        
        cv2.circle(color, (cx, cy), 1, (0, 255, 0), 3 )
        
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(color,[box],0,(0,0,255),2)
        
        cv2.circle(color, (int(rect[0][0]),int(rect[0][1])), 1, (0,0,255), 3)
        
        # calculate distance
        distance = math.sqrt((cx - int(rect[0][0]))**2 + (cy - int(rect[0][1]))**2)
        
        print(distance) 
        
        if distance >= 3:
            label_bent = True
        else:
            label_bent = False
        print(label_bent)
        
        display_image2 = cv2.resize(color, (854,480), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("label on edge", display_image2)
        
        #cv2.imshow("label on edge", color )   
        

        return  label_bent

        
        
    
def label_oriention(cnt):
   # Compute the centroid of the contour
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    max_dist = 0
    farthest_pt = None
    for pt in cnt:
        dist = np.sqrt((pt[0][0]-cx)**2 + (pt[0][1]-cy)**2)
        if dist > max_dist:
            max_dist = dist
            farthest_pt = pt[0]
        
    # Draw a line between the centroid and the farthest point
    cv2.line(color, (cx, cy), (farthest_pt[0], farthest_pt[1]), (0, 0, 255), 2)
    dx = farthest_pt[0] - cx
    dy = farthest_pt[1] - cy
    angle = np.arctan2(dy, dx) * 180 / np.pi
    print(angle)
    display_image = cv2.resize(color, (854,480), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("angle", display_image)
    return angle


def convex_hull_oriention(cnt):
    # Compute the convex hull of the contour
    hull = cv2.convexHull(cnt, returnPoints=False)
    
    # Compute the convexity defects of the contour and hull
    defects = cv2.convexityDefects(cnt, hull)
    
    # Find the largest defect depth
    max_depth = 0
    max_depth_start = None
    max_depth_end = None
    for i in range(defects.shape[0]):
        start_idx, end_idx, farthest_idx, depth = defects[i][0]
        if depth > max_depth:
            max_depth = depth
            max_depth_start = tuple(cnt[start_idx][0])
            max_depth_end = tuple(cnt[end_idx][0])
    
    # Draw a line between the start and end points of the largest defect
    if max_depth_start is not None and max_depth_end is not None:
        cv2.line(color, max_depth_start, max_depth_end, (0, 0, 255), 2)
        dx = max_depth_end[0] - max_depth_start[0]
        dy = max_depth_end[1] - max_depth_start[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        print(angle)
        display_image3 = cv2.resize(color, (854,480), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("angle", display_image3)
        return angle
    else:
        return None


# Replace 'your_folder_path' with the actual path to your folder containing images
mask_folder_path = 'testing/yodo'



# List all files in the folder
file_list = os.listdir(mask_folder_path)
print(file_list)

for file in file_list:
    # Check if the file is an image (e.g., has a .jpg or .png extension)
    if file.lower().endswith(('.png')):
        # Create the full file path by joining the folder path and file name
        file_path = os.path.join(mask_folder_path, file)

        img_gray = cv2.imread(file_path, 0)

        img_gray, original_image = compare_to_original_image(img_gray)

        color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        # get the shape of the image

        #thresholding the white background away
        img_thresh = cv2.threshold(img_gray, 0, 5, cv2.THRESH_BINARY)[1]

        #inverting the binary image so it works
        #img_thresh = cv2.bitwise_not(img_thresh)

        #getting the contours
        contours = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for cnt in contours:
            # get the area of the contour
            area = cv2.contourArea(cnt)

            # check if the area is greater than or equal to 1000
            if area >= 1000:

                #labels_on_edge(cnt)

                #label_oriention(cnt)
                
                convex_hull_oriention(cnt)
        

                break

        cv2.waitKey(0)