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
        
       # display_image2 = cv2.resize(color, (854,480), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("label on edge", color)
        
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
    #display_image = cv2.resize(color, (854,480), interpolation=cv2.INTER_CUBIC)
    #cv2.imshow("angle", color)
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
        #display_image3 = cv2.resize(color, (854,480), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("angle", color)
        return angle
    else:
        return None
    
    
def convex(cnt):
    hull = cv2.convexHull(cnt)

    # Approximate the contour with a polygon
    epsilon = 0.01 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # Find the minimum area rectangle that bounds the polygon
    rect = cv2.minAreaRect(approx)

    # Get the rotation angle of the rectangle
    angle = rect[2]


    # Draw the rotated rectangle onto the original image
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(color, [box], 0, (0, 0, 255), 2)

    # Print the rotation angle
    print('Rotation angle:', angle)
    cv2.imshow("angle", color)





def segments_intersect(p1, q1, p2, q2):
    # Returns True if line segment p1-q1 intersects with line segment p2-q2
    # Source: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    o1 = orientation1(p1, q1, p2)
    o2 = orientation1(p1, q1, q2)
    o3 = orientation1(p2, q2, p1)
    o4 = orientation1(p2, q2, q1)
    
    if o1 != o2 and o3 != o4:
        print("yes")
        return True
    
    if o1 == 0 and on_segment(p1, p2, q1):
        print("edge 1")
        return True
    
    if o2 == 0 and on_segment(p1, q2, q1):
        print("edge 2")
        return True
    
    if o3 == 0 and on_segment(p2, p1, q2):
        print("edge 3")
        return True
    
    if o4 == 0 and on_segment(p2, q1, q2):
        print("edge 4")
        return True
    
    return False

def orientation1(p, q, r):
    # Returns the orientation of the triplet (p, q, r)
    # 0 = colinear, 1 = clockwise, 2 = counterclockwise
    # Source: https://www.geeksforgeeks.org/orientation-3-ordered-points/
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    
    if val == 0:
        return 0
    elif val > 0:
        return 2
    else:
        return 1
    
    
    
def on_segment(p, q, r):
# Returns True if point q lies on line segment pr
# Source: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    else:
        return False    
    

    
def hough(cnt):
    # Apply the Canny edge detector to find edges in the image
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)

    cv2.imshow("edge", edges)

    # Apply the Hough transform to detect lines in the image
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

    print(np.max(img_gray.shape))

    max_line_length = np.max(img_gray.shape)

    # Check for four lines that form a square
    if lines is not None:
        for i in range(len(lines)):
            rho1, theta1 = lines[i][0]
            a1 = np.cos(theta1)
            b1 = np.sin(theta1)
            x10 = a1 * rho1
            y10 = b1 * rho1
            x11 = int(x10 + max_line_length * (-b1))
            y11 = int(y10 + max_line_length * (a1))
            x12 = int(x10 - max_line_length * (-b1))
            y12 = int(y10 - max_line_length * (a1))
            
            for j in range(i+1, len(lines)):
                rho2, theta2 = lines[j][0]
                a2 = np.cos(theta2)
                b2 = np.sin(theta2)
                x20 = a2 * rho2
                y20 = b2 * rho2
                x21 = int(x20 + max_line_length * (-b2))
                y21 = int(y20 + max_line_length * (a2))
                x22 = int(x20 - max_line_length * (-b2))
                y22 = int(y20 - max_line_length * (a2))
                
                # Check if the angle between the two lines is close to 90 degrees
                angle = np.abs(np.arctan2(b1, a1) - np.arctan2(b2, a2)) * 180 / np.pi
                
                if np.abs(angle - 90) < 10:
                    for k in range(j+1, len(lines)):
                        rho3, theta3 = lines[k][0]
                        a3 = np.cos(theta3)
                        b3 = np.sin(theta3)
                        x30 = a3 * rho3
                        y30 = b3 * rho3
                        x31 = int(x30 + max_line_length * (-b3))
                        y31 = int(y30 + max_line_length * (a3))
                        x32 = int(x30 - max_line_length * (-b3))
                        y32 = int(y30 - max_line_length * (a3))
                        
                        # Check if the angle between the third line and the first two lines is close to 90 degrees
                        angle = np.abs(np.arctan2(b1, a1) - np.arctan2(b3, a3)) * 180 / np.pi
                        
                        if np.abs(angle - 90) < 10:
                            # Check if the fourth line intersects with the other three lines to form a square
                            for l in range(k+1, len(lines)):
                                rho4, theta4 = lines[l][0]
                                a4 = np.cos(theta4)
                                b4 = np.sin(theta4)
                                x40 = a4 * rho4
                                y40 = b4 * rho4
                                x41 = int(x40 + max_line_length * (-b4))
                                y41 = int(y40 + max_line_length * (a4))
                                x42 = int(x40 - max_line_length * (-b4))
                                y42 = int(y40 - max_line_length * (a4))
                                
                                                            # Check if the angle between the fourth line and the first three lines is close to 90 degrees
                            angle = np.abs(np.arctan2(b1, a1) - np.arctan2(b4, a4)) * 180 / np.pi
                            
                            if np.abs(angle - 90) < 10:
                                
                                
                                
                                # Check if the fourth line intersects with the other three lines to form a square
                                pt1 = np.array([x11, y11])
                                pt2 = np.array([x12, y12])
                                pt3 = np.array([x21, y21])
                                pt4 = np.array([x22, y22])
                                pt5 = np.array([x31, y31])
                                pt6 = np.array([x32, y32])
                                pt7 = np.array([x41, y41])
                                pt8 = np.array([x42, y42])
                                #print(pt1)
                                
                                
                                # Draw the four lines that form a square
                                cv2.line(color, (x11, y11), (x12, y12), (0, 0, 255), 2)
                                cv2.line(color, (x21, y21), (x22, y22), (0, 0, 255), 2)
                                cv2.line(color, (x31, y31), (x32, y32), (0, 0, 255), 2)
                                cv2.line(color, (x41, y41), (x42, y42), (0, 0, 255), 2)
                                #cv2.circle(color, (x11,y11),500,(255,255,0),1000)
                                
                                intersects = (segments_intersect(pt1, pt3, pt2, pt4) or
                                              segments_intersect(pt1, pt5, pt2, pt6) or
                                              segments_intersect(pt1, pt7, pt2, pt8))
                                
                                if intersects:
                                    print("k")
                                    
                                    # Draw the four lines that form a square
                                    cv2.line(color, (x11, y11), (x12, y12), (0, 0, 255), 2)
                                    cv2.line(color, (x21, y21), (x22, y22), (0, 0, 255), 2)
                                    cv2.line(color, (x31, y31), (x32, y32), (0, 0, 255), 2)
                                    cv2.line(color, (x41, y41), (x42, y42), (0, 0, 255), 2)

    cv2.imshow("angle", color)

            
    

    

            
            
    


# Replace 'your_folder_path' with the actual path to your folder containing images
mask_folder_path = 'testing/labels_on_the_edge'



# List all files in the folder
file_list = os.listdir(mask_folder_path)
for file in file_list:
    # Check if the file is an image (e.g., has a .jpg or .png extension)
    if file.lower().endswith(('.png')):
        # Create the full file path by joining the folder path and file name
        file_path = os.path.join(mask_folder_path, file)

        img_gray = cv2.imread(file_path, 0)

        #img_gray, original_image = compare_to_original_image(img_gray)

        color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)


        #inverting the binary image so it works
        #img_thresh = cv2.bitwise_not(img_thresh)

        #getting the contours
        contours = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for cnt in contours:
            # get the area of the contour
            area = cv2.contourArea(cnt)

            # check if the area is greater than or equal to 1000
            if area >= 1000:

                labels_on_edge(cnt)

                #label_oriention(cnt)
                
                #convex(cnt)
                
                #hough(cnt)
        

                break

        cv2.waitKey(0)