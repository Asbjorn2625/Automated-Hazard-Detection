import cv2
import numpy as np
import math
import sys
sys.path.append('/workspaces/Automated-Hazard-Detection')
sys.path.append('/workspaces/P6-Automated-Hazard-Detection')
from src.Text_reader.ReaderClass import ReadText

class classifier():
    def __init__(self):
        pass
        
    def calc_cnt(self,img):
        contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours    
    def labels_on_edge(self,img, Distancethreshold=3):
        cnt = self.calc_cnt(img)
        # draw the contour on the original grayscale image
        moments = cv2.moments(cnt)
        cx, cy = int(moments['m10']/moments['m00']),int(moments['m01']/moments['m00'])
        rect = cv2.minAreaRect(self.cnt_simple)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # calculate distance
        distance = math.sqrt((cx - int(rect[0][0]))**2 + (cy - int(rect[0][1]))**2)
        if distance >= Distancethreshold:
            label_bent = True
        else:
            label_bent = False
        return  label_bent
    
    
    def label_oriention(self,img):
        contours = self.calc_cnt(img)
        area = 0 
        for innercnt in contours:
            # Calculate the largest contour moment
            innerarea = cv2.contourArea(innercnt)
            if innerarea > area:
                area = innerarea
                cnt = innercnt

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
        dx = farthest_pt[0] - cx
        dy = farthest_pt[1] - cy
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return round(angle,2)
    
    def convex_hull_oriention(self, img):
        contours = self.calc_cnt(img)
        area = 0 
        for innercnt in contours:
            # Calculate the largest contour moment
            innerarea = cv2.contourArea(innercnt)
            if innerarea > area:
                area = innerarea
                cnt = innercnt

    
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
            dx = max_depth_end[0] - max_depth_start[0]
            dy = max_depth_end[1] - max_depth_start[1]
            angle = np.arctan2(dy, dx) * 180 / np.pi
            return round(angle,2)
        else:
            return None
    def Orientation(self, img):
        contours = self.calc_cnt(img)
        area = 0 
        if contours is not None:
            for innercnt in contours:
                # Calculate the largest contour moment
                innerarea = cv2.contourArea(innercnt)
                if innerarea > area:
                    area = innerarea
                    cnt = innercnt
                else:
                    continue
            # Approximate the contour with a polygon
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            corner_coordinates = [tuple(coord[0]) for coord in approx]
            corner_coordinates = np.array(corner_coordinates)

            # Find the index of the corner with the minimum x value
            #points = approx[:, 0, :]
            min_x_idx = np.argmin(corner_coordinates[:, 0])

            # Calculate the Euclidean distances between all corners and the corner with the minimum x value
            distances = np.linalg.norm(corner_coordinates - corner_coordinates[min_x_idx], axis=1)

            # Find the index of the corner with the maximum distance
            max_dist_idx = np.argmax(distances)

            # Calculate the angle between the corner with the minimum y value and the corner with the maximum distance
            delta_x = corner_coordinates[max_dist_idx, 0] - corner_coordinates[min_x_idx, 0]
            delta_y = corner_coordinates[max_dist_idx, 1] - corner_coordinates[min_x_idx, 1]
            angle_rad = np.arctan2(delta_y, delta_x)
            angle_deg = np.rad2deg(angle_rad)
            return round(angle_deg,2)
        else:
            return None
        