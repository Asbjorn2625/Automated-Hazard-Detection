import cv2
import numpy as np
import math
import sys
sys.path.append('/workspaces/Automated-Hazard-Detection')
sys.path.append('/workspaces/P6-Automated-Hazard-Detection')
from src.Text_reader.ReaderClass import ReadText

class Classifier:
    def __init__(self, ocr_model, preprocess_model):
        self.reader = ocr_model
        self.ocr_results = {"OCR_dic": {}}
        self.pp = preprocess_model
        
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
    def detect_package_orientation(self, image, Display=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        ret, thresh = cv2.threshold(image,20,255, cv2.THRESH_BINARY)
        cnt, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        largest_extent = float("-inf")
        for c in cnt:
            #measure the extent of the contour
            cnt_area = cv2.contourArea(c)
            # find the rotated rectangle that encloses the contour
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = [[int(point[0]), int(point[1])] for point in box]
        
            box_area = cv2.contourArea(np.array(box))
            extent = cnt_area/box_area
            
            if extent > largest_extent:
                largest_extent = extent
                best_box = box
                box = np.array(box)
                # get the longest side of the box
                long_side_index = np.argmax([np.linalg.norm(box[i] - box[(i+1)%4]) for i in range(4)])
                
                # Get the angel of the longest side
                dx = box[long_side_index][0] - box[(long_side_index+1)%4][0]
                dy = box[long_side_index][1] - box[(long_side_index+1)%4][1]
                angle = np.arctan2(dy, dx) * 180 / np.pi
                #angle = 90-angle
                if angle < 0:
                    angle += 180
                angle = (angle)%180
                if angle > 90:
                    angle = (angle)-180
                angle = abs(angle)
            
                
        cv2.drawContours(image, [np.array(best_box)], 0, (0, 0, 255), 2)
        # Resize image
        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
        # Display the image
        if Display:
            cv2.imshow("Image", image)
            cv2.waitKey(0)
        return round(angle,2)
        