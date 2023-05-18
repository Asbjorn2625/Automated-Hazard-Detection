import cv2
import numpy as np
import math
import sys
import csv
sys.path.append('/workspaces/Automated-Hazard-Detection')
sys.path.append('/workspaces/P6-Automated-Hazard-Detection')
from src.Text_reader.ReaderClass import ReadText
from src.Preprocess.prep import PreProcess

class Classifier:
    def __init__(self, ocr_model, preprocess_model):
        self.reader = ocr_model
        self.ocr_results = {"OCR_dic": {}}
        self.pp = preprocess_model
        self.packaging = None
        self.Weight = None
        self.UN_nr = None
        self.place_holder = None
        self.Handeling = None
        self.Hazard = None
        self.TSU = []
        self.CAO = False
        self.env = False
        
    def calc_cnt(self,img):
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    def labels_on_edge(self,img, Distancethreshold=3):
        cnt = self.calc_cnt(img)
        # draw the contour on the original grayscale image
        cnt = max(cnt, key=cv2.contourArea)
        moments = cv2.moments(cnt)
        cx, cy = int(moments['m10']/moments['m00']),int(moments['m01']/moments['m00'])
        rect = cv2.minAreaRect(cnt)
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        if len(image.shape) == 3:
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

    def check_list(self, row_nr):
        with open('Dangerous_goods_list_for_testing.csv', 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            self.packaging = rows[row_nr][0]
            self.Weight = rows[row_nr][1]
            self.UN_nr = rows[row_nr][2]
            self.PS = rows[row_nr][3]
            self.adress = rows[row_nr][4]
            self.Hazard = rows[row_nr][5]
            self.Handeling = rows[row_nr][6]
        
        return None
    
    def Initial_classify(self, img, masks, row_nr):
        # Set ground truth
        self.check_list(row_nr)
        
        #Hazard = 0, CAO = 1, Lithium = 2, PS = 3, TSU = 4, UN_circle = 5
        # Determine presence of ROI in mask
        mask_truth = [np.max(mask) != 0 for mask in masks]
        
        imgs_masked = []

        # If Hazard label is detected on the picture, check orientation
        if mask_truth[0]:
            angle = self.Orientation(masks[0])
            bent = self.labels_on_edge(masks[0])
            if angle > 10 or bent:
                imgs_masked.append("empty")
            else:
                mask_gray = cv2.cvtColor(masks[0], cv2.COLOR_BGR2GRAY)
                masked_img = cv2.bitwise_and(img, img, mask=mask_gray)
                imgs_masked.append(masked_img)
        else:
            imgs_masked.append("empty")

        # Classifying CAO
        if self.Handeling == "CARGO CRAFT only":
            self.CAO = mask_truth[1]
        else:
            self.CAO = None

        if self.Handeling == "Enviormental damage":
            self.env = True

        # If Lithium battery handling label is expected, create cropped image if found
        if self.Handeling == "Lithium bat" and mask_truth[2]:
            mask_gray = cv2.cvtColor(masks[2], cv2.COLOR_BGR2GRAY)
            masked_img = cv2.bitwise_and(img, img, mask=mask_gray)
            imgs_masked.append(masked_img)
        else:
            imgs_masked.append("empty")

        #PS is cropped
        if mask_truth[3]:
            mask_gray = cv2.cvtColor(masks[3], cv2.COLOR_BGR2GRAY)
            masked_img = cv2.bitwise_and(img, img, mask=mask_gray)
            imgs_masked.append(masked_img)
        else:
            imgs_masked.append("empty")
        
        # Check whether TSU is found
        TSU_check = False
        if mask_truth[4]:
            #filter out noice from mask
            mask_gray = cv2.cvtColor(masks[4], cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cont in contours:
                if cv2.contourArea(cont) < 75:
                    cv2.drawContours(mask_gray, [cont], 0, (0), thickness=-1)
            contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #check if mask is still actual without noice
            if len(contours) == 3:
                #find orientation
                TSU_ori = self.detect_package_orientation(mask_gray)
                #check if orientation is within limit
                if TSU_ori < 25:
                    TSU_check = True
        
        if TSU_check:
            self.TSU.append(True)
        else:
            self.TSU.append(False)
                   

        #check for UN circle
        if mask_truth[5]:
            masks[5] = cv2.cvtColor(masks[5], cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(masks[5], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            height, width = masks[5].shape[:2]
            dilated_mask = np.zeros((height, width), dtype=np.uint8)
            dilation_size_y1 = 1000
            dilation_size_x1 = 20
            kernel1 = np.ones((dilation_size_y1, dilation_size_x1), np.uint8)
            dilation_size_y2 = 20
            dilation_size_x2 = 1000
            kernel2 = np.ones((dilation_size_y2, dilation_size_x2), np.uint8)
            # Iterate through each contour
            
            for contour in contours:
                # Draw the contour on the mask and fill it
                cv2.drawContours(dilated_mask, [contour], -1, (255), thickness=-1)

            # Dilate the mask to expand the contours
            cert_mask1 = cv2.dilate(dilated_mask, kernel1)
            cert_mask2 = cv2.dilate(dilated_mask, kernel2)
            cert_mask = cv2.bitwise_or(cert_mask1, cert_mask2)
            cert_mask = cv2.bitwise_and(img, img, mask=cert_mask)
            imgs_masked.append(cert_mask)
        else:
            imgs_masked.append("empty")

        return imgs_masked
    def finalClassification(self, imgs):
        #imgs 0 = Hazard, 1 = Lithium, 2 = PS, 3  = cert
        if imgs[3] != "empty":
            box_count = self.reader.findText(imgs[3])
            for box in box_count:
                UN_nr = self.reader.readText(imgs[3], box, True)
                print(UN_nr)
        if imgs[2] != "empty":
            pass
        if imgs[0] != "empty":
            pass
        if imgs[1] != "empty":
            pass
        
        #with open('fulloutput.csv', 'a', newline='') as f:
         #   full_data = ["Package type = " + data1] +["weight test = " + data2] + ["UN_nr = " + data3] + ["Hazard =" + data6] + ["Handeling =" + data7] + ["TSU = " + TSU_res]
         #   writer = csv.writer(f)
          #  writer.writerow(full_data)
        #print("full data saved", full_data)
        return 0
        


