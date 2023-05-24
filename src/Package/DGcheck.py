import csv
from src.Classification.Classy import Classifier
from src.Handling_labels.Handling import HandlingLabels
from src.Hazard_labels.hazard import Hazard_labels
from src.Segmentation.segmentation import Segmentation
from src.Preprocess.prep import PreProcess
import re
import numpy as np
import cv2
from src.ProperShippingName.proper import ProperShippingName 

class Package(Classifier):
    def __init__(self, ocr_model, preprocessor):
        super().__init__(ocr_model, preprocessor)
        self.seg = Segmentation()
        self.Haz_lab = Hazard_labels(ocr_model, preprocessor)
        self.hand_lab = HandlingLabels(ocr_model, preprocessor)
        self.proper = ProperShippingName(ocr_model, preprocessor)
        self.PreProd = preprocessor
        self.packaging = None
        self.Weight = None
        self.UN_nr = None
        self.PS = None
        self.adress = None
        self.Hazard = None
        self.Handeling = None
        self.masks = []
        self.DG_check = []
        self.depth = []
        self.img = []
        self.homography = None
        self.trans_img = None
        self.cert_XYZ_true = None
        self.pass1, self.pass2, self.pass3, self.pass4, self.pass5, self.pass6, self.pass7 = "fail", "fail", "fail", "fail", "fail", "fail", "fail"
        self.TSU_list = []
        self.cert_text = []
        
    def main(self, img, depth, row_nr):
        self.img = img
        self.depth = depth
        self.trans_img, self.homography = self.PreProd.retrieve_transformed_plane(self.img, self.depth)
        self.masks = self.seg.generateMasks(self.trans_img)
        self.gatherDGD(row_nr)
        self.errorCheck(self.masks)
        
        pass
        
    def gatherDGD(self, row_nr):
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
            #making sure that the ground truth matches the data format
            if self.packaging == "Fibreboard Box":
                self.packaging = "4G"
        return None
    def log_full(self):
        if self.TSU_list == [True, False, True, False] or self.TSU_list == [False, True, False, True]:
            self.pass6 = "pass"
        with open('fulloutput1.csv', 'a', newline='') as f:
            full_data = ["Package type = " + self.pass1] +["weight test = " + self.pass2] + ["UN_nr = " + self.pass3] + ["Hazard = " + self.pass4] + ["Handeling = " + self.pass5] + ["TSU = " + self.pass6]
            writer = csv.writer(f)
            writer.writerow(full_data)
        self.TSU_list = []
        self.pass1, self.pass2, self.pass3, self.pass4, self.pass5, self.pass6, self.pass7 = "fail", "fail", "fail", "fail", "fail", "fail", "fail"
    def log_img(self,input, csv_path):
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(input)
            
    def errorCheck(self, masks):
       
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cont in contours:
                if cv2.contourArea(cont) < 1000:
                    cv2.drawContours(mask, [cont], 0, (0), thickness=-1)
        
        self.mask_truth = [np.max(mask) != 0 for mask in masks]
        
        self.log_img(self.mask_truth, 'mask_truth.csv')
        #checking packaging for errors
        
        if "X" not in self.Weight and "Y" not in self.Weight and "Z" not in self.Weight:
            self.pass2 = "N/A"
            self.pass1 = "N/A"
            self.log_img(["None needed"], 'cert_text.csv')
        elif self.mask_truth[5]:
            values= []
            #checking packaging for errors
            cert = self.Isolate_cert(self.trans_img, masks[5])
            box_count = self.reader.findText(cert)
            for box in box_count:
                self.cert_text.append(self.reader.readText(cert, box, display = False))
            self.log_img(self.cert_text, 'cert_text.csv')
            for string in self.cert_text:
                Pac_values = re.findall(self.packaging, string)
                if len(Pac_values) > 0:
                    self.pass1 = "pass"
                    values = re.findall(r'Z\d+(?:\.\d+)?', string)
                if len(values) > 0:
                    break
                values = re.findall(r'Y\d+(?:\.\d+)?', string)
                if len(values) > 0:
                    break
                values = re.findall(r'X\d+(?:\.\d+)?', string)
                if len(values) > 0:
                    break
            if len(values) > 0:
                numbers1 = re.findall(r'\d+(?:\.\d+)?', self.Weight)
                numbers2 = re.findall(r'\d+(?:\.\d+)?', values[0])
                if len(numbers1) < 1:
                    numbers1 = 0
                else:
                    numbers1  = numbers1[0]
                if float(numbers2[0]) > float(numbers1):
                    if "Z" in values[0]:
                        self.pass2 = "pass"
                    elif "Y" in values[0]:
                        if "Y" in self.Weight or "X" in self.Weight:
                            self.pass2 = "pass"
                    elif "X" in values[0]:
                        if "X" in self.Weight:
                            self.pass2 = "pass"
        
        else: self.log_img(["None found"], 'cert_text.csv')
        self.cert_text = []
        
        #Checking PS
        text_list = []
        if self.mask_truth[3]:
            text_list.append(self.proper.classify(self.trans_img, self.depth, masks[3], self.homography))
        if self.mask_truth[2]:
            text_list.append(self.proper.classify(self.trans_img, self.depth, masks[2], self.homography))
        self.log_img(text_list, 'PS.csv')   
        for x in range(len(text_list)):
            for text in text_list[x]:
                if text[1] > 6:
                    if self.UN_nr in text[0]:
                        self.pass3 = "pass"
        
        
        
        #check for errors in Handeling labels
        if self.Handeling != "NONE":
            if self.mask_truth[1]:
                height, width = self.hand_lab.classify(self.trans_img, self.depth, masks[1], self.homography)
                self.log_img([height, width], 'CAO.csv')
                if self.Handeling == "CARGO CRAFT only":
                    if height >= 115 and width >= 105:
                        self.pass5 = "pass"
            else: self.log_img(["None found"], 'CAO.csv')
            if self.Handeling == "Lithium bat":
                if self.mask_truth[3]:
                    self.pass5 = "pass"
            if self.Handeling == "Enviormental damage":
                self.pass5 = "Env"    
        else:
            self.log_img(["None found"], 'CAO.csv')
            self.pass5 = "N/A"
        
        # Check whether TSU is found
        if self.mask_truth[4]:
            #filter out noice from mask
            contours, _ = cv2.findContours(masks[4], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cont in contours:
                if cv2.contourArea(cont) < 75:
                    cv2.drawContours(masks[4], [cont], 0, (0), thickness=-1)
            contours, _ = cv2.findContours(masks[4], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #check if mask is still actual without noice
            if len(contours) == 3:
                #find orientation
                TSU_ori = self.detect_package_orientation(masks[4])
                self.log_img([TSU_ori], 'TSU.csv')
                #check if orientation is within limit
                if TSU_ori < 25:
                    self.TSU_list.append(True)
                else:
                    self.TSU_list.append(False)
            else:
                self.TSU_list.append(False)
                self.log_img(["None  of correct dimensions found"], 'TSU.csv')
        else:
            self.TSU_list.append(False)
            self.log_img(["None found"], 'TSU.csv')
                    
        
        
        #check for errors in Hazard labels
        if self.mask_truth[0]:
            ori, bent = self.Haz_lab.Haz_Ori_bent(masks[0])
            haz_data = ["ori = " + str(ori), "bent = " + str(bent)]        
            if ori <= 10 and not bent:
                Haz_classes, Haz_size = self.Haz_lab.classify(self.trans_img, self.depth, masks[0], self.homography)
                haz_data = haz_data + ["Haz_classes = "] + [Haz_classes] + ["Haz_size = "] + [Haz_size]
                if Haz_size > 95:
                    self.log_img(haz_data, "Haz.csv")
                    haz_string = str(self.Hazard)
                    haz_string = haz_string.replace(" ", "")
                    haz_amount = "1"
                    if "or" in haz_string:
                        self.Hazard = self.Hazard.split("or")
                    else:
                        haz_string = re.sub("[a-zA-Z]", "", haz_string)
                        self.Hazard = haz_string.split(",")
                        haz_amount = str(len(self.Hazard))
                    haz_count = 0
                    for Haz_class in Haz_classes:
                        for haz in self.Hazard:
                            if Haz_class == self.find_class(self.Haz_lab.class_List(), haz):
                                haz_count += 1
                    self.pass4 = str(haz_count) + "/" + haz_amount
                else: self.log_img(haz_data, "Haz.csv")
            else: self.log_img(haz_data, "Haz.csv")
        else: self.log_img(["None found"], "Haz.csv")

    def Isolate_cert(self, img, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = mask.shape[:2]
        dilated_mask = np.zeros((height, width), dtype=np.uint8)
        dilation_size_y1 = 1000
        dilation_size_x1 = 50
        kernel1 = np.ones((dilation_size_y1, dilation_size_x1), np.uint8)
        dilation_size_y2 = 50
        dilation_size_x2 = 1000
        kernel2 = np.ones((dilation_size_y2, dilation_size_x2), np.uint8)
        # Iterate through each contour
        for contour in contours:
            # Draw the contour on the mask and fill it
            cv2.drawContours(dilated_mask, [contour], -1, (255), thickness=-1)

        # Dilate the mask to expand the contours
        img_masked1 = cv2.dilate(dilated_mask, kernel1)
        img_masked2 = cv2.dilate(dilated_mask, kernel2)
        img_masked = cv2.bitwise_or(img_masked1, img_masked2)
        img_masked = cv2.resize(img_masked, (img.shape[1], img.shape[0]))
        img_masked = cv2.bitwise_and(img, img, mask=img_masked)
        img_res = cv2.resize(img_masked, (562,562))
        return img_masked              
    
        
    def find_class(self,classes, number):
        for class_name, identifiers in classes.items():
            if number in identifiers:
                return class_name
            