import cv2
import numpy as np
import sys
sys.path.append("/workspaces/Automated-Hazard-Detection")
from src.Text_reader.ReaderClass import ReadText
from src.Preprocess.prep import PreProcess
from src.Segmentation.segmentation import Segmentation



class Classifier:
    def __init__(self):
        self.seg = Segmentation()
        self.pp = PreProcess()
        self.reader = ReadText()
    def rotate_image(img_masked, camera_ori = -90):
        (h, w) = img_masked.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, camera_ori, 1.0)
        rotated = cv2.warpAffine(img_masked, M, (w, h))
        
        return rotated
    
    def classifi_hazard(self, img):
        
        mask = self.seg.locateHazard(img) 
         
        Hazard = np.max(mask)
        
        if Hazard == 0:
            Haz_res = False
        else:
            
            Haz_res = True
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        img_masked = cv2.bitwise_and(img, img, mask=mask)
        rotated = Classifier.rotate_image(img_masked)
        box_count = self.reader.findText(rotated)
        text = []
        for box in box_count:
            text = text + [self.reader.readText(rotated, box, False)]
        return Haz_res, text
    
    def classifi_Handeling(self, img):
        
        mask1 = self.seg.locateLithium(img)
        mask2 = self.seg.locateCao(img)
        mask3 = self.seg.locateTSU(img)
        lithium = np.max(mask1)
        CAO = np.max(mask2)
        TSU = np.max(mask3)
        if lithium == 0:
            lit_res = False
            UN = "NONE"
        else:
            lit_res = True
            mask = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
            img_masked = cv2.bitwise_and(img, img, mask=mask)
            rotated = Classifier.rotate_image(img_masked)
            box_count = self.reader.findText(rotated)
            UN = []
            for box in box_count:
                UN = UN + [self.reader.readText(rotated, box, False)]
        if CAO == 0:
            CAO_res = False
        else:
            CAO_res = True
        if TSU == 0:
            TSU_res = False
        else:
            TSU_res = True
        return lit_res, CAO_res, TSU_res, UN
    
    def image_prep(self, img, depth):
        depth = depth.reshape(int(1080), int(1920))
        img = self.pp.undistort_images(img)
        depth = self.pp.undistort_images(depth)
        trans_img, homography = self.pp.retrieve_transformed_plane(img, depth)
        return trans_img
        
    def classif_PS(self,img):
        mask = self.seg.locatePS(img)
        PS = np.max(mask)
        if PS == 0:
            Haz_res = False
        else:
            Haz_res = True
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        img_masked = cv2.bitwise_and(img, img, mask=mask)
        rotated = Classifier.rotate_image(img_masked)
        box_count = self.reader.findText(rotated)
        text = []
        for box in box_count:
            text = text + [self.reader.readText(rotated, box, False)]
            
        return Haz_res, text
    
    def classifi_UN(self, img):
        UN_mask = self.seg.locateUN(img)
        
        
        UN_mask = cv2.cvtColor(UN_mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(UN_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = UN_mask.shape[:2]
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
        img_masked1 = cv2.dilate(dilated_mask, kernel1)
        img_masked2 = cv2.dilate(dilated_mask, kernel2)
        img_masked = cv2.bitwise_or(img_masked1, img_masked2)
        img_masked = cv2.bitwise_and(img, img, mask=img_masked)
        rotated = Classifier.rotate_image(img_masked)
        box_count = self.reader.findText(rotated)
        text = []
        for box in box_count:
            text = text + [self.reader.readText(rotated, box, False)]
            
        return rotated, text
            
        
        
        
        

