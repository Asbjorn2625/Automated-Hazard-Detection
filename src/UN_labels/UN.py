import sys
import os
import cv2
import json
import re
sys.path.append(os.getcwd().replace("\\", "/") + "/src")
from Text_reader.ReaderClass import ReadText
from Segmentation.segmentation import Segmentation
from Data_acquisition.Image_fetcher import ImageFetcher
from Preprocess.prep import PreProcess
import numpy as np

class UNLabels:
    def __init__(self):
      self.reader = ReadText()
      self.list = []

    def unLogo(self, image):
        cv2.imshow(image)
        cv2.waitKey(0)
    def packagingCodes(self,image):
        import re
        pattern = r"^\d+\w+\/\w+\d(\.\d)*\/(\w)?"
        text= []
        Package = []
        bounding = self.reader.findText(image)
        config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./- --psm 6 --oem 3'
        for boxes in bounding:
            text.append(self.reader.readText(image, boxes,config=config))
        for txt in text:
            match = re.match(pattern, txt)
            if match:
                found=match.group(0).split("/")
                PackageType = found[0]
                certification = found[1]
                solid = True if found[2] == "S" else False
                Package.append([PackageType, certification, solid])
        return Package

""""""
segsy = Segmentation()
fetcher = ImageFetcher(os.getcwd().replace("\\", "/") + "/Dataset")
img_lib = fetcher.get_rgb_depth_images()
PreProcesser = PreProcess()
UN = UNLabels()

for filename, (rgb_img, depth_img) in img_lib.items():
    segs_image = segsy.locateUN(rgb_img)
    segs_image = cv2.bitwise_and(rgb_img, segs_image)
    if cv2.mean(segs_image) != 0:
        tran_img = PreProcesser.transform(segs_image)
        for bounds in tran_img:
            cropped = segs_image[bounds[1]:bounds[3], bounds[0]:bounds[2]]
            enhance = PreProcesser.image_enhancer(cropped)
            logo= UN.unLogo(cropped)
            
        mask = np.where((depth_img > 1000) | (depth_img < 10), 0, 255).astype(np.uint8)
        masked = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
        UNinv= cv2.bitwise_not(segs_image)
        masked = cv2.bitwise_and(masked, UNinv)
        print("Text found" + str(UN.packagingCodes(masked)))