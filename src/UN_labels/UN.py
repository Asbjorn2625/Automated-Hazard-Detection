import sys
import os
import cv2
import json
sys.path.append(os.getcwd().replace("\\", "/") + "/src")
from Text_reader.ReaderClass import ReadText
from Segmentation.segmentation import Segmentation
from Data_acquisition.Image_fetcher import ImageFetcher
from Preprocess.prep import PreProcess
import numpy as np

class UNLabels:
    def __init__(self):
      self.reader = ReadText()

    def unLogo(self, image):
        bounding = self.reader.findText(image)
        text =  []
        for boxes in bounding:
            text.append(self.reader.readText(image, boxes))
        return text
    def packagingCodes(self,image):
        text= []
        bounding = self.reader.findText(image)
        config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./- --psm 5 --oem 3'
        for boxes in bounding:
            text.append(self.reader.readText(image, boxes,display=True,config=config))
        return text
    def boxDimensions(self):
        pass


segsy = Segmentation()
fetcher = ImageFetcher(os.getcwd().replace("\\", "/") + "/Dataset")
img_lib = fetcher.get_rgb_depth_images()
PreProcesser = PreProcess()
UN = UNLabels()

for filename, (rgb_img, depth_img) in img_lib.items():
    labels = []
    segs_image = segsy.locateUN(rgb_img)
    segs_image = cv2.bitwise_and(rgb_img, segs_image)
    if cv2.mean(segs_image) != 0:
        tran_img = PreProcesser.transform(segs_image)
        for bounds in tran_img:
            segs_image = segs_image[bounds[1]:bounds[3], bounds[0]:bounds[2]]
            enhance = PreProcesser.image_enhancer(segs_image)
            mask = np.where((depth_img > 1000) | (depth_img < 10), 0, 255).astype(np.uint8)
            masked = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
            print("Text found" + str(UN.packagingCodes(masked)))
            cv2.imshow("RGB", masked)
            cv2.waitKey(0)

