import os
import sys
import cv2

sys.path.append(os.getcwd().replace("\\", "/") + "/src" )

from Segmentation.segmentation import Segmentation
from Preprocess.prep import PreProcess
from Data_acquisition.Image_fetcher import ImageFetcher

image_path = os.path.join(os.getcwd().replace("\\","/") + "/testing/labels_on_the_edge")

segment = Segmentation()

get_img = ImageFetcher(image_path)

imma_pre = PreProcess()

imglib = get_img.get_rgb_images()

i = 1

for file in imglib:
    
    
    preds =  segment.locateHazard(file)
    
    Roi = imma_pre.segmentation_to_ROI(preds)
    
    for bounds in Roi:
            cropped = file[bounds[1]:bounds[3], bounds[0]:bounds[2]]
    
   
    
    cv2.imshow("image", cropped)
 
    #cv2.imwrite("/workspaces/Automated-Hazard-Detection/testing/Out_cropped_hazard_rgb/image_"+ str(i) + ".png", cropped)
    
    
    i = i + 1
    
    cv2.waitKey(0)