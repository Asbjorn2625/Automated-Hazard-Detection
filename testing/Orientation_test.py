import sys
import os
sys.path.append('/workspaces/Automated-Hazard-Detection')

from src.Preprocess.prep import PreProcess
from src.Data_acquisition.Image_fetcher import ImageFetcher
from src.Classification.Classy import classifier
from src.Segmentation.segmentation import Segmentation
import pandas as pd
import cv2
import numpy as np

classy = classifier()
Imagefetch = ImageFetcher(os.getcwd() + "/Dataset")
imglib = Imagefetch._Orientation_test()
pp= PreProcess()
seg = Segmentation(model_type="CombinedLoss")
def format_degree(value):
        return f"{value}$^\circ$"


columns = { "Real angle":[], "Predicted angle": [],"difference": []}



for filename, (rgb_img, depth_img) in imglib.items():
    # undistort the image
    img = pp.undistort_images(rgb_img)
    
    depth = pp.undistort_images(depth_img)
    
    depth_blurred = cv2.medianBlur(depth, 5)
    
    trans_img, homography = pp.retrieve_transformed_plane(img, depth_blurred)
    segmented_img=seg.locateHazard(trans_img)
    ROI = pp.segmentation_to_ROI(segmented_img)
    
    for bounds in ROI:
        image = segmented_img[bounds[1]:bounds[3], bounds[0]:bounds[2]]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        resized = cv2.resize(trans_img, (int(trans_img.shape[1]*0.5),int(trans_img.shape[0]*0.5)), interpolation = cv2.INTER_AREA)
        
        cv2.imshow("image", image)
        cv2.imshow("trans",resized)
        cv2.waitKey(0)
    if "+" in filename:
        real_angle = float(filename.split("+")[1])
        columns["Real_angle"].append(format_degree(round(real_angle,2)))
        prediction= abs(classy.Orientation(image))
        print(round(prediction,2))
        columns["Predicted_angle"].append(format_degree(round(prediction,2)))
        columns["diffrence"].append("\\textbf{"+ format_degree(round(abs(real_angle - prediction),2))+ "}")
        
    elif "-" in filename:
        real_angle = float(filename.split("-")[1])
        columns["Real_angle"].append(format_degree(round(real_angle,2)))
        prediction= abs(classy.Orientation(image))
        print(round(prediction,2))
        columns["Predicted_angle"].append(format_degree(round(prediction,2)))
        columns["diffrence"].append("\\textbf{"+ format_degree(round(abs(real_angle - prediction),2))+ "}")
    else:
        continue


df = pd.DataFrame(columns)
print(df.to_latex(index=False,escape=False))