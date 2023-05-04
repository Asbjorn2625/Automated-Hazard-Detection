import sys
sys.path.append('/workspaces/Automated-Hazard-Detection')




from src.Text_reader.ReaderClass import ReadText
from src.Preprocess.prep import PreProcess
import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def display_depth_image(depth_image, title='Depth Image'):
    plt.imshow(depth_image, cmap=plt.cm.viridis)
    plt.title(title)
    plt.axis('off')
    plt.show()

pp = PreProcess()

read = ReadText()


# Create list of image filenames
rgb_images = [f'./testing/New_reading_test/{img}' for img in os.listdir("./testing/New_reading_test") if img.startswith("rgb_image")]


# Loop through the images
for image in rgb_images:
    img = cv2.imread(image)
    

    
    
    depth = np.fromfile(image.replace("rgb_image", "depth_image").replace("png", "raw"), dtype=np.uint16)
    # Reconstruct the depth map
    depth = depth.reshape(int(1080), int(1920))
     
    # undistort the image
    img = pp.undistort_images(img)
    
    depth = pp.undistort_images(depth)
    
    depth_blurred = cv2.medianBlur(depth, 5)
    
    
    
    trans_img, homography = pp.retrieve_transformed_plane(img, depth_blurred)    



    # Increase brightness by 50
    alpha = 1
    beta = 0
    brightened_img = cv2.convertScaleAbs(trans_img, alpha=alpha, beta=-20)
    
    
    box_text = read.findText(brightened_img)
    
    #resized_depth_img = cv2.resize(brightened_img, (960, 540))
    
    resized_img = cv2.resize(img, (960, 540))  
            
    #cv2.imshow("img", resized_depth_img)
    #cv2.imshow("img1", resized_img)
    for box in box_text:
    
        text, segmented = read.readText(brightened_img, box, False, True)
        cv2.imshow("segs",segmented)
        print("text was found: " , text)
        
        


    
    cv2.waitKey(0)
"""
    # display result
    show_img = cv2.resize(trans_img, (960, 540))
    show_mask = cv2.resize(mask, (960, 540))  
    show_result = cv2.resize(result, (960, 540))  
    show_Clahe =   cv2.resize(img_clahe, (960, 540))  
    cv2.imshow('Original Image', show_img)
    cv2.imshow('Mask', show_mask)
    cv2.imshow('eq_histo', show_result)
    cv2.imshow('Clahe', brightened_img)
    cv2.waitKey(0)
    """