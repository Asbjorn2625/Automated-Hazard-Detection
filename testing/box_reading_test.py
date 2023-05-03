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
rgb_images = [f'./testing/Reading_test/{img}' for img in os.listdir("./testing/Reading_test") if img.startswith("rgb_image")]


# Loop through the images
for image in rgb_images:
    img = cv2.imread(image)
    depth = np.fromfile(image.replace("rgb_image", "depth_image").replace("png", "raw"), dtype=np.uint16)
    # Reconstruct the depth map
    depth = depth.reshape(int(1080), int(1920))
    
    #display_depth_image(depth)
    
    # undistort the image
    img = pp.undistort_images(img)
    
    depth = pp.undistort_images(depth)
    
    depth_blurred = cv2.medianBlur(depth, 5)
    
    
    
    trans_img, homography = pp.retrieve_transformed_plane(img, depth_blurred)

    trans_img = cv2.resize(trans_img, (960, 540))
    
    img = cv2.resize(img, (960, 540))  
    
    
      
    cv2.imshow("img", trans_img)
    cv2.imshow("img1", img)
    cv2.waitKey(0)

"""
    box_text = read.findText(trans_img)
    
    for box in box_text:
    
        text = read.readText(trans_img, box)
    

"""           
        
