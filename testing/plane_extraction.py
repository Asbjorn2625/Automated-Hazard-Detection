import sys
sys.path.append('/workspaces/P6-Automated-Hazard-Detection')

from src.Preprocess.prep import PreProcess
import os
import numpy as np
import cv2
import random


pp = PreProcess()

# Create list of image filenames
rgb_images = [f'./testing/orientation_tests/{img}' for img in os.listdir("./testing/orientation_tests") if img.startswith("rgb_image")]

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
    
    # Draw somewhere random on the image
    #x, y = random.randint(0, 400), random.randint(0, 400)
    #cv2.circle(trans_img, tuple((x,y)), 5, (0, 0, 255), -1)
    # Transform the circle into the original image and draw it there
    #point = pp.transformed_to_original_pixel((x,y), homography)
    #cv2.circle(img, point, 5, (0, 0, 255), -1)
    
    cv2.imshow("img", trans_img)
    #cv2.imshow("img1", img)
    cv2.waitKey(0)
    