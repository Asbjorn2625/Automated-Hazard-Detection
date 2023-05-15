import numpy as np
import cv2
import os
import sys

sys.path.append("/workspaces/P6-Automated-Hazard-Detection")
from src.Preprocess.prep import PreProcess
from src.Segmentation.segmentation import Segmentation
import matplotlib.pyplot as plt

def display_image(warped_image, title='Warped Image'):
    plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # start the models
    segmentor = Segmentation()
    pp = PreProcess()
    
    # load the images
    rgb_images = [f'./testing/first data set/{img}' for img in os.listdir("./testing/first data set") if img.startswith("rgb_image")]
    
    # Loop through the images
    for image in rgb_images:
        img = cv2.imread(image)
        depth = np.fromfile(image.replace("rgb_image", "depth_image").replace("png", "raw"), dtype=np.uint16)
        # Reconstruct the depth map
        depth = depth.reshape(int(1080), int(1920))
        
        # undistort the image
        img = pp.undistort_images(img)
        depth = pp.undistort_images(depth)
    
        
        trans_img, homography = pp.retrieve_transformed_plane(img, depth)
        
        # Segment the image
        mask = segmentor.locateUN(trans_img)
        
        # Show the results
        display_image(mask, title='Segmented Image')
        display_image(trans_img, title='Warped Image')