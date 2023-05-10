import numpy as np
import cv2
import os
import sys

sys.path.append("/workspaces/Automated-Hazard-Detection")
from src.Preprocess.prep import PreProcess
from src.Segmentation.segmentation import Segmentation
import matplotlib.pyplot as plt

def display_images(images):
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(num_images*5, 5))
    for i, img in enumerate(images):
        # Change image to RGB and rotate it 90 degrees
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.show()

types_of_models = ['CombinedLoss', 'Dice_loss', 'IoU','CrossEntropy']

if __name__ == "__main__":
    # start the models
    for type in types_of_models:
        segmentor = Segmentation(model_type=type)
        pp = PreProcess()
        
        # load the images
        rgb_images = [f'./images/{img}' for img in os.listdir("./images") if img.startswith("rgb_image")]
        
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
            mask1 = segmentor.locateHazard(trans_img)
            mask2 = segmentor.locateUN(trans_img)
            mask3 = segmentor.locateCao(trans_img)
            mask5 = segmentor.locateTSU(trans_img)
            mask4 = segmentor.locatePS(trans_img)
            
            # Show the results
            display_images([mask1,mask2,mask3,mask4,mask5, trans_img])