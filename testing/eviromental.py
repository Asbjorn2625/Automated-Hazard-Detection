import sys
import numpy as np
sys.path.append("/workspaces/Automated-Hazard-Detection")
from src.Segmentation import segmentation
from src.Preprocess.prep import PreProcess
from testing.classifier import Classifier
from src.Data_acquisition.Image_fetcher import ImageFetcher
from src.Segmentation.segmentation import Segmentation
import cv2
seg = Segmentation()
classifi = Classifier()

img = cv2.imread("images/rgb_image_0015.png")
depth = np.fromfile("images/depth_image_0015.raw", dtype=np.uint16)
    
trans = classifi.image_prep(img , depth)
mask = seg.locateHazard(trans)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
img_masked = cv2.bitwise_and(trans, trans, mask=mask)
rotated = Classifier.rotate_image(img_masked)
base_img = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
# Convert the min_val and max_val to numpy arrays
min_val = np.array([np.min(base_img) + 20])
max_val = np.array([np.max(base_img) - 40])
mask = cv2.inRange(base_img, min_val, max_val)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


cv2.imshow("mask", mask)
cv2.waitKey(0)
