#from Hazard_labels.hazard import Hazard_labels
from Preprocess.prep import PreProcess
from images.image_loader import ImageFetcher
import cv2
import numpy as np

image_fetcher = ImageFetcher("Mark_files")

# Create a PreProcess object
pre_processor = PreProcess(image_fetcher)

# Process the images
for sr_image in pre_processor.process_images():
    # Do something with the super resolution image
    print(sr_image.shape)
    cv2.imwrite("hej.png",sr_image)
    cv2.imshow("orig", sr_image)
    cv2.waitKey(0)
              