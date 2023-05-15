import cv2
import numpy as np
import math
import sys

sys.path.append("/workspaces/P6-Automated-Hazard-Detection")
sys.path.append("/workspaces/Automated-Hazard-Detection")
from src.Classification.Classy import Classifier

class ProperShippingName(Classifier):
    def __init__(self, ocr_model, preprocessor):
        # Initialize the parent class
        super().__init__(ocr_model, preprocessor)
    
    def classify(self, image, depth_map, mask, homography):
        # Put the classification code here
        
        # get the bounding box of the text
        boxes = self.reader.findText(image)
        text_list = []
        # read the text
        for box in boxes:
            # Get the size of the text
            text_size = self._get_size(box, image, depth_map, homography)
            # read the text
            text = self.reader.readText(image, box)  # Maybe consider changing the whitelist here to only cover the letters in the proper shipping name
            text_list.append({"text":text, "size":text_size})
    
    
    def _get_size(self, text_box, image, depth_map, homography):
        # Get the corners of the text box
        bbox = np.array(text_box)

        # Find the top and bottom points
        top_point = min(bbox, key=lambda point: point[1])
        bottom_point = max(bbox, key=lambda point: point[1])

        corners = [top_point, bottom_point]
        
        # Change the corners into the original image
        corner_groups = [self.pp.transformed_to_original_pixel(image, pixel, homography) for pixel in corners]
        
        # Calculate the distances between consecutive corners
        corner1 = corner_groups[0]
        corner2 = corner_groups[1]  # Get the next corner in the group
        distance = self._distance_between_corners(corner1, corner2, depth_map)

        return distance
    