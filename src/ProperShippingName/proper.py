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
            
    def _distance_between_corners(self, corner1, corner2, depth_map):
        x1, y1 = corner1
        x2, y2 = corner2

        # Calculate the number of steps to iterate through the pixels
        num_steps = max(abs(x1 - x2), abs(y1 - y2))

        # Calculate the step size for x and y directions
        step_x = (x2 - x1) / num_steps
        step_y = (y2 - y1) / num_steps

        # Initialize the total distance
        total_distance = 0

        # Iterate through the pixels between the corners
        for step in range(1, num_steps + 1):
            # Calculate the current x and y coordinates
            x = int(round(x1 + step_x * step))
            y = int(round(y1 + step_y * step))

            # Get the depth value for the current pixel
            depth = depth_map[y, x]

            # Calculate the pixel size for the current pixel
            pixel_size = self.pp.get_pixelsize(depth)

            # Calculate the real-life distance between the current pixel and the previous pixel
            real_distance_x = abs(step_x) * pixel_size[0]
            real_distance_y = abs(step_y) * pixel_size[1]

            # Calculate the Euclidean distance between the current pixel and the previous pixel
            distance = np.sqrt(real_distance_x ** 2 + real_distance_y ** 2)

            # Add the distance to the total distance
            total_distance += distance

        return total_distance        
    
    
    def _get_size(self, text_box, image, depth_map, homography):

        # Find the top and bottom points
        top_point = text_box[0] 
        bottom_point = text_box[1]

        corners = [top_point, bottom_point]
        
        # Change the corners into the original image
        corner_groups = [self.pp.transformed_to_original_pixel(image, pixel, homography) for pixel in corners]
        
        # Calculate the distances between consecutive corners
        corner1 = corner_groups[0]
        corner2 = corner_groups[1]  # Get the next corner in the group
        distance = self._distance_between_corners(corner1, corner2, depth_map)

        return distance
    