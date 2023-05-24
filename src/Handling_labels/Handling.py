import cv2
import numpy as np
import math
import sys


sys.path.append("/workspaces/P6-Automated-Hazard-Detection")
sys.path.append("/workspaces/Automated-Hazard-Detection")
from src.Classification.Classy import Classifier


class HandlingLabels(Classifier):
    def __init__(self, ocr_model, preprocessor):
        # Initialize the parent class
        super().__init__(ocr_model, preprocessor)

        
    def classify(self, image, depth_map, mask, homography):
        # Put the classification code here
        
        # Get size of the label
        label_width, label_height = self._get_size(mask, image, depth_map, homography)
        return label_height, label_width
    
    def _get_corners(self, mask, contour_area_threshold=100, epsilon_ratio=0.02):
        # Find the contours of the diamond shapes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        all_corner_groups = []

        # Loop through the contours
        for contour in contours:

            # Filter out small contours by area
            if cv2.contourArea(contour) > contour_area_threshold:
                # Approximate the contour shape
                epsilon = epsilon_ratio * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Check if the shape has 4 vertices (corners)
                if len(approx) == 4:
                    # Extract the corner coordinates
                    corner_coordinates = [tuple(coord[0]) for coord in approx]

                    # Append the corner coordinates to the list of corner groups
                    all_corner_groups.append(corner_coordinates)
                else:
                    print(f"Warning: Found {len(approx)} points instead of 4. Using the 4 most extreme points as corners.")
                    corners = np.zeros((4, 2), dtype=np.float32)
                    corners[0] = tuple(contour[contour[:, :, 0].argmin()][0])  # leftmost point
                    corners[1] = tuple(contour[contour[:, :, 1].argmin()][0])  # topmost point
                    corners[2] = tuple(contour[contour[:, :, 0].argmax()][0])  # rightmost point
                    corners[3] = tuple(contour[contour[:, :, 1].argmax()][0])  # bottommost point
                    all_corner_groups.append(corners)
            else:
                    print(f"Warning: Found {len(approx)} points instead of 4. Using the 4 most extreme points as corners.")
                    corners = np.zeros((4, 2), dtype=np.float32)
                    corners[0] = tuple(contour[contour[:, :, 0].argmin()][0])  # leftmost point
                    corners[1] = tuple(contour[contour[:, :, 1].argmin()][0])  # topmost point
                    corners[2] = tuple(contour[contour[:, :, 0].argmax()][0])  # rightmost point
                    corners[3] = tuple(contour[contour[:, :, 1].argmax()][0])  # bottommost point
                    all_corner_groups.append(corners)
                    
        return all_corner_groups
    def _get_diamond_corners(self, mask, contour_area_threshold=100, epsilon_ratio=0.02):
        # Find the contours of the diamond shapes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        all_corner_groups = []

        # Loop through the contours
        for contour in contours:
            # Filter out small contours by area
            if cv2.contourArea(contour) > contour_area_threshold:
                # Approximate the contour shape
                epsilon = epsilon_ratio * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Check if the shape has 4 vertices (corners)
                if len(approx) == 4:
                    # Extract the corner coordinates
                    corner_coordinates = [tuple(coord[0]) for coord in approx]

                    # Append the corner coordinates to the list of corner groups
                    all_corner_groups.append(corner_coordinates)
        return all_corner_groups
    
        
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
    
    def _get_size(self, mask, image, depth_map, homography):
        # Get the corner coordinates of the diamond shapes
        corner_groups = self._get_diamond_corners(mask)
        # Change the corners into the original image

        if len(corner_groups) != 0:
            corner_groups = [self.pp.transformed_to_original_pixel(image, np.array(pixel), homography) for pixel in corner_groups[0]]
            # Calculate the real-life distance of the sides for each contour
            for corner_group in corner_groups:
                num_corners = len(corner_group)

                side_distances = []
                for i in range(num_corners):
                    corner1 = corner_groups[i]
                    corner2 = corner_groups[(i + 1) % num_corners]  # Get the next corner in the group
                    distance = self._distance_between_corners(corner1, corner2, depth_map)
                    side_distances.append(distance)

                # Calculate the average distance of the sides for the current contour
                average_distance = sum(side_distances) / num_corners
            print("average distance", average_distance)
            return average_distance, average_distance
        else:
            return 0