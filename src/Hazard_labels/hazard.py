import cv2
import numpy as np
import math
import Levenshtein
import sys

sys.path.append("/workspaces/P6-Automated-Hazard-Detection")
sys.path.append("/workspaces/Automated-Hazard-Detection")
from src.Classification.Classy import Classifier


class Hazard_labels(Classifier):
    def __init__(self, ocr_model, preprocessor):
        # Initialize the parent class
        super().__init__(ocr_model, preprocessor)

        self.classes = {"Explosives": ["EXPLOSIVE","1", "1.1","1.2","1.3","1.4","1.5","1.6"],
        "Flammable Gas": ["FLAMMABLE","GAS","FLAMMABLE GAS","2","2.1"],
        "Non-flammable gas": ["NON-FLAMMABLE","GAS","2","2.1"],
        "Toxic Gas": ["TOXIC","GAS","TOXIC GAS","2","2.3"],
        "Flammable Liquid": ["Flammable Liquid","Flammable", "Liquid","3"],
        "Flammable Solid": ["FLAMMABLE", "SOLID", "FLAMMABLE SOLID","4","4.1"],
        "Spontaneosly Combustible": ["SPONTANEOUSLY COMBUSTIBLE","SPONTANEOUSLY", "COMBUSTIBLE","4","4.2"],
        "Dangerous When Wet": ["DANGEROUS","WHEN", "WET","DANGEROUS WHEN WET", "4","4.3"],
        "Oxidizing Agent": ["OXIDISING","AGENT","OXIDISING AGENT","5","5.1"],
        "Organic Peroxides": ["ORGANIC", "PEROXIDES","ORGANIC PEROXIDES","5","5.2"],
        "Toxic": ["TOXIC","6","6.1"],
        "Infectous Substance": ["INFECTIOUS","SUBSTANCE","INFECTIOUS SUBSTANCE", "6","6.2"],
        "Corrosive": ["CORROSIVE","8"],
        "Miscellanous": ["MISCELLANEOUS","9"],
        "Lithium Batteries": ["LITHIUM BATTERIES", "9"]}

    def classify(self, image, depth_map, mask, homography):
        # Put the classification code here
        
        # Get size of the label
        label_size = self._get_size(mask, image, depth_map, homography)
        pass
    
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
        corner_groups = [self.pp.transformed_to_original_pixel(image, pixel, homography) for pixel in corner_groups]
        
        # Calculate the real-life distance of the sides for each contour
        for corner_group in corner_groups:
            num_corners = len(corner_group)
            if num_corners != 4:
                raise ValueError("Expected 4 corners for a rectangle, got {}".format(num_corners))
            
            side_distances = []
            for i in range(num_corners):
                corner1 = corner_group[i]
                corner2 = corner_group[(i + 1) % num_corners]  # Get the next corner in the group
                distance = self._distance_between_corners(corner1, corner2, depth_map)
                side_distances.append(distance)

            # Calculate the average distance of the sides for the current contour
            average_distance = sum(side_distances) / num_corners

        return average_distance
    def written_material(self, rgb_img, mask_img):
        writtenText=[]
        masked=cv2.bitwise_and(rgb_img, rgb_img, mask=mask_img)
        ROI=self.pp.segmentation_to_ROI(mask_img)
        for bounds in ROI:
            cropped = masked[bounds[1]:bounds[3], bounds[0]:bounds[2]]
            self.reader.findText(cropped)
            bounding = self.reader.findText(cropped)
            config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\ ./- --psm 7 --oem 3'
            for boxes in bounding:
                predtext= self.reader.readText(cropped, boxes,config=config)
                writtenText.append(predtext)
        if len(writtenText) == 0:
            return "No text found"
        else:
            return self._find_closest_match(writtenText, self.classes)[1]
    def _find_closest_match(self, input_list, DGRlist):
        scores = []
        for key, value in DGRlist.items():
            similarity = 0
            for word in value:
                for input_word in input_list:
                    if ((word != "" and input_word != "") and ((not input_word.replace(".","").isdigit()) and (not word.replace(".","").isdigit()))) and len(input_word) > 2:
                        if Levenshtein.distance(word, input_word) <= 2:
                            similarity += 1
                            
                    elif (word != "" and input_word != "") and (input_word.replace(".","").isdigit() and word.replace(".","").isdigit()):
                        if word == input_word:
                            """ 
                            print("word: ", word, "input_word: ", input_word)
                            """
                            similarity += 1
            if similarity > 0:
                scores.append((key, similarity))
            

        if scores:
            max_score = max(scores, key=lambda x: x[1])
            max_score_value = max_score[1]
            max_score_indices = [i for i, score in enumerate(scores) if score[1] == max_score_value]
            keys_with_highest_scores = [scores[i][0] for i in max_score_indices]

            return scores, keys_with_highest_scores
        else:
            return scores, ""