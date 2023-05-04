from pytesseract import *
from craft_text_detector import Craft
from collections import deque
import torch
import numpy as np
import cv2

class ReadText:
    # Start by loading the pre-trained CRAFT model
    def __init__(self):
        self.craft = Craft(output_dir=None, cuda="cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise use CPU
    
    
    def findText(self, img):
        """
        Function for finding the text in an image
        :param img: input image
        :return: list Bounding boxes [corner1, corner2, corner3, corner4]
        """
        # Detect text regions
        prediction_result = self.craft.detect_text(img.copy())
        return prediction_result["boxes"]

    
    def readText(self, image, box, display=False, Check_color=False, config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./- --psm 7 --oem 3', padding=10):
        """
        Function for extracting text given a text area and a image.
        
        :param image: image to extract the text from
        :param box: bounding box of the text area(Recommended to be found through scene detection such as CRAFT)
        :param display: True for cv2 display, false for not
        :param config: config string for the tesseract, defaults to Block letters with psm 6 and oem 3
        :param padding: How much padding should be added to the bounding box, default 10 pixels
        :return: returns the text found
        """
        # Extract the bounding area
        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]

         # Find the minimum area rectangle enclosing the text region
        rect = cv2.minAreaRect(np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.float32))

        # Get the angle of rotation
        angle = rect[2]
        #if angle < -45:
        #    angle += 90

        # Create the rotation matrix and apply it
        center = rect[0]
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_image = image #cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))
        
        #cv2.imshow("rotated_image?", rotated_image)
        
        xmin = min(x1, x2, x3, x4)
        xmax = max(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        ymax = max(y1, y2, y3, y4)

        # Padding is to make sure we get the entire text
        xmin = int(xmin) - padding
        xmax = int(xmax) + padding
        ymin = int(ymin) - padding
        ymax = int(ymax) + padding

        (h, w) = rotated_image.shape[:2]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        
        cropped_image = rotated_image[ymin:ymax, xmin:xmax]
        
        if ymax - ymin > xmax - xmin:
            cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
        
        
        if Check_color: 
            #checking the color of the background    
            pixel_value = rotated_image[int(ymax*0.9), int(xmax/2)]

            # Print the pixel value
            print('Pixel value at ({}, {}): {}'.format(int(ymax*0.9), int(xmax/2), pixel_value))
        
        #cv2.imshow("cropped image?", cropped_image)
        
        # Convert to grayscale
        if len(image > 2):
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # Increase the size to better morph the image without damaging the detected text
        scale_factor = 3
        resized_image = cv2.resize(gray, (gray.shape[1] * scale_factor, gray.shape[0] * scale_factor), interpolation=cv2.INTER_LANCZOS4)

        #cv2.imshow("resise", resized_image)


        
        # Get the dimensions of the image
        h, w = resized_image.shape[:2]

        # Define the percentage values for cropping
        cropping_precentage = 0.1

        # Calculate the number of pixels to crop from each side
        left_to_right = int(w * cropping_precentage)
        top_to_bottom = int(h * cropping_precentage)
        

        # Crop the image to get rid of background noise, we only want to look at the text color and background color 
        text_check = resized_image[top_to_bottom:h-top_to_bottom, left_to_right:w-left_to_right]
        
        cv2.imshow("crop", text_check)
        
        # Find the maximum and minimum pixel values
        max_pixel_value = np.max(text_check)
        min_pixel_value = np.min(text_check)

        # Calculate the mean pixel value
        mean_pixel_value = cv2.mean(text_check)[0]
        
        #print(mean_pixel_value)
        
        #print("This is the max and min pix val, max: ", max_pixel_value,"min: ", min_pixel_value)

        # Set the thresholding method based on the maximum, minimum, and mean pixel values
        # used to find out what the text color and thresholds acordingly 
        if (max_pixel_value - min_pixel_value > 100) and (min_pixel_value < 20):
            # If the range of pixel values is small, use the mean to determine the thresholding method
            if mean_pixel_value > 60:
                _, thresh = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                _, thresh = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
             _, thresh = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


        
        kernel = np.ones((3, 3), np.uint8)  # Increase the kernel size
        eroded_image = cv2.erode(thresh, kernel, iterations=1)
        dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
        
        # Remove edge blobs
        segmented = self._remove_edge_blobs(dilated_image)
        
        segmented = cv2.bitwise_not(segmented)
        
        # Display the process
        if display:
            cv2.imshow("segmented image", segmented)
            cv2.waitKey(0)
        
        # Extract the text through the tesseract
        text = pytesseract.image_to_string(segmented, config=config)
        return text.strip()
    
    
    def _remove_edge_blobs(self, img):
        """
        Function for removing the edge blob through grassfire function
        
        :param img: np.array - Binary img
        :return: returns the binary image without the edge BLOBS
        """
        # Simple grassfire function
        def grassfire(img, start, new_value):
            # Extract image shape
            rows, cols = img.shape
            # Setup a queue
            queue = deque()
            queue.append(start)
            # List of visited pixels 
            old_value = img[start]
            
            # Run through the queue
            while queue:
                x, y = queue.popleft()
                # Check if we leave the image
                if x < 0 or x >= rows or y < 0 or y >= cols:
                    continue
                if img[x, y] != old_value:
                    continue

                img[x, y] = new_value
                # Append new values
                queue.append((x-1, y))
                queue.append((x+1, y))
                queue.append((x, y-1))
                queue.append((x, y+1))
    
        # Extract image shape
        rows, cols = img.shape
        new_value = 0 # You can set this value to any number different from the blob's value
        # Process top and bottom edges
        for col in range(cols):
            if img[0, col] > 0:
                grassfire(img, (0, col), new_value)
            if img[-1, col] > 0:
                grassfire(img, (rows-1, col), new_value)

        # Process left and right edges
        for row in range(rows):
            if img[row, 0] > 0:
                grassfire(img, (row, 0), new_value)
            if img[row, -1] > 0:
                grassfire(img, (row, cols-1), new_value)
        return img
        