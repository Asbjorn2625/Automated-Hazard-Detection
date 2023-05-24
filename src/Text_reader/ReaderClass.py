from pytesseract import *
from craft_text_detector import Craft
from collections import deque
import torch
import numpy as np
import cv2
import warnings
import matplotlib.pyplot as plt

# Ignore specific warning
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.models._utils')
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
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(30,30))
        clahe_img = clahe.apply(gray)
        
        
        # Apply contrast stretching
        # Compute min and max pixel values
        min_val, max_val = np.min(gray), np.max(gray)
        # Apply contrast stretching
        stretched = (gray - min_val) / (max_val - min_val) * 255
        stretched = np.uint8(stretched)


        # Detect text regions
        equalized_result = self.craft.detect_text(equalized.copy())
        img_result = self.craft.detect_text(img.copy())
        clahe_result = self.craft.detect_text(clahe_img.copy())
        stretched_result = self.craft.detect_text(stretched.copy())

        
        combined_boxes = []
        combined_boxes.extend(equalized_result["boxes"])
        combined_boxes.extend(img_result["boxes"])
        combined_boxes.extend(clahe_result["boxes"])
        combined_boxes.extend(stretched_result["boxes"])
        
        aabb_boxes = [quadrilateral_to_aabb(box) for box in combined_boxes]

        # Apply NMS to the AABBs
        nms_boxes = self._merge_boxes(aabb_boxes)

        # Convert the NMS results back to quadrilaterals
        final_result = [aabb_to_quadrilateral(box) for box in nms_boxes]
        
        return np.array(final_result)

    
    def readText(self, image, box, display=True, config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.\ /- --psm 7 --oem 3', padding=5, RETURN_PIXEL_HEIGHT = False):
        """
        Function for extracting text given a text area and a image.
        
        :param image: image to extract the text from
        :param box: bounding box of the text area(Recommended to be found through scene detection such as CRAFT)
        :param display: True for cv2 display, false for not
        :param config: config string for the tesseract, defaults to Block letters with psm 7 and oem 3
        :param padding: How much padding should be added to the bounding box, default 10 pixels
        :return: returns the text found
        """
        # Extract the bounding area
        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]
        
        xmin = min(x1, x2, x3, x4)
        xmax = max(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        ymax = max(y1, y2, y3, y4)

        if (xmax - xmin) > 70:
            padding = padding + 20
        
        # Padding is to make sure we get the entire text
        xmin = int(xmin) - padding
        xmax = int(xmax) + padding
        ymin = int(ymin) - padding
        ymax = int(ymax) + padding

        (h, w) = image.shape[:2]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        
        cropped_image = image[ymin:ymax, xmin:xmax]
        
        if ymax - ymin > xmax - xmin:
            cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
        
        # Rotate the image to follow the horizontal axis
        if cv2.countNonZero(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)) < 0:
            return ["", 0] if RETURN_PIXEL_HEIGHT else ""
    
        
        # Convert to grayscale
        if len(image > 2):
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped_image

        # Apply normalize the image
        equalized = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        
        # Increase the size to better morph the image without damaging the detected text
        scale_factor = 3
        resized_image = cv2.resize(equalized, (equalized.shape[1] * scale_factor, equalized.shape[0] * scale_factor), interpolation=cv2.INTER_LANCZOS4)


        _, thresh = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # White text on black background
        if np.sum(thresh > 100) > np.sum(thresh < 100):
            thresh = cv2.bitwise_not(thresh) 

            kernel = np.ones((3, 3), np.uint8)  # Increase the kernel size
            eroded_image = cv2.erode(thresh, kernel, iterations=2)
            dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
            for_show= dilated_image.copy
        else:
            kernel = np.ones((3, 3), np.uint8)  # Increase the kernel size
            eroded_image = cv2.erode(thresh, kernel, iterations=1)
            dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
            for_show = dilated_image.copy()
        
        # Remove edge blobs
        segmented = self._remove_edge_blobs(dilated_image)
        
        if cv2.countNonZero(segmented) < 0:
             return ["", 0] if RETURN_PIXEL_HEIGHT else ""
            
        #checking for the pixle height of the letters
        if RETURN_PIXEL_HEIGHT:
            
            scale_factor = 3
            resized_segment = cv2.resize(segmented, (int(segmented.shape[1]/scale_factor), int(segmented.shape[0]/scale_factor)), interpolation=cv2.INTER_LANCZOS4)
            
            # Find contours
            contours, _ = cv2.findContours(resized_segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Find the bounding rectangle for the contour
                x, y, _, h = cv2.boundingRect(contours[0])
                
                points = np.array([[xmin+x, ymin+y],[xmin+x, ymin+y+h]])
            else:
                points = np.array([0, 0])

        segmented = cv2.bitwise_not(segmented)    
            
                
        # Display the process
        if display:
            cv2.imshow("segmented image", segmented)
            cv2.imshow("thresh", gray)
            cv2.imshow("cropped", cropped_image)
            cv2.imshow("thres", thresh)
            cv2.imshow("dilate",for_show)
            cv2.waitKey(0)
            
        ratio = gray.shape[1]/gray.shape[0]
        
        new_width = int(ratio*125)    
        
        segmented = cv2.resize(segmented, (new_width, 100), interpolation=cv2.INTER_LANCZOS4)
        
        # Extract the text through the tesseract
        text = pytesseract.image_to_string(segmented, config=config)
        if len(text.strip()) < 3:
            segmented = cv2.rotate(segmented, cv2.ROTATE_90_CLOCKWISE)
           
            text = pytesseract.image_to_string(segmented, config=config)
    
        return [text.strip(), points] if RETURN_PIXEL_HEIGHT else text.strip()
    
    
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

    def _merge_boxes(self, boxes, overlapThresh=0.5):
        if len(boxes) == 0:
            return []
        
        # Convert bounding boxes to format [x1, y1, x2, y2]
        boxes = [[x, y, x+w, y+h] for (x, y, w, h) in boxes]

        # Convert bounding boxes to numpy array
        boxes = np.array(boxes).astype(float)

        # Initialize a list to hold the picked indexes	
        pick = []

        # Grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Compute the area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # Keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # Grab the last index in the indexes list and add the index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # Compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # Compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # Delete all indexes from the index list that have overlap greater than the provided overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        # Return only the bounding boxes that were picked
        boxes = boxes[pick].astype("int")

        # Convert bounding boxes back to format [x, y, w, h]
        boxes = [[x1, y1, x2-x1, y2-y1] for (x1, y1, x2, y2) in boxes]
        return boxes
    
    def _get_rotation_angle(self, image):
        # Perform edge detection
        edged = cv2.Canny(image, 50, 150, apertureSize = 3)

        # Perform a dilation and erosion to close gaps in between object edges
        dilated_edged = cv2.dilate(edged.copy(), None, iterations=5)
        eroded_edged = cv2.erode(dilated_edged.copy(), None, iterations=5)

        # Perform Hough Line Transform
        lines = cv2.HoughLinesP(eroded_edged, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        # Calculate the angles of the lines
        if lines is None:
            return 0
        # Calculate the angles of the lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        # Compute the median angle (more robust than mean)
        rotation_angle = np.median(angles)
        
        # Adjust the angle
        if rotation_angle < -45:
            rotation_angle = 90 + rotation_angle

        # If the text is vertical, adjust the angle
        if abs(rotation_angle) > 45:
            rotation_angle = 90 - rotation_angle

        return rotation_angle
    
   

def quadrilateral_to_aabb(box):
    """Convert a quadrilateral to an axis-aligned bounding box."""
    x_coords = [point[0] for point in box]
    y_coords = [point[1] for point in box]
    x1, x2 = min(x_coords), max(x_coords)
    y1, y2 = min(y_coords), max(y_coords)
    return [x1, y1, x2-x1, y2-y1]  # format: [x, y, w, h]

def aabb_to_quadrilateral(box):
    """Convert an axis-aligned bounding box to a quadrilateral."""
    x, y, w, h = box
    return [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)