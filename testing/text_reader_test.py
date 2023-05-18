import sys
sys.path.append('/workspaces/P6-Automated-Hazard-Detection')

from src.Text_reader.ReaderClass import ReadText
import os
import numpy as np
import cv2
from craft_text_detector import Craft
from pytesseract import *
import torch
from collections import deque
from matplotlib import pyplot as plt

reader = ReadText()

# Function to remove edge blobs
def grassfire(img, start, new_value):
    rows, cols = img.shape
    queue = deque()
    queue.append(start)
    old_value = img[start]

    while queue:
        x, y = queue.popleft()
        if x < 0 or x >= rows or y < 0 or y >= cols:
            continue
        if img[x, y] != old_value:
            continue

        img[x, y] = new_value

        queue.append((x-1, y))
        queue.append((x+1, y))
        queue.append((x, y-1))
        queue.append((x, y+1))

def remove_edge_blobs(img):
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


def readText(image, box, display=True, config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./- --psm 7 --oem 3', padding=10, RETURN_PIXEL_HEIGHT = False):
        # Extract the bounding area
        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]
        
        xmin = min(x1, x2, x3, x4)
        xmax = max(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        ymax = max(y1, y2, y3, y4)

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
            return "" if not RETURN_PIXEL_HEIGHT else "", 0
    
        
        # Convert to grayscale
        if len(image > 2):
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped_image

        # apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5,5))
        equalized = clahe.apply(gray)
        
        # Apply normalize the image
        equalized = cv2.normalize(equalized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Increase the size to better morph the image without damaging the detected text
        scale_factor = 3
        resized_image = cv2.resize(equalized, (equalized.shape[1] * scale_factor, equalized.shape[0] * scale_factor), interpolation=cv2.INTER_LANCZOS4)

        blurred = cv2.GaussianBlur(resized_image, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # White text on black background
        if np.sum(thresh > 100) > np.sum(thresh < 100):
            thresh = cv2.bitwise_not(thresh) 

            kernel = np.ones((3, 3), np.uint8)  # Increase the kernel size
            eroded_image = cv2.erode(thresh, kernel, iterations=2)
            dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
        else:
            kernel = np.ones((3, 3), np.uint8)  # Increase the kernel size
            eroded_image = cv2.erode(thresh, kernel, iterations=1)
            dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
        
        # Remove edge blobs
        segmented = remove_edge_blobs(dilated_image)
        
        if cv2.countNonZero(segmented) < 0:
             return "" if not RETURN_PIXEL_HEIGHT else "", 0
            
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
        
        # Compute the horizontal gradient using the Sobel operator
        grad = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)

        # Compute the absolute gradient
        abs_grad = np.abs(grad)

        # Normalize the gradient to the range 0-255
        norm_grad = cv2.normalize(abs_grad, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Threshold the gradient to create a binary mask of the stripes
        _, mask = cv2.threshold(norm_grad, 50, 255, cv2.THRESH_BINARY)  # adjust the threshold as needed
        # create a vertical line kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        # Dialate and erode the mask to remove noise
        mask = cv2.erode(mask, kernel, iterations=3)
        
        # Use Hough Line Transform to find lines in the gradient image
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 20, None, 30, 5)
        # Initialize an empty list to store the y-coordinates of the lines
        y_coords = [0]

        # create color display
        display = equalized.copy()
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        # Iterate over the detected lines
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                y_coords.append(l[0])
                cv2.line(display, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        y_coords.append(equalized.shape[1])
        y_coords = list(set(y_coords))
        # display equalized image
        cv2.imshow("Equalized", display)
        cv2.waitKey(0)
        # Sort the y-coordinates in ascending order
        y_coords.sort()

        # Use the y-coordinates to segment the image into individual sections
        sections = [equalized[:, int(y_coords[i]):int(y_coords[i+1])] for i in range(len(y_coords)-1)]

        new_image = np.zeros_like(equalized)
        # Process each section
        for i, section in enumerate(sections):
            section = equalized[:, int(y_coords[i]):int(y_coords[i+1])]
            _, section_thresh = cv2.threshold(section, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            new_image[:, int(y_coords[i]):int(y_coords[i+1])] = section_thresh
            # Save the processed section to a file
            cv2.imshow(f"thresh{i}", section_thresh)
            cv2.waitKey(0)
                
        # Display the process
        if display:
            cv2.imshow("segmented image", segmented)
            cv2.imshow("thresh", equalized)
            cv2.imshow("cropped", new_image)
            cv2.imshow("thres", thresh)
            cv2.waitKey(0)
            
        ratio = gray.shape[1]/gray.shape[0]
        
        new_width = int(ratio*125)    
        
        segmented = cv2.resize(segmented, (new_width, 100), interpolation=cv2.INTER_LANCZOS4)
        
        # Extract the text through the tesseract
        text = pytesseract.image_to_string(segmented, config=config)
        if len(text.strip()) < 3:
            segmented = cv2.rotate(segmented, cv2.ROTATE_90_CLOCKWISE)
           
            text = pytesseract.image_to_string(segmented, config=config)
    
        return text.strip() #if not RETURN_PIXEL_HEIGHT else text.strip(), points

current_folder = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(current_folder, "for_miki")
# Create list of image filenames
rgb_images = [os.path.join(image_folder, image) for image in os.listdir(image_folder) if image.startswith("rgb")]

# Get the mask and rgb image
for image in rgb_images:
    image_name = image.split("/")[-1]
    if len(image_name.split("_")) > 3:
        mask = cv2.imread(image, 0)
    else:
        rgb = cv2.imread(image)

# Apply the mask to the rgb image
image = cv2.bitwise_and(rgb, rgb, mask=mask)        
# Crop to fit the mask
x, y, w, h = cv2.boundingRect(mask)
image = image[y:y+h, x:x+w]
# Rotate the image
image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# Detect the text
boxes = reader.findText(image)


for box in boxes:
    print(box)
    # Draw the boxes on the image
    #cv2.rectangle(image, (box[0][0], box[0][1]), (box[2][0], box[2][1]), (0, 255, 0), 2)
    # Extract the text
    text = readText(image, box, display=True)
#display the image
#cv2.imshow("image", image)
#cv2.waitKey(0)