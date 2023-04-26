import sys
sys.path.append('/workspaces/P6-Automated-Hazard-Detection')

import os
import numpy as np
import cv2
from craft_text_detector import Craft
from pytesseract import *
import torch
from collections import deque


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


def extract_text(image, box, display_name, padding=10):
    x1, y1 = box[0]
    x2, y2 = box[1]
    x3, y3 = box[2]
    x4, y4 = box[3]

    xmin = min(x1, x2, x3, x4)
    xmax = max(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    ymax = max(y1, y2, y3, y4)

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

    # Convert to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    scale_factor = 3
    resized_image = cv2.resize(gray, (gray.shape[1] * scale_factor, gray.shape[0] * scale_factor), interpolation=cv2.INTER_LANCZOS4)

    # Try Otsu's thresholding instead of adaptive thresholding
    _, thresh = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)  # Increase the kernel size
    eroded_image = cv2.erode(thresh, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
    
    # Remove edge blobs
    segmented = remove_edge_blobs(dilated_image)
    
    # Display the process
    cv2.imshow(display_name, segmented)
    
    # Set Tesseract configuration to focus on alphanumeric characters
    config = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./- --psm 6 --oem 3'
    text = pytesseract.image_to_string(segmented, config=config)
    return text.strip()

# Create list of image filenames
rgb_images = [f'./testing/text_test/{img}' for img in os.listdir("./testing/text_test") if img.startswith("rgb_image")]

# Load the pre-trained CRAFT model
craft = Craft("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise use CPU

# Loop through images
for image in rgb_images:
    img = cv2.imread(image)
    depth = np.fromfile(image.replace("rgb_image", "depth_image").replace("png", "raw"), dtype=np.uint16)
    # Reshape the depth image
    depth = depth.reshape(int(1080), int(1920))
    # Cut out the background
    mask = np.where((depth > 1000) | (depth < 10), 0, 255).astype(np.uint8)
    # Dialate the mask
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    # Apply the mask to the image
    img = cv2.bitwise_and(img, img, mask=mask)
    # Rotate image
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    orig = img.copy()
    
    # Detect text regions
    prediction_result = craft.detect_text(img)

    # Draw the bounding boxes on the image
    detected_texts = {"text": [], "box": []}
    temp_point = 0
    for box in prediction_result["boxes"]:
        text = extract_text(orig, box, f'dialated_{temp_point}')
        temp_point += 1
        if text:
            box = np.array(box, dtype=np.int32)  # Convert box coordinates to int32
            # Append the detected text and its bounding box to the list
            detected_texts["text"].append(text)
            detected_texts["box"].append(box)

    # Print the extracted texts
    print("Detected Texts:")
    for text, box in zip(detected_texts["text"], detected_texts["box"]):
        print(text)
        cv2.polylines(orig, [box], True, (0, 255, 0), 2)

    # Display the output image
    #resize_image(orig, "result", 0.4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Release the CRAFT model
craft.unload_craftnet_model()
craft.unload_refinenet_model()