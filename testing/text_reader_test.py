import os
import numpy as np
from Libs.Preprocess.processing_functions import resize_image
import cv2
from craft_text_detector import Craft
import torch
from pytesseract import *


def extract_text(image, box, padding=10):
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

    scale_factor = 2
    resized_image = cv2.resize(gray, (gray.shape[1] * scale_factor, gray.shape[0] * scale_factor), interpolation=cv2.INTER_LANCZOS4)

    blurred_image = cv2.GaussianBlur(resized_image, (3, 3), 0)

    # Try Otsu's thresholding instead of adaptive thresholding
    _, thresh = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)  # Increase the kernel size
    eroded_image = cv2.erode(thresh, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

    # Display the process
    resize_image(gray, "cropped_image", 0.4)
    resize_image(thresh, "gray", 0.4)
    resize_image(dilated_image, "equalized_image", 0.4)

    # Set Tesseract configuration to focus on alphanumeric characters
    config = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./- --psm 4 --oem 3'
    text = pytesseract.image_to_string(dilated_image, config=config)
    return text.strip()


# Create list of image filenames
rgb_images = [f'./text_test/{img}' for img in os.listdir("./text_test") if img.startswith("rgb_image")]

# Load the pre-trained CRAFT model
craft = Craft("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise use CPU
# Load the OCR model
pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"

# Loop through images
for image in rgb_images:
    img = cv2.imread(image)
    # Rotate image
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    orig = img.copy()

    # Detect text regions
    prediction_result = craft.detect_text(img)

    # Draw the bounding boxes on the image
    detected_texts = {"text": [], "box": []}
    for box in prediction_result["boxes"]:
        text = extract_text(orig, box)
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
    resize_image(orig, "result", 0.4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Release the CRAFT model
craft.unload_craftnet_model()
craft.unload_refinenet_model()