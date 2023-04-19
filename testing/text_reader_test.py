import os
import numpy as np
from Libs.Preprocess.processing_functions import resize_image
import cv2
from craft_text_detector import Craft
import torch
from pytesseract import *
print(torch.cuda.is_available())
exit()

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
    #equalized_image = cv2.equalizeHist(gray)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Remove noise using morphological operations
    kernel = np.ones((2, 2), np.uint8)
    denoised = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    # Dilate text regions
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    # Show the preprocess
    resize_image(thresh, "tresh", 1)
    resize_image(denoised, "noise", 1)
    resize_image(dilated, "dilated", 1)

    # init tesseract
    #results = pytesseract.image_to_data(image, lang="eng", output_type=Output.DICT)
    text = pytesseract.image_to_string(dilated)
    return text.strip()


# Create list of image filenames
rgb_images = [f'./text_test/{img}' for img in os.listdir("./text_test") if img.startswith("rgb_image")]


# Loop through images
for image in rgb_images:
    img = cv2.imread(image)
    # Rotate image
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    orig = img.copy()

    # Load the pre-trained CRAFT model
    craft = Craft("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise use CPU
    # Load the OCR model
    pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"

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