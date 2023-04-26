
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import os

def load_images(image_folder):
    # Get the path to the image directory using the current working directory
    image_directory = os.path.join(os.getcwd(),"Libs\\Preprocess", image_folder)

    
    # Get a list of all the image files in the specified directory
    image_files = os.listdir(image_directory)
    
    # Load each image file using OpenCV
    images = []
    for image_file in image_files:
        # Construct the path to the image file
        image_path = os.path.join(image_directory, image_file).replace("\\","/")

        # Load the image using OpenCV
        image = cv2.imread(image_path)
        
        # Add the image to the list of images
        if image is not None:
            images.append(image)
    
    return images


def image_enhancer(image):
    # Convert numpy.ndarray to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Adjust brightness and gamma
    brightness_enhancer = ImageEnhance.Brightness(pil_image)
    brightened_img = brightness_enhancer.enhance(2)
    # Sharpen the image
    contrast_enhanced_img = ImageEnhance.Contrast(brightened_img)
    contrast_enhanced_img = contrast_enhanced_img.enhance(1.5)
    sharpened_img = contrast_enhanced_img.filter(ImageFilter.SHARPEN)
    # Convert back to numpy.ndarray
    image = cv2.cvtColor(np.array(sharpened_img), cv2.COLOR_RGB2BGR)
    # Convert to HSL
    hls_img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # Apply CLAHE to the lightness channel
    hls_img[:, :, 1] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(hls_img[:, :, 1])
    # Convert back to RGB
    rgb_img = cv2.cvtColor(hls_img, cv2.COLOR_HLS2BGR)
    return rgb_img

images = load_images("Early_images")
counter = 0

for image in images:
    image = image_enhancer(image)
    cv2.imwrite("Libs/UN_labels/Enhanced/Enhanced_image_%s.png" % counter,image)
    counter = counter + 1