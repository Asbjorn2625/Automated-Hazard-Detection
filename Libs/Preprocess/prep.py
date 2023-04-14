import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf
import tensorflow_hub as hub


class PreProcess:
    def __init__(self, fetched_images):
        self.image_fetcher = fetched_images
        self.counter = 0
        self.model = None
        self.esrgn_path = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

    def load_model(self):
        if self.counter == 0:
            self.model = hub.load(self.esrgn_path)
            self.counter += 1


    def preprocessing(self, img):
        imageSize = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4
        cropped_image = tf.image.crop_to_bounding_box(img, 0, 0, imageSize[0], imageSize[1])
        preprocessed_image = tf.cast(cropped_image, tf.float32)
        return tf.expand_dims(preprocessed_image, 0)


    def srmodel(self, img):
        self.load_model()
        preprocessed_image = self.preprocessing(img)
        new_image = self.model(preprocessed_image)
        return tf.squeeze(new_image) / 255.0

    def image_enhancer(self, image):
        # Convert numpy.ndarray to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Adjust brightness and gamma
        brightness_enhancer = ImageEnhance.Brightness(pil_image)
        brightened_img = brightness_enhancer.enhance(1.2)  # Adjust brightness by factor of 1.2
        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(brightened_img)
        contrast_enhanced_img = contrast_enhancer.enhance(1.2)  # 1.2 is the contrast factor
        sharpened_img = contrast_enhanced_img.filter(ImageFilter.SHARPEN)
        # Convert back to numpy.ndarray
        image = cv2.cvtColor(np.array(sharpened_img), cv2.COLOR_RGB2BGR)
        return image
    
    def transform(self, image):
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thr = cv2.threshold(grey, 0, 200, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Roi = []
        for cnt in contours:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(cnt)
            # Extract the four corner coordinates of the ROI
            xStart = x
            xEnd = x + w
            yStart = y
            yEnd = y + h
            Roi.append([xStart, yStart, xEnd, yEnd])
        return Roi
    
    def process_images(self):
        for name, Image in self.image_fetcher:
            outerbounds = self.transform(Image)
            for bounds in outerbounds:
                diamond = Image[bounds[1]:bounds[3], bounds[0]:bounds[2]]
                # Apply image enhancement
                enhanced_image = self.image_enhancer(diamond)
                # Apply super resolution
                sr_image = self.srmodel(enhanced_image)
                sr_image = tf.squeeze(sr_image).numpy() * 255.0  # convert to numpy array and scale pixel values back to [0, 255]
                sr_image = np.clip(sr_image, 0, 255).astype(np.uint8)  # clip pixel values to [0, 255] and convert to uint8 data type
                yield sr_image, name