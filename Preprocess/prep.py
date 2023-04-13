import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf
import tensorflow_hub as hub

class preProcess:
    def __init__(self,image_stream):
        self.image_folder = image_stream
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

