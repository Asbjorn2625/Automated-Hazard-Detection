import os
import cv2
import numpy as np

class ImageFetcher:
    def __init__(self, path):
        self.path = path
        self.RGBimages = []
        self.depthImages = []
        self._fetch_RGB()
        self._fetch_depth()

    def _fetch_RGB(self):
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".png"):
                    image_path = os.path.join(root, file)
                    image = cv2.imread(image_path)
                    self.RGBimages.append((image_path, image))
    def _fetch_depth(self):
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".raw"):
                    raw_path = os.path.join(root, file)
                    # Read raw file as a binary buffer
                    with open(raw_path, 'rb') as f:
                        buffer = f.read()
                    # Decode buffer as a 16-bit little-endian numpy array
                    depth_image = np.frombuffer(buffer, dtype=np.uint16).reshape((1080, 1920))
                    # Cut out the background
                    mask = np.where((depth_image > 1000) | (depth_image < 10), 0, 255).astype(np.uint8)
                    # Dialate the mask
                    kernel = np.ones((9, 9), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=2)
                    self.depthImages.append((raw_path, depth_image, mask))
    def get_images(self):
        images = []
        for i in self.RGBimages:
            file_nr1=i[0].replace(".png","").replace("Dataset/","").split("_")[2]
            for j in self.depthImages:
                file_nr2=j[0].replace(".raw","").replace("Dataset/","").split("_")[2]
                if file_nr1 == file_nr2:
                    images.append((cv2.bitwise_and(i[1], i[1], mask=j[2]),cv2.bitwise_and(j[1], j[1], mask=j[2])))
        rgb_images = [img[0] for img in images]
        depth_images = [img[1] for img in images]
        return rgb_images, depth_images
    def get_rgb_images(self):
        return [img[1] for img in self.RGBimages]
    def get_depth_images(self):
        return [img[1] for img in self.depthImages]



