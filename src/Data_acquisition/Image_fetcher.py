import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

class ImageFetcher:
    def __init__(self, path):
        self.path = path
        self.RGBimages = []
        self.depthImages = []
        self._fetch_RGB()
        self._fetch_depth()

    def _fetch_RGB(self):
        """
        Fetches all RGB images from the dataset folder
        :return: None
        """
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".png"):
                    image_path = os.path.join(root, file)
                    image = cv2.imread(image_path)
                    self.RGBimages.append((image_path, image))
    def _fetch_depth(self):
        """
        Fetches all depth images from the dataset folder
        :return: None
        """
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
        """Get depth images and RGB images that match
        :return: list of tuples (RGB image, depth image)"""
        
        images = []
        for i in tqdm(self.RGBimages):
            file_nr1=i[0].replace(".png","").replace("Dataset/","").split("_")[2]
            for j in self.depthImages:
                file_nr2=j[0].replace(".raw","").replace("Dataset/","").split("_")[2]
                if file_nr1 == file_nr2:
                    images.append((cv2.bitwise_and(i[1], i[1], mask=j[2]),cv2.bitwise_and(j[1], j[1], mask=j[2])))
        rgb_images = [img[0] for img in images]
        depth_images = [img[1] for img in images]
        return rgb_images, depth_images
    def get_rgb_images(self):
        """Get RGB images
        :param: None
        :return: list of tuples (filename, RGB image)"""
        rgb_images = []
        for img_path, img in self.RGBimages:
            filename = os.path.basename(img_path)
            rgb_images.append({"filename": filename, "image": img})
        return rgb_images
    def get_depth_images(self):
        """Get depth images
        :return: list of tuples (filename, depth image)"""
        return [img[1] for img in tqdm(self.depthImages)]
    def get_rgb_depth_images(self):
        """Get RGB images and depth images that match
        :return: list of tuples (RGB image, depth image)"""
        lib = {}
        for imgpath, img, mask in self.depthImages:
            filename = os.path.basename(imgpath).replace(".raw","").replace("depth_","")
            rgb = cv2.imread(imgpath.replace("depth","rgb").replace("raw","png"))
            lib[filename] = rgb, img
        return lib
    
    
    def alterate_set(self, path2excl):
        """[summary]
        Returns a dataframe with the images and the corresponding labels.
        """
        excl = pd.read_csv(path2excl)

        # Add a new column "filenames" to the existing DataFrame
        excl['filenames'] = None

        # Fetching the images
        imglib = self.get_rgb_depth_images()

        # Group the filenames into sets of four
        filenames_list = list(imglib.keys())
        grouped_filenames = [filenames_list[i:i + 4] for i in range(0, len(filenames_list), 4)]

        for index, row in excl.iterrows():
            # Fetch the image filenames related to the current row based on the chronological order
            try:
                filenames = grouped_filenames[index]
            except IndexError:
                print(f"No more sets of 4 images available for row {index}")
                continue

            # Update the "filenames" column in the DataFrame with the list of filenames
            excl.at[index, 'filenames'] = filenames
        return excl
    
image_processor = ImageFetcher(os.getcwd() + "/images")
excl = image_processor.alterate_set(os.getcwd() + "/Dangerous_goods_list_for_testing.csv")
print(excl)
            
            
            



