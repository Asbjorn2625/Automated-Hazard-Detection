import cv2
import os

class ImageFetcher:
    def __init__(self, image_folder):
        self.image_folder = image_folder
    
    def load_images(self):
        # Get the path to the image directory using the current working directory
        image_directory = os.path.join(os.getcwd(),"images", self.image_folder)
        
        
        
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
                images.append([image_path, image])
        
        return images

    def __iter__(self):
        self.current_index = 0
        self.images = self.load_images()
        return self

    def __next__(self):
        if self.current_index >= len(self.images):
            raise StopIteration
        else:
            img = self.images[self.current_index]
            self.current_index += 1
            return img