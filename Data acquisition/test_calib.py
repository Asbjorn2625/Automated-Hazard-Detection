import numpy as np
import cv2
import os


def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, int(width), int(height))
    cv2.imshow(image_name, image)


# Create list of image filenames
image_names = [f'./data/{img}' for img in os.listdir("./data") if img.startswith("rgb_image")]

# Load the intrinsics
calibs = np.load("calibration.npz")

# Loop through the images
for image in image_names:
    img = cv2.imread(image)
    # Undistort an image
    img_undistorted = cv2.undistort(img, calibs["mtx"], calibs["dist"])

    resize_image(img_undistorted, "result", 0.4)
    cv2.waitKey(0)