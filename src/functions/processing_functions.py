import numpy as np
import cv2


# Function to get the current real world pixel size
def get_pixelsize(depth):
    fov = np.deg2rad([69, 42])
    # 2 * depth * tan(FOV / 2) * (object width in pixels / image width in pixels)
    width = 2 * depth * np.tan(fov[0]/2)/1920
    height = 2 * depth * np.tan(fov[1]/2)/1080
    print(depth)
    return width, height


def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, int(width), int(height))
    cv2.imshow(image_name, image)