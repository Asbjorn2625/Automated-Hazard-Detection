import numpy as np
import cv2
import os


# Function to get the current real world pixel size
def get_pixelsize(depth):
    fov = np.deg2rad([64, 41])
    width = np.tan(fov[0]/2)*depth
    height = np.tan(fov[1]/2)*depth
    return width, height

def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, int(width), int(height))
    cv2.imshow(image_name, image)


# Create list of image filenames
rgb_images = [f'./early_test_images/{img}' for img in os.listdir("./early_test_images") if img.startswith("rgb_image")]

# Load the intrinsics
with np.load("calibration.npz") as a:
    mtx = a["mtx"]
    dist = a["dist"]

# Loop through the images
for image in rgb_images:
    img = cv2.imread(image)
    depth = np.fromfile(image.replace("rgb_image", "depth_image").replace("png", "raw"), dtype=np.uint16)
    # Reconstruct the depth map
    depth = depth.reshape(int(1080), int(1920))
    # Undistort an image
    img_undistorted = cv2.undistort(img, mtx, dist)
    depth_undistorted = cv2.undistort(depth, mtx, dist)

    # Get the width of the box
    width = 0
    for i in range(790, 1180):
        width += get_pixelsize(depth_undistorted[605, i])[0]
    print(width)


    # Create a boolean mask based on the pixel values in the 16-bit image
    mask = np.logical_and(depth_undistorted > 50, depth_undistorted < 1200)

    # Dilate the mask using OpenCV's morphology functions
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(np.bool)

    # Apply the mask
    img_masked = np.zeros_like(img_undistorted)
    img_masked[mask] = img_undistorted[mask]

    # Convert the depth image to a heatmap
    depth_heatmap = cv2.applyColorMap(cv2.convertScaleAbs(depth_undistorted, alpha=0.03), cv2.COLORMAP_JET)
    # Display the results
    resize_image(img_masked, "result", 0.4)
    resize_image(depth_heatmap, "depth", 0.4)
    cv2.waitKey(0)