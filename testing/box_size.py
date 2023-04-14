import numpy as np
import cv2
import os


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
    for i in range(795, 1185):
        width += get_pixelsize(depth_undistorted[915, i])[0]
        # Draw the pixel on the image
        cv2.circle(img_undistorted, (i, 915), 1, (0, 0, 255), 2)
    print(width)
    # Get the height of the box
    height = 0
    for i in range(590, 970):
        height += get_pixelsize(depth_undistorted[i, 1100])[1]
        # Draw the pixel on the image
        cv2.circle(img_undistorted, (1100, i), 1, (0, 255, 0), 2)
    print(height)

    # Convert the depth image to a heatmap
    depth_heatmap = cv2.applyColorMap(cv2.convertScaleAbs(depth_undistorted, alpha=0.03), cv2.COLORMAP_JET)
    # Display the results
    resize_image(img_undistorted, "result", 0.4)
    resize_image(depth_heatmap, "depth", 0.4)
    cv2.waitKey(0)