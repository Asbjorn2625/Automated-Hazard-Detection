import numpy as np
import cv2
import os


# Function to get the current real world pixel size
def get_pixelsize(depth):
    fov = np.deg2rad([64, 41])
    width = np.tan(fov(0)/2)*depth
    height = np.tan(fov(1)/2)*depth
    return width, height

def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, int(width), int(height))
    cv2.imshow(image_name, image)


def depth_to_display(depth_frame):
    depth_segmentation_value = 255  # maximum value for each channel

    # scale depth frame to fit within 3 channels of bit depth 8
    depth_frame = depth_frame / 8192 * 3 * depth_segmentation_value

    # segment depth image into 3 color channels for better visualisation
    depth_frame_b = np.where(depth_frame > 2 * depth_segmentation_value - 1,
                             cv2.subtract(depth_frame, 2 * depth_segmentation_value), np.zeros_like(depth_frame))
    depth_frame = np.where(depth_frame > 2 * depth_segmentation_value - 1, np.zeros_like(depth_frame), depth_frame)
    depth_frame_g = np.where(depth_frame > depth_segmentation_value - 1,
                             cv2.subtract(depth_frame, depth_segmentation_value), np.zeros_like(depth_frame))
    depth_frame_r = np.where(depth_frame > depth_segmentation_value - 1, np.zeros_like(depth_frame), depth_frame)

    # Aligned and depth images have different shapes, so we check for both
    shape = depth_frame_b.shape
    if len(shape) <= 1:
        depth_frame_color = cv2.merge([depth_frame_b[:, :, 0], depth_frame_g[:, :, 0], depth_frame_r[:, :, 0]])
    else:
        depth_frame_color = cv2.merge([depth_frame_b[:, :], depth_frame_g[:, :], depth_frame_r[:, :]])

    depth_frame_color = depth_frame_color.astype(np.uint8)
    return depth_frame_color


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
    depth = depth.reshape(int(1080/3), int(1920/3))
    # Undistort an image
    img_undistorted = cv2.undistort(img, mtx, dist)
    depth_map_scaled = cv2.resize(depth, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    depth_undistorted = cv2.undistort(depth_map_scaled, mtx, dist)

    # Find the missing data
    missing_data = depth_undistorted == 0

    # Inpaint the missing data
    inpaint_method = cv2.INPAINT_NS  # adjust this to set the inpainting method
    inpaint_radius = 5  # adjust this to set the inpainting radius
    depth_undistorted = cv2.inpaint(depth_undistorted, missing_data.astype(np.uint8), inpaint_radius, inpaint_method)

    # Get a mask of the background from depth image
    background_depth = np.max(depth_undistorted[400:, 300:])
    print(background_depth)
    mask = depth_undistorted > background_depth - 200

    # Get gray image
    gray = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)
    # Remove the mask from the color image
    gray[mask] = 0
    # Get the edges of the color image
    edges = cv2.Canny(gray, 50, 200)
    # Display the edges
    resize_image(gray, "edges", 0.4)

    # Get the depth in color for display
    d_img = depth_to_display(depth_undistorted)



    # Display the results
    resize_image(img_undistorted, "result", 0.4)
    resize_image(d_img, "depth", 0.4)
    cv2.waitKey(0)