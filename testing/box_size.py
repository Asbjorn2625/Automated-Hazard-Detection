import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN

# Function to get the current real world pixel size
def get_pixelsize(depth):
    fov = np.deg2rad([69, 42])
    # 2 * depth * tan(FOV / 2) * (object width in pixels / image width in pixels)
    width = 2 * depth * np.tan(fov[0]/2)/1920
    height = 2 * depth * np.tan(fov[1]/2)/1080
    return width, height

def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, int(width), int(height))
    cv2.imshow(image_name, image)

def display_depth(depth, image_name, procent):
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
    resize_image(depth_colormap, image_name, procent)

def extract_planes_from_gradients(depth_image, min_distance=50, max_distance=1000, eps=0.1, min_samples=10):
    # Compute the gradients in the x and y directions
    grad_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=5)

    # Normalize the gradients to the range [0, 1]
    grad_x = cv2.normalize(grad_x, None, 0, 1, cv2.NORM_MINMAX)
    grad_y = cv2.normalize(grad_y, None, 0, 1, cv2.NORM_MINMAX)

    # Combine the gradients and depth image into a single array
    gradients = np.dstack((grad_x, grad_y, depth_image / max_distance))

    # Reshape the gradients array for clustering
    gradients = gradients.reshape((-1, 3))

    # Apply DBSCAN clustering to group pixels with similar gradients
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(gradients)

    # Get the unique cluster labels and remove the noise cluster (-1)
    unique_labels = np.unique(clustering.labels_)
    unique_labels = unique_labels[unique_labels != -1]

    # Extract the planes from the clusters
    planes = []
    for label in unique_labels:
        # Get the indices of the pixels in the current cluster
        indices = np.argwhere(clustering.labels_ == label)

        # Convert the indices back to the original image coordinates
        coordinates = np.column_stack((indices % depth_image.shape[1], indices // depth_image.shape[1]))

        # Create a mask for the current cluster
        mask = np.zeros_like(depth_image, dtype=np.uint8)
        mask[tuple(coordinates.T)] = 255

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the minimum bounding rectangle for each contour
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            planes.append(box)

    return planes


def display_results(rgb_image, planes):
    # Create a copy of the image to draw on
    image_with_planes = rgb_image.copy()

    # Draw the planes on the image
    for plane in planes:
        cv2.drawContours(image_with_planes, [plane], 0, (0, 255, 0), 2)

    # Display the image with the extracted planes
    resize_image(image_with_planes, "image_with_planes", 0.5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Create list of image filenames
rgb_images = [f'./testing/early_test_images/{img}' for img in os.listdir("./testing/early_test_images") if img.startswith("rgb_image")]

# Load the intrinsics
with np.load("src/Data acquisition/calibration.npz") as a:
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

    planes = extract_planes_from_gradients(depth_undistorted)
    
    display_results(img_undistorted, planes)