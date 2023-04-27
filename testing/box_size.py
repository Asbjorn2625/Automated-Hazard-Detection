import numpy as np
import cv2
import os
import open3d as o3d

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


def depth_image_to_point_cloud(depth_image, camera_intrinsics):
    """Convert a depth image to a point cloud."""
    height, width = depth_image.shape

    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    xx = (xx - cx) * depth_image / fx
    yy = (yy - cy) * depth_image / fy

    points = np.column_stack((xx.ravel(), yy.ravel(), depth_image.ravel())) / 1000  # Convert to meters
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    return point_cloud


def resize_depth_image(depth_image, scale):
    height, width = depth_image.shape
    new_height, new_width = int(height * scale), int(width * scale)
    resized_image = cv2.resize(depth_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def display_results(rgb_image, planes):
    # Create a copy of the image to draw on
    image_with_planes = rgb_image.copy()

    # Draw the planes on the image
    for plane in planes:
        cv2.drawContours(image_with_planes, [plane], 0, (0, 255, 0), 2)

    # Display the image with the extracted planes
    cv2.imshow("image_with_planes", image_with_planes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def project_points_to_image(points, camera_intrinsics):
     # Filter out points with a z-coordinate close to zero.
    valid_points = points[np.abs(points[:, 2]) > 1e-6]
    
    # Project the valid points onto the image plane.
    uv = (valid_points @ camera_intrinsics.T)[:, :2] / valid_points[:, 2, np.newaxis]
    return uv.round().astype(int)

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
    
    # Remove background - Set pixels below the threshold as black
    grey_color = 50
    depth_undistorted[np.where((depth_undistorted > 1000) | (depth_undistorted < 10))] = grey_color
    
    # Resize the images to speed up processing
    #scale = 0.5
    #img_undistorted_resized = resize_image(img_undistorted, scale)
    #depth_undistorted_resized = resize_image(depth_undistorted, scale)

    # Update the camera intrinsics matrix based on the scaling factor
    #scaled_mtx = mtx.copy()
    #scaled_mtx[:2, :2] *= scale  # Scale the focal length and principal point
    
    segmented_rgb_image = img_undistorted.copy()
    # Convert the depth image to a point cloud.
    point_cloud = depth_image_to_point_cloud(depth_undistorted, mtx)

    max_planes = 5
    plane_colors = [
        (0, 100, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 0, 0),    # Red
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]

    for i in range(max_planes):
        # Apply RANSAC to segment planes from the point cloud.
        plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        
        if len(inliers) == 0:
            break  # No more planes found

        # Extract the plane points and remove them from the point cloud.
        plane_points = point_cloud.select_by_index(inliers)
        point_cloud = point_cloud.select_by_index(inliers, invert=True)
        
        # Remove points with low depth values (e.g., less than 0.1 meters)
        mask = np.asarray(point_cloud.points)[:, 2] > 0.1
        point_cloud = point_cloud.select_by_index(np.where(mask)[0])

        # Project the plane points back onto the image plane.
        plane_points_image = project_points_to_image(np.asarray(plane_points.points), mtx)

        # Filter out points with coordinates outside the image bounds.
        valid_indices = np.where((plane_points_image[:, 0] >= 0) & (plane_points_image[:, 0] < segmented_rgb_image.shape[0]) &
                                (plane_points_image[:, 1] >= 0) & (plane_points_image[:, 1] < segmented_rgb_image.shape[1]))[0]
        valid_points_image = plane_points_image[valid_indices]

        # Transpose the image before applying the mask
        segmented_rgb_image = np.transpose(segmented_rgb_image, (1, 0, 2))

        # Assign a unique color to the plane points on the RGB image
        segmented_rgb_image[valid_points_image[:, 0], valid_points_image[:, 1]] = plane_colors[i % len(plane_colors)]

        # Transpose the image back after applying the mask
        segmented_rgb_image = np.transpose(segmented_rgb_image, (1, 0, 2))

    # Show the original RGB image and the segmented RGB image.
    resize_image(segmented_rgb_image,"image", 0.5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()