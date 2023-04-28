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

def project_points_onto_plane(points, plane_model):
    plane_normal = np.array(plane_model[:3])
    plane_point = plane_normal * plane_model[3]
    projection_matrix = np.eye(3) - np.outer(plane_normal, plane_normal)

    projected_points = np.dot(points, projection_matrix.T) + plane_point
    return projected_points


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

def transform_rgb_image_to_plane_view(rgb_image, plane_points, output_size=(400, 400)):
    # Define the destination points for the transformation (rectangle)
    destination_points = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype=np.float32)

    # Compute the homography matrix
    homography_matrix, _ = cv2.findHomography(plane_points, destination_points)

    # Warp the RGB image using the computed homography matrix
    transformed_image = cv2.warpPerspective(rgb_image, homography_matrix, output_size)

    return transformed_image


# Create list of image filenames
rgb_images = [f'./testing/first data set/{img}' for img in os.listdir("./testing/first data set") if img.startswith("rgb_image")]

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
    grey_color = 0
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
    
    min_angle_diff = float("inf")
    best_plane_inliers = None
    best_plane_normal_vector = None
    for i in range(max_planes):
        # Apply RANSAC to segment planes from the point cloud.
        plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

         # Calculate the normalized normal vector
        normal_vector = plane_model[:3] / np.linalg.norm(plane_model[:3])
        # Calculate the angle between the normal vector and the vertical direction (in degrees)
        angle_y = np.degrees(np.arccos(np.dot(normal_vector, np.array([1, 0, 0])))) # Our camera is turned, so the floor is on the horizontal
        angle_x = np.degrees(np.arccos(np.dot(normal_vector, np.array([0, 0, 1])))) # the closer to zero the closer to being orthogonal to the cameras optical axis
        
        # Remove the floor and noisy data
        if angle_y < 35 or len(inliers) < 80000:
            # Extract the plane points and remove them from the point cloud.
            plane_points = point_cloud.select_by_index(inliers)
            point_cloud = point_cloud.select_by_index(inliers, invert=True)
            continue
        if len(inliers) == 0:
            break  # No more planes found
        angle_diff = abs(angle_x)
        if angle_diff < min_angle_diff:
            min_angle_diff = angle_diff
            # Extract the plane points and remove them from the point cloud.
            best_plane_inliers = point_cloud.select_by_index(inliers)
            best_plane_normal_vector = point_cloud.select_by_index(inliers, invert=True)
    
   
    plane_points = best_plane_inliers
    point_cloud = best_plane_normal_vector
    
    # Remove points with low depth values (e.g., less than 0.1 meters)
    mask = np.asarray(point_cloud.points)[:, 2] > 0.1
    point_cloud = point_cloud.select_by_index(np.where(mask)[0])

    # Project the plane points back onto the image plane.
    plane_points_image = project_points_to_image(np.asarray(plane_points.points), mtx)

    # Filter out points with coordinates outside the image bounds.
    valid_indices = np.where((plane_points_image[:, 0] >= 0) & (plane_points_image[:, 0] < segmented_rgb_image.shape[1]) &
                        (plane_points_image[:, 1] >= 0) & (plane_points_image[:, 1] < segmented_rgb_image.shape[0]))[0]
    valid_points_image = plane_points_image[valid_indices]

    # Assign a unique color to the plane points on the RGB image
    segmented_rgb_image[valid_points_image[:, 1].round().astype(int), valid_points_image[:, 0].round().astype(int)] = plane_colors[i % len(plane_colors)]

    # Create binary mask
    plane_mask = np.zeros_like(segmented_rgb_image)
    plane_mask[valid_points_image[:, 1].round().astype(int), valid_points_image[:, 0].round().astype(int)] = 255
    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    eroded = cv2.erode(cv2.cvtColor(plane_mask, cv2.COLOR_RGB2GRAY), kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=5)
    # Get the convex hull
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull_list = []
    for i in range(len(contours)):
        if len(contours[i]) > 200:
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)
    # Approximate the convex hull with a polygonal curve
    epsilon = 0.1 * cv2.arcLength(hull, True)
    corners = cv2.approxPolyDP(hull, epsilon, True)
    
    for corner in corners:
        cv2.circle(plane_mask, tuple(corner[0]), 5, (0, 0, 255), -1)
    
    cv2.drawContours(plane_mask, hull_list, -1, [255,0,0], thickness=2)
    output_image = transform_rgb_image_to_plane_view(img_undistorted, corners)      
    # Rotate the image 90 degress
    plane_mask = cv2.rotate(plane_mask, cv2.ROTATE_90_CLOCKWISE)
    segmented_rgb_image = cv2.rotate(segmented_rgb_image, cv2.ROTATE_90_CLOCKWISE)
    output_image = cv2.rotate(output_image, cv2.ROTATE_90_CLOCKWISE)
    
    # Show the original RGB image and the segmented RGB image.
    resize_image(plane_mask,"mask", 0.5)
    resize_image(segmented_rgb_image,"image", 0.5)
    resize_image(output_image,"output", 0.5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()