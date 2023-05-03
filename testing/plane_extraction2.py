import sys
sys.path.append('/workspaces/P6-Automated-Hazard-Detection')

from src.Preprocess.prep import PreProcess
import os
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA



####### DEBUG AND DISPLAY FUNCTIONS #######
def display_depth_image(depth_image, title='Depth Image'):
    plt.imshow(depth_image, cmap=plt.cm.viridis)
    plt.title(title)
    plt.axis('off')
    plt.show()

def display_point_cloud(point_cloud, title='Point Cloud'):
    points = np.asarray(point_cloud.points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

def display_inlier_points(point_cloud, inliers, title='Inlier Points'):
    inlier_cloud = point_cloud.select_by_index(inliers)
    display_point_cloud(inlier_cloud, title)

def display_projected_points(rgb_image, projected_points, title='Projected Points'):
    img_with_points = rgb_image.copy()
    for point in projected_points:
        cv2.circle(img_with_points, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)

    plt.imshow(cv2.cvtColor(img_with_points, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def display_contour_and_corners(rgb_image, contour, corners, title='Contour and Corners'):
    img_with_contour = rgb_image.copy()
    cv2.drawContours(img_with_contour, [contour], 0, (0, 255, 0), 2)
    for corner in corners:
        cv2.circle(img_with_contour, tuple(corner), 5, (0, 0, 255), -1)

    plt.imshow(cv2.cvtColor(img_with_contour, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def display_warped_image(warped_image, title='Warped Image'):
    plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

####### USEFUL FUNCTIONS #######
def depth_biliteral(depth_data, d=5, sigma_color=50, sigma_space=50):
    depth_data = depth_data.astype(np.float32)
    
    # Normalize depth data to the range [0, 1]
    depth_data /= np.max(depth_data)
    
    # Apply a bilateral filter
    filtered_depth_data = cv2.bilateralFilter(depth_data, d, sigma_color, sigma_space)
    
    # Convert the filtered depth data back to its original scale
    filtered_depth_data *= np.max(depth_data)
    filtered_depth_data = filtered_depth_data.astype(np.uint16)
    
    return filtered_depth_data

def depth_to_point_cloud(depth_image, intrinsics_matrix, min_depth=30, max_depth=1500):
    points = []
    height, width = depth_image.shape
    for v in range(height):
        for u in range(width):
            depth = depth_image[v, u]
            if depth > min_depth and depth < max_depth:
                x = (u - intrinsics_matrix[0, 2]) * depth / intrinsics_matrix[0, 0]
                y = (v - intrinsics_matrix[1, 2]) * depth / intrinsics_matrix[1, 1]
                z = depth
                points.append([x, y, z])
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(points)))

def extract_plane_ransac(point_cloud, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    plane_model, inliers = point_cloud.segment_plane(distance_threshold, ransac_n, num_iterations)
    return plane_model, inliers

def angle_between_vectors(n, k):
    cos_theta = np.dot(n, k) / (np.linalg.norm(n) * np.linalg.norm(k))
    angle = np.arccos(np.clip(cos_theta, -1, 1))
    return np.degrees(angle)

def project_points_to_image(points, intrinsics_matrix):
    image_points = []
    for point in points:
        point_homogeneous = np.array([point[0], point[1], point[2]])
        image_point_homogeneous = intrinsics_matrix @ point_homogeneous
        image_point = image_point_homogeneous[:2] / image_point_homogeneous[2]
        image_points.append(image_point)
    return np.array(image_points)

def get_contour_corners(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)

def compute_normals(point_cloud, radius=0.01, max_nn=30):
    point_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return point_cloud

def filter_points_by_angle(point_cloud, axis, max_angle_degrees):
    max_angle_rad = np.deg2rad(max_angle_degrees)
    cos_max_angle = np.cos(max_angle_rad)
    normals = np.asarray(point_cloud.normals)
    
    axis_dot_normals = normals @ axis
    mask = axis_dot_normals <= cos_max_angle
    filtered_points = point_cloud.select_by_index(np.where(mask)[0])

    return filtered_points




pp = PreProcess()

# Create list of image filenames
rgb_images = [f'./testing/first data set/{img}' for img in os.listdir("./testing/first data set") if img.startswith("rgb_image")]

# Loop through the images
for image in rgb_images:
    img = cv2.imread(image)
    depth = np.fromfile(image.replace("rgb_image", "depth_image").replace("png", "raw"), dtype=np.uint16)
    # Reconstruct the depth map
    depth = depth.reshape(int(1080), int(1920))
    
    # undistort the image
    img = pp.undistort_images(img)
    depth = pp.undistort_images(depth)
    
    # Preprocess the depth data
    # Apply a bilateral filter
    #filtered_depth_data = depth_biliteral(depth)

    # Optionally, apply a median filter
    filtered_depth_data = cv2.medianBlur(depth, 5)
    
    # Get the point cloud
    point_cloud = depth_to_point_cloud(filtered_depth_data, pp.mtx)
    
    # Filter out the floor
    filtered_point_cloud = point_cloud

    
    min_angle_diff = float("inf")
    best_plane_inliers = None
    best_plane_normal_vector = None
    for i in range(5):
        # Apply RANSAC to segment planes from the point cloud.
        plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

         # Calculate the normalized normal vector
        normal_vector = plane_model[:3] / np.linalg.norm(plane_model[:3])
        # Calculate the angle between the normal vector and the vertical direction (in degrees)
        angle_y = np.degrees(np.arccos(np.dot(normal_vector, np.array([1, 0, 0])))) # Our camera is turned, so the floor is on the horizontal
        angle_x = np.degrees(np.arccos(np.dot(normal_vector, np.array([0, 0, 1])))) # the closer to zero the closer to being orthogonal to the cameras optical axis
        print(angle_x, angle_y)
        # Remove the floor and noisy data
        if angle_y < 35 or len(inliers) < 800:
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
    
    inlier_cloud = filtered_point_cloud.select_by_index(best_plane_inliers)
    inlier_points = np.asarray(inlier_cloud.points)
    
    projected_points = project_points_to_image(inlier_points, pp.mtx)
    
    binary_image = np.zeros(img.shape[:2], dtype=np.uint8)
    for point in projected_points:
        binary_image[int(point[1]), int(point[0])] = 255

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    box_corners = get_contour_corners(largest_contour)
    
    width, height = 400, 300  # Define the desired dimensions of the target plane
    target_corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    homography, _ = cv2.findHomography(box_corners, target_corners)
    
    warped_image = cv2.warpPerspective(img, homography, (width, height))
    
    
    # Display the results
    display_depth_image(filtered_depth_data, title='Filtered Depth Data')
    display_projected_points(img, projected_points, title='Projected Points on Image')
    display_contour_and_corners(img, largest_contour, box_corners, title='Detected Contour and Corners')
    display_warped_image(warped_image, title='Warped Perspective')
    plt.show()
    
    display_point_cloud(point_cloud, title='Filtered Point Cloud')
