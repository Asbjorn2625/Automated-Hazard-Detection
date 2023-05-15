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

def display_inlier_outlier_points(point_cloud, inliers, iteration, title='Inlier and Outlier Points'):
    inlier_cloud = point_cloud.select_by_index(inliers)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)

    inlier_cloud.paint_uniform_color([1, 0, 0])  # Red color for inliers
    outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color for outliers

    combined_cloud = inlier_cloud + outlier_cloud
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(f"{title} - Iteration {iteration}", width=int(800*2), height=1000)
    vis.add_geometry(combined_cloud)

    # Configure visualization options
    render_option = vis.get_render_option()
    render_option.point_size = 2  # Set point size to 1 for smaller points

    # Display the point cloud
    vis.run()
    vis.destroy_window()

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



pp = PreProcess()
# Get current folder
curr_dir = os.path.dirname(os.path.abspath(__file__))
curr_dir = os.path.join(curr_dir, "for_miki")
# Create list of image filenames
rgb_images = [os.path.join(curr_dir, img) for img in os.listdir(curr_dir) if img.startswith("rgb_image")]

# Loop through the images
for image in rgb_images:
    print(image)
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
    point_cloud = pp._depth_image_to_point_cloud(depth)
    
    # Remove background - Set pixels below the threshold as black
    grey_color = 0
    depth[np.where((depth > 1000) | (depth < 10))] = grey_color
    
    # Convert the depth image to a point cloud
    point_cloud = pp._depth_image_to_point_cloud(depth)
    min_angle_diff = float("inf")
    best_plane_inliers = None
    best_plane_normal_vector = None
    for i in range(5):
        # Apply RANSAC to segment planes from the point cloud.
        plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.008, ransac_n=8, num_iterations=10000)
        display_inlier_outlier_points(point_cloud, inliers, i, title='Inlier and Outlier Points')

        # Calculate the normalized normal vector from the plane
        normal_vector = plane_model[:3] / np.linalg.norm(plane_model[:3])
        # Calculate the angle between the normal vector and the vertical direction (in degrees)
        angle_y = np.degrees(np.arccos(np.dot(normal_vector, np.array([1, 0, 0])))) # Our camera is turned, so the floor is on the horizontal
        angle_x = np.degrees(np.arccos(np.dot(normal_vector, np.array([0, 0, 1])))) # the closer to zero the closer to being orthogonal to the cameras optical axis
        
        # Parameters for filtering planes
        min_distance_from_camera = 0.2  # In meters
        min_cardboard_area = 80000
        min_cardboard_normal_angle = 35

        # Remove the floor and noisy data
        plane_points = point_cloud.select_by_index(inliers)
        mean_point = np.mean(np.asarray(plane_points.points), axis=0)

        # Calculate distance from the camera
        distance_from_camera = np.linalg.norm(mean_point)

        if (angle_y < min_cardboard_normal_angle or
            len(inliers) < min_cardboard_area or
            distance_from_camera < min_distance_from_camera):
            # Remove the plane points from the point cloud.
            point_cloud = point_cloud.select_by_index(inliers, invert=True)
            continue
        # Find the angle difference to the camera's optical axis
        angle_diff = abs(angle_x)
        if angle_diff < min_angle_diff:
            min_angle_diff = angle_diff
            # Extract the plane points and remove them from the point cloud.
            best_plane_inliers = point_cloud.select_by_index(inliers)
            best_plane_normal_vector = point_cloud.select_by_index(inliers, invert=True)

    # Extract the best ones
    plane_points = best_plane_inliers
    point_cloud = best_plane_normal_vector
    
    # Remove points with low depth values (e.g., less than 0.1 meters)
    background_mask = np.asarray(point_cloud.points)[:, 2] > 0.1
    point_cloud = point_cloud.select_by_index(np.where(background_mask)[0])

    # Project the plane points back onto the image plane.
    plane_points_image = pp._project_points_to_image(np.asarray(plane_points.points))

    # Filter out points with coordinates outside the image bounds.
    valid_indices = np.where((plane_points_image[:, 0] >= 0) & (plane_points_image[:, 0] < img.shape[1]) &
                        (plane_points_image[:, 1] >= 0) & (plane_points_image[:, 1] < img.shape[0]))[0]
    valid_points_image = plane_points_image[valid_indices]
    
    # Create binary mask
    plane_mask = np.zeros_like(img)
    plane_mask[valid_points_image[:, 1].round().astype(int), valid_points_image[:, 0].round().astype(int)] = 255
    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    eroded = cv2.erode(cv2.cvtColor(plane_mask, cv2.COLOR_RGB2GRAY), kernel, iterations=3)
    dilated = cv2.dilate(eroded, kernel, iterations=7)
    # Get the convex hull
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    
    # Approximate the convex hull with a polygonal curve
    epsilon = 0.08 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    if len(approx) == 4:
        corners = np.squeeze(approx)
    else:
        # Fallback in case the contour approximation doesn't result in 4 points
        print(f"Warning: Found {len(approx)} points instead of 4. Using the 4 most extreme points as corners.")
        corners = np.zeros((4, 2), dtype=np.float32)
        corners[0] = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])  # leftmost point
        corners[1] = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])  # topmost point
        corners[2] = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])  # rightmost point
        corners[3] = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])  # bottommost point


    output_image, homography_matrix = pp._transform_image_to_plane_view(img, corners)
    
    
    # Display the results
    display_depth_image(filtered_depth_data, title='Filtered Depth Data')
    #display_projected_points(img, plane_points, title='Projected Points on Image')
    display_contour_and_corners(img, largest_contour, corners, title='Detected Contour and Corners')
    display_warped_image(output_image, title='Warped Perspective')
    plt.show()
    
    #display_point_cloud(point_cloud, title='Filtered Point Cloud')
