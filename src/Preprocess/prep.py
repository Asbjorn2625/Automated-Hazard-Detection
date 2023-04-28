import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import open3d as o3d


class PreProcess:
    def __init__(self):
        # Load the camera intrinsics
        with np.load("src/Data acquisition/calibration.npz") as a:
            self.mtx = a["mtx"]
            self.dist = a["dist"]
    def image_enhancer(self, image):
        # Convert numpy.ndarray to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Adjust brightness and gamma
        brightness_enhancer = ImageEnhance.Brightness(pil_image)
        brightened_img = brightness_enhancer.enhance(1.2)  # Adjust brightness by factor of 1.2
        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(brightened_img)
        contrast_enhanced_img = contrast_enhancer.enhance(1.2)  # 1.2 is the contrast factor
        sharpened_img = contrast_enhanced_img.filter(ImageFilter.SHARPEN)
        # Convert back to numpy.ndarray
        image = cv2.cvtColor(np.array(sharpened_img), cv2.COLOR_RGB2BGR)
        return image
    
    def undistort_images(self, img, depth):
        """
        Undistort depth and RGB images, based on the camera intrinsics.
        :param img: The RGB image, as a numpy.ndarray
        :param depth: The depth image, as a numpy.ndarray (dtype=np.uint16)
        :return: (numpy, numpy) The undistorted RGB image, and the undistorted depth image
        """
        # Undistort the images
        img_undistorted = cv2.undistort(img, self.mtx, self.dist)
        depth_undistorted = cv2.undistort(depth, self.mtx, self.dist)
        
        return img_undistorted, depth_undistorted
    
    def retrieve_transformed_plane(self, img, depth, max_planes = 5):
        """
        Transforms the RGB image and the depth image to a point cloud and segments front plane from the point cloud.
        :param img: The RGB image, as a numpy.ndarray
        :param depth: The depth image, as a numpy.ndarray (dtype=np.uint16)
        :param max_planes: The maximum number of planes to segment from the point cloud
        :return: (numpy, numpy) The transformed box plane, and the homography matrix
        """
        
        # Remove background - Set pixels below the threshold as black
        grey_color = 0
        depth[np.where((depth > 1000) | (depth < 10))] = grey_color
        
        # Convert the depth image to a point cloud
        point_cloud = self._depth_image_to_point_cloud(depth)
        min_angle_diff = float("inf")
        best_plane_inliers = None
        best_plane_normal_vector = None
        for i in range(max_planes):
            # Apply RANSAC to segment planes from the point cloud.
            plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

            # Calculate the normalized normal vector from the plane
            normal_vector = plane_model[:3] / np.linalg.norm(plane_model[:3])
            # Calculate the angle between the normal vector and the vertical direction (in degrees)
            angle_y = np.degrees(np.arccos(np.dot(normal_vector, np.array([1, 0, 0])))) # Our camera is turned, so the floor is on the horizontal
            angle_x = np.degrees(np.arccos(np.dot(normal_vector, np.array([0, 0, 1])))) # the closer to zero the closer to being orthogonal to the cameras optical axis
            
            # Remove the floor and noisy data
            if angle_y < 35 or len(inliers) < 80000: # floor is closer to 0 degrees, inliers descirbes amount of points in the plane
                # remove the points from the point cloud.
                plane_points = point_cloud.select_by_index(inliers)
                point_cloud = point_cloud.select_by_index(inliers, invert=True)
                continue
            if len(inliers) == 0:
                break  # No more planes found
            # Find the angle difference to the camera's optical axis
            angle_diff = abs(angle_x)
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                # Extract the plane points and remove them from the point cloud.
                best_plane_inliers = point_cloud.select_by_index(inliers)
                best_plane_normal_vector = point_cloud.select_by_index(inliers, invert=True)
        # In case there is nothing to be found
        if best_plane_inliers is None:
            return np.zeros_like(img)
        # Extract the best ones
        plane_points = best_plane_inliers
        point_cloud = best_plane_normal_vector
        
        # Remove points with low depth values (e.g., less than 0.1 meters)
        mask = np.asarray(point_cloud.points)[:, 2] > 0.1
        point_cloud = point_cloud.select_by_index(np.where(mask)[0])

        # Project the plane points back onto the image plane.
        plane_points_image = self._project_points_to_image(np.asarray(plane_points.points))
    
        # Filter out points with coordinates outside the image bounds.
        valid_indices = np.where((plane_points_image[:, 0] >= 0) & (plane_points_image[:, 0] < img.shape[1]) &
                            (plane_points_image[:, 1] >= 0) & (plane_points_image[:, 1] < img.shape[0]))[0]
        valid_points_image = plane_points_image[valid_indices]
        
        # Create binary mask
        plane_mask = np.zeros_like(img)
        plane_mask[valid_points_image[:, 1].round().astype(int), valid_points_image[:, 0].round().astype(int)] = 255
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        eroded = cv2.erode(cv2.cvtColor(plane_mask, cv2.COLOR_RGB2GRAY), kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=5)
        # Get the convex hull
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(contours[0])
        # Approximate the convex hull with a polygonal curve
        epsilon = 0.1 * cv2.arcLength(hull, True)
        corners = cv2.approxPolyDP(hull, epsilon, True)
        
        # Use the corners to transform the image to a plane view
        output_image, homography_matrix = self._transform_rgb_image_to_plane_view(img, corners)
        
        return output_image, homography_matrix
        
    def transformed_to_original_pixel(self, transformed_pixel, homography_matrix):
        """
        Convert pixel coordinates in the transformed image back to the original image.
        :param transformed_pixel: tuple or np.ndarray, (x, y) coordinates in the transformed image
        :param homography_matrix: np.ndarray, the homography matrix used to transform the original image
        :return: np.ndarray, (x, y) coordinates in the original image
        """
        transformed_pixel = np.array([*transformed_pixel, 1])
        original_pixel = np.linalg.inv(homography_matrix) @ transformed_pixel
        original_pixel = original_pixel / original_pixel[2]
        return original_pixel[:2].round().astype(int)
        
    def segmentation_to_ROI(self, image):
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thr = cv2.threshold(grey, 0, 200, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Roi = []
        for cnt in contours:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(cnt)
            # Extract the four corner coordinates of the ROI
            xStart = x
            xEnd = x + w
            yStart = y
            yEnd = y + h
            Roi.append([xStart, yStart, xEnd, yEnd])
        return Roi
    
    def _depth_image_to_point_cloud(self, depth_image):
        """
        Convert a depth image to a point cloud.
        :param depth_image: The depth image, np.ndarray with dtype uint16.
        """
        camera_intrinsics = self.mtx
        height, width = depth_image.shape

        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]

        # Create a grid of (x,y,z) coordinates corresponding to the pixels of the depth image.
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        xx = (xx - cx) * depth_image / fx
        yy = (yy - cy) * depth_image / fy

        # Stack the coordinates to a point cloud and convert to meters.
        points = np.column_stack((xx.ravel(), yy.ravel(), depth_image.ravel())) / 1000  # Convert to meters
        # Send the points to open3d.PointCloud.
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        return point_cloud
    
    def _project_points_to_image(self, points):
        """
        project cloud points to image plane
        :points: np.ndarray
        :return: points in image plane
        """
        # Filter out points with a z-coordinate close to zero.
        valid_points = points[np.abs(points[:, 2]) > 1e-6]
        
        # Project the valid points onto the image plane.
        uv = (valid_points @ self.mtx.T)[:, :2] / valid_points[:, 2, np.newaxis]
        return uv.round().astype(int)
    
    def _transform_rgb_image_to_plane_view(self, rgb_image, plane_points):
        """
        transform the RGB image depending on the plane points
        :param rgb_image: np.ndarray, RGB image
        :param plane_points: np.ndarray, [[[x1, y1]], [[x2, y2]], [[x3, y3]], [[x4, y4]]]
        :return: (np array, np array), transformed image and homography matrix
        """
        # Reshape the plane_points to (4, 2)
        plane_points = plane_points.reshape(-1, 2)

        # Sort the plane points based on their y-coordinates
        sorted_points = plane_points[np.argsort(plane_points[:, 1]), :]

        # Get the closest two points and compute the height of the output image
        closest_points = sorted_points[:2, :]
        output_height = int(np.linalg.norm(closest_points[0] - closest_points[-1]))

        # Compute the width of the output image based on the aspect ratio
        aspect_ratio = rgb_image.shape[1] / rgb_image.shape[0]
        output_width = int(output_height * aspect_ratio)
        # Define the destination points for the transformation (rectangle)
        destination_points = np.array([
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1]
        ], dtype=np.float32)

        # Compute the homography matrix
        homography_matrix, _ = cv2.findHomography(plane_points, destination_points)

        # Warp the RGB image using the computed homography matrix
        transformed_image = cv2.warpPerspective(rgb_image, homography_matrix, (output_width, output_height))

        return transformed_image, homography_matrix