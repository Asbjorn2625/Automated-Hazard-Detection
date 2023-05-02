import numpy as np
import cv2
import os


def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, int(width), int(height))
    cv2.imshow(image_name, image)


# Define the size of the checkerboard
checkerboard_size = (8, 6)
square_size = 40 # in mm

# Generate the object points
objp = np.zeros((np.prod(checkerboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Initialize lists to store object points and image points for all images
objpoints = []
imgpoints = []

# Create list of image filenames
image_names = [f'./Callibration images/{img}' for img in os.listdir("./Callibration images")]
# Loop through the,
for image in image_names:
    # Load the image
    img = cv2.imread(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the corners of the checkerboard
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    # If the corners were found, refine the corner locations
    if ret:
        #           Criteria,                                 , iterations, accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.0001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)

        resize_image(img, "result", 0.4)
        cv2.waitKey(5)  # Change if the corner detection has to be examined

        # Add the object and image points to the lists
        objpoints.append(objp)
        imgpoints.append(corners)

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Display the result
for image in image_names[-5:]:
    img = cv2.imread(image)
    # Undistort an image
    img_undistorted = cv2.undistort(img, mtx, dist)

    resize_image(img_undistorted, "result", 0.4)
    cv2.waitKey(0)

# Save the result
np.savez('calibration.npz', mtx=mtx, dist=dist)

