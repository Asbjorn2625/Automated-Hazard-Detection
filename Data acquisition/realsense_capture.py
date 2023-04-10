import pyrealsense2 as rs
import numpy as np
import cv2
import os


def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, int(width), int(height))
    cv2.imshow(image_name, image)


# Create a pipeline
pipeline = rs.pipeline()

# Create a configuration for the pipeline
config = rs.config()

# Add the RGB and depth streams to the configuration
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, int(1920/3), int(1080/3), rs.format.z16, 15)

# Start the pipeline
pipeline.start(config)

try:
    # Get the list of existing RGB and Depth image files in the save directory
    rgb_files = [f for f in os.listdir("./data") if f.startswith("rgb_image")]

    # Get the current highest index for RGB and Depth image names
    if len(rgb_files) > 0:
        curr_index = max([int(f.split("_")[2].split(".")[0]) for f in rgb_files])
    else:
        curr_index = 0
    while True:
        # Wait for the next set of frames from the camera
        frames = pipeline.wait_for_frames()

        # Get the RGB and depth frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Convert the RGB frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Convert the depth frame to a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        resize_image(color_image, 'color', 0.4)
        key = cv2.waitKey(1)
        if key == ord('s'):
            curr_index += 1
            # Save the RGB image
            cv2.imwrite("data/rgb_image_{:04d}.png".format(curr_index), color_image)

            # Save the depth image
            depth_image.tofile("data/depth_image_{:04d}.raw".format(curr_index))
        elif key == ord('q'):
            break
finally:
    # Stop the pipeline and release resources
    pipeline.stop()
