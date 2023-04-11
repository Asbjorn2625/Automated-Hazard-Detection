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
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1.5  #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

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

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_image = aligned_frames.get_color_frame()

        # Get data from images
        color_image = np.asanyarray(color_image.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        # Get a displayable depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Remove background - Set pixels below the threshold as black
        grey_color = 0
        color_image[np.where((depth_image > clipping_distance) | (depth_image < 10))] = grey_color

        # Display the images
        resize_image(depth_colormap, 'depth', 0.4)
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
