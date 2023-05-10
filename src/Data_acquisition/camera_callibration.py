import pyrealsense2 as rs
import numpy as np

# Create a context object to manage the connection to the device
ctx = rs.context()
if len(ctx.devices) == 0:
    print("No RealSense devices were found")
    exit(1)

# Get the first connected device
dev = ctx.devices[0]

# Get the depth sensor and depth stream profile
color_sensor = dev.first_color_sensor()
depth_stream_profile = color_sensor.get_stream_profiles()[0].as_video_stream_profile()

# Get the intrinsic values
intrinsics = depth_stream_profile.get_intrinsics()

# Create a camera matrix (mtx) and distortion coefficients (dist) from intrinsics
mtx = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1]])

dist = np.array([intrinsics.coeffs])

# Save the camera matrix and distortion coefficients to a file
np.savez('calibration.npz', mtx=mtx, dist=dist)