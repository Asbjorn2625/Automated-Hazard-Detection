import cv2
import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

sys.path.append('/workspaces/P6-Automated-Hazard-Detection')
sys.path.append('/workspaces/Automated-Hazard-Detection')
from src.Preprocess.prep import PreProcess


def get_pixelsize(depth):
    fov = np.deg2rad([69, 42])
    # 2 * depth * tan(FOV / 2) * (object width in pixels / image width in pixels)
    width = 2 * depth * np.tan(fov[0]/2)/1920
    height = 2 * depth * np.tan(fov[1]/2)/1080
    return width, height

def merge_masks(mask_folder, mask_list):
    # start finding duplicates
    for i, image1 in enumerate(mask_list):
        mask_name1 = image1[:-10]
        for j, image2 in enumerate(mask_list):
            mask_name2 = image2[:-10]
            if (i != j) and (mask_name1 == mask_name2):
                img1 = cv2.imread(os.path.join(mask_folder, image1), 0)
                img2 = cv2.imread(os.path.join(mask_folder, image2), 0)
                img1 = cv2.add(img1, img2)
                cv2.imwrite(os.path.join(mask_folder, image1), img1)
                os.remove(os.path.join(mask_folder, image2))
                mask_list.pop(j)
    return mask_list


def get_diamond_corners(mask, contour_area_threshold=100, epsilon_ratio=0.02):
    # Find the contours of the diamond shapes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_corner_groups = []

    # Loop through the contours
    for contour in contours:
        # Filter out small contours by area
        if cv2.contourArea(contour) > contour_area_threshold:
            # Approximate the contour shape
            epsilon = epsilon_ratio * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the shape has 4 vertices (corners)
            if len(approx) == 4:
                # Extract the corner coordinates
                corner_coordinates = [tuple(coord[0]) for coord in approx]

                # Append the corner coordinates to the list of corner groups
                all_corner_groups.append(corner_coordinates)

    return all_corner_groups


def distance_between_corners(corner1, corner2, depth_map):
    x1, y1 = corner1
    x2, y2 = corner2

    # Calculate the number of steps to iterate through the pixels
    num_steps = max(abs(x1 - x2), abs(y1 - y2))

    # Calculate the step size for x and y directions
    step_x = (x2 - x1) / num_steps
    step_y = (y2 - y1) / num_steps

    # Initialize the total distance
    total_distance = 0

    # Iterate through the pixels between the corners
    for step in range(1, num_steps + 1):
        # Calculate the current x and y coordinates
        x = int(round(x1 + step_x * step))
        y = int(round(y1 + step_y * step))

        # Get the depth value for the current pixel
        depth = depth_map[y, x]

        # Calculate the pixel size for the current pixel
        pixel_size = get_pixelsize(depth)

        # Calculate the real-life distance between the current pixel and the previous pixel
        real_distance_x = abs(step_x) * pixel_size[0]
        real_distance_y = abs(step_y) * pixel_size[1]

        # Calculate the Euclidean distance between the current pixel and the previous pixel
        distance = np.sqrt(real_distance_x ** 2 + real_distance_y ** 2)

        # Add the distance to the total distance
        total_distance += distance

    return total_distance

# Function to calculate the average of a specific entry
def average_entry(entry_name, data):
    count = 0
    total_columns = {"1": 0, "2": 0, "3": 0, "4": 0}

    for item in data:
        if item["Degree"] == entry_name:
            count += 1
            total_columns["1"] += list(item.values())[1]
            total_columns["2"] += list(item.values())[2]
            total_columns["3"] += list(item.values())[3]
            total_columns["4"] += list(item.values())[4]

    if count > 0:
        keys = list(data[0].keys())
        avg_columns = {
            "Degree": entry_name,
            keys[1]: total_columns["1"] / count,
            keys[2]: total_columns["2"] / count,
            keys[3]: total_columns["3"] / count,
            keys[4]: total_columns["4"] / count

        }
        return avg_columns
    else:
        return None

if __name__ == "__main__":
    pp = PreProcess()
    results = []
    # Get the images
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    image_folder = os.path.join(curr_folder, "Label_size_test")
    mask_folder = os.path.join(image_folder, "masks")
    # Import the csv file
    csv_file = os.path.join(image_folder, "label_sizes.csv")
    df = pd.read_csv(csv_file)

    # Get the images
    image_list = [img for img in os.listdir(image_folder) if img.startswith("rgb")]
    mask_list = [img for img in os.listdir(mask_folder) if img.startswith("rgb")]

    # Merge the masks
    mask_list = merge_masks(mask_folder, mask_list)

    # Add these lines before the for loop that iterates through the images
    degree_errors = {}
    degree_counts = {}
    # Loop through the images
    for mask_name in mask_list:
        degree = mask_name.split("_")[1]
        image_name = mask_name[:-10]+".png"
        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, mask_name)
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        depth = np.fromfile(image_path.replace("rgb", "depth").replace("png", "raw"), dtype=np.uint16)

        # Reconstruct the depth map
        depth = depth.reshape(int(1080), int(1920))
        # Undistort the images
        img = pp.undistort_images(img)
        mask = pp.undistort_images(mask)
        depth = pp.undistort_images(depth)

        # Create a copy of the image for visualization
        img_vis = img.copy()

        # Get the corner groups of the diamond shapes
        corner_groups = get_diamond_corners(mask)
        distances = []
        # Calculate the real-life distance of the sides for each contour
        for group_idx, corner_group in enumerate(corner_groups):
            num_corners = len(corner_group)
            side_distances = []

            for i in range(num_corners):
                corner1 = corner_group[i]
                corner2 = corner_group[(i + 1) % num_corners]  # Get the next corner in the group
                distance = distance_between_corners(corner1, corner2, depth)
                side_distances.append(distance)

            # Calculate the average distance of the sides for the current contour
            average_distance = sum(side_distances) / num_corners
            distances.append(average_distance)

        # Sort the distances
        distances.sort()
        # Compare them to the csv file
        individual_error = {"1": 0, "2": 0, "3": 0, "4": 0}
        print(f"Degree: {degree}")
        for i, distance in enumerate(distances):
            csv_distance = df[str(i+1)].item()
            print(f"Distance {i+1}: {distance} mm")
            print(f"CSV distance {i+1}: {csv_distance} mm")
            print(f"Error: {abs(distance-csv_distance)} mm")
            individual_error[str(i+1)] = abs(distance-csv_distance)
        # Append the results to the list
        results.append({
            'Degree': degree,
            str(df["1"].item())+"mm": individual_error["1"],
            str(df["2"].item())+"mm": individual_error["2"],
            str(df["3"].item())+"mm": individual_error["3"],
            str(df["4"].item())+"mm": individual_error["4"]
        })

    # Get unique names from the dataset
    unique_names = set(item["Degree"] for item in results)

    # Calculate the average for each unique name and store it in a new list
    averaged_data = []
    for name in unique_names:
        result = average_entry(name, results)
        if result:
            averaged_data.append(result)


    def format_degree(value):
        return f"{value}$^\circ$"


    def format_error(value):
        return f"{value} mm"

    results_df = pd.DataFrame(averaged_data).sort_values(by='Degree').round(2)
    results_df['Degree'] = results_df['Degree'].apply(format_degree)
    for column in results_df.columns[1:]:
        results_df[column] = results_df[column].apply(format_error)
    with open('label_size_results.tex', 'w') as f:
        f.write(results_df.to_latex(index=False, escape=False))

