import os
import shutil
import pandas as pd

base_folder = r"C:\Users\Muku\Desktop\second set"
src_data_set_1 = os.path.join(base_folder, "first data set")
src_data_set_2 = os.path.join(base_folder, "second data set")
src_masks_set = os.path.join(base_folder, "masks")
csv_file = os.path.join(base_folder, "instance_mask_annotations.csv")

# Define the destination directories
dst_rgb_depth = os.path.join(base_folder, "rgb_depth")
dst_masks = os.path.join(base_folder, "masks_")


def increment_set_name(set_name):
    # Split the image name into prefix and number
    number = set_name.split('_')[-1].split(".")[0]

    # Convert the number to an integer, increment it, and then format it back to a string with leading zeros
    new_number = '{:04d}'.format(int(number) + 1)

    # Combine the prefix and the incremented number to create the new image name
    new_image_name = set_name.replace(number, new_number)

    return new_image_name

def move_files(src_image_folder, src_mask_folder, dst_image_folder, dst_mask_folder, file_dic, start_num=0):
    # Sort the files according to type of mask
    grouped = file_dic.groupby('class_name')

    # Create an empty dictionary to store the data for each class_name
    class_dict = {}

    # Iterate over the groups, converting each group's data to a list of tuples
    for class_name, group in grouped:
        class_dict[class_name] = list(zip(group['image_id'], group['mask_id']))

    # Run through the dictionary
    for type in class_dict:
        # Run through the files in the dictionary
        for file in class_dict[type]:
            # Get the file name
            file_name = file[0]+".png"
            depth_name = file_name.replace("rgb_image", "depth_image").replace("png", "raw")
            # Get the file number
            file_num = int(file_name.split('_')[-1].split('.')[0])
            # Get the new number
            new_num = start_num + file_num
            # Get the new name
            new_img_name = file_name.replace(str(file_num).zfill(4), str(new_num).zfill(4))
            new_depth_name = depth_name.replace(str(file_num).zfill(4), str(new_num).zfill(4))
            new_mask_name = file[1].replace(str(file_num-1).zfill(4), str(new_num).zfill(4)) + ".png"
            new_mask_name = new_mask_name.replace("rgb1_", "rgb_")
            # Get the file paths
            image_file_path = os.path.join(src_image_folder, file_name)
            depth_file_path = os.path.join(src_image_folder, depth_name)
            mask_file_path = os.path.join(src_mask_folder, file[1] + ".png")
            # Get the destination folders
            mask_dst = dst_mask_folder + type
            os.makedirs(mask_dst, exist_ok=True)
            os.makedirs(dst_image_folder, exist_ok=True)
            # Copy the files to the destination folders
            shutil.copy(image_file_path, os.path.join(dst_image_folder, new_img_name))
            shutil.copy(depth_file_path, os.path.join(dst_image_folder, new_depth_name))
            shutil.copy(mask_file_path, os.path.join(mask_dst, new_mask_name))


# Read csv file
df = pd.read_csv(csv_file)

# Get the 2 sets
first_set = df[df['image_id'].str.startswith("rgb_image_")].copy()
second_set = df[df['image_id'].str.startswith("rgb1_image_")].copy()
# Fix the second set
second_set["image_id"] = second_set['image_id'].str.replace("rgb1_", "rgb_")
second_set["image_id"] = second_set["image_id"].apply(increment_set_name)

# Get the highest number of the last file
if not first_set.empty:
    # Extract the number from the image_id string
    first_set['number'] = first_set['image_id'].str.extract(r'(\d+)$').astype(int)
    second_set['number'] = second_set['image_id'].str.extract(r'(\d+)$').astype(int)
    # Find the highest number
    highest_number = first_set['number'].max()
    max = second_set['number'].max()
    # Create a boolean mask for rows in the second_set with the highest number
    mask = second_set['number'] != max

    # Drop the rows with the highest number from the second_set using the boolean mask
    second_set = second_set[mask]

    print(f"The highest number is: {highest_number}")
else:
    print("No matching files were found.")
    exit()

# Move the 2 sets
move_files(src_data_set_1, src_masks_set, dst_rgb_depth, dst_masks, first_set)
move_files(src_data_set_2, src_masks_set, dst_rgb_depth, dst_masks, second_set, start_num=highest_number)

print("Files have been moved and renamed successfully.")