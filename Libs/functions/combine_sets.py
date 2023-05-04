import os
import shutil
import re

base_folder = "C:\\Users\\Muku\\Desktop\\second set"
src_rgb_depth_1 = "first_rgb_depth"
src_masks_1 = "first_masks"
src_rgb_depth_2 = "second_rgb_depth"
src_masks_2 = "second_masks"

# Define the destination directories
dst_rgb_depth = "rgb_depth"
dst_masks = "masks"

# Create destination directories if they don't exist
os.makedirs(dst_rgb_depth, exist_ok=True)
os.makedirs(dst_masks, exist_ok=True)

def move_files(src_rgb_depth_dir, src_masks_dir, dst_rgb_depth_dir, dst_masks_dir):
    rgb_depth_files = [f for f in os.listdir(src_rgb_depth_dir) if f.startswith("rgb_image")]
    masks_files = [f for f in os.listdir(src_masks_dir) if f.startswith("rgb_image")]

    rgb_depth_files.sort()
    masks_files.sort()

    if rgb_depth_files:
        last_file = [f for f in os.listdir(dst_rgb_depth_dir) if f.startswith("rgb_image")]
        last_file.sort()
        if len(last_file) != 0:
            last_num = int(last_file[-1].split('_')[-1].split('.')[0])
        else:
            last_num = 0

        for file in rgb_depth_files:
            file_num = int(file.split('_')[-1].split('.')[0])
            new_num = last_num + file_num
            new_name = file.replace(str(file_num).zfill(4), str(new_num).zfill(4))

            # Get the individual files
            rgb_file_path = os.path.join(src_rgb_depth_dir, file)
            depth_file_path = os.path.join(src_rgb_depth_dir, file.replace("rgb", "depth").replace("png", "raw"))

            shutil.copy(rgb_file_path, os.path.join(dst_rgb_depth_dir, new_name))
            shutil.copy(depth_file_path, os.path.join(dst_rgb_depth_dir, new_name.replace("rgb", "depth").replace("png", "raw")))
            #print(file)
            for mask in masks_files:
                if file.split(".")[0] in mask:
                    mask_file_path = os.path.join(src_masks_dir, mask)
                    new_name = mask.replace(str(file_num).zfill(4), str(new_num).zfill(4))
                    shutil.copy(mask_file_path, os.path.join(dst_masks_dir, new_name))

# Move and rename files from the first set
move_files(os.path.join(base_folder,src_rgb_depth_1), os.path.join(base_folder,src_masks_1), dst_rgb_depth, dst_masks)

# Move and rename files from the second set
move_files(os.path.join(base_folder,src_rgb_depth_2), os.path.join(base_folder,src_masks_2), dst_rgb_depth, dst_masks)

print("Files have been moved and renamed successfully.")