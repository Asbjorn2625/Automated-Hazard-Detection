import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def killTheExtraNumber(folder):
    # use os library to search for all files in a masks directory
    Mask_cwd = os.path.join(os.getcwd(), folder)
    file_list = os.listdir(Mask_cwd)

    # start finding duplicates
    for i, image1 in enumerate(file_list):
        imag1_split = image1.split("_")
        new_name=os.path.join(Mask_cwd, "%s_%s_%s.png" % (imag1_split[0], imag1_split[1], imag1_split[2])).replace("\\", "/")
        os.rename(os.path.join(Mask_cwd, image1).replace("\\", "/"), new_name)
        
                
def merge_masks(folder):
    # use os library to search for all files in a masks directory
    Mask_cwd = os.path.join(os.getcwd(), folder)
    file_list = os.listdir(Mask_cwd)

    # start finding duplicates
    for i, image1 in enumerate(file_list):
        imag1_split = image1.split("_")
        for j, image2 in enumerate(file_list):
            imag2_split = image2.split("_")
            if i != j and imag1_split[2] == imag2_split[2]:
                print("found duplicate")
                if imag1_split[3] != imag2_split[3]:
                    img1 = cv2.imread(os.path.join(Mask_cwd, image1))
                    img2 = cv2.imread(os.path.join(Mask_cwd, image2))
                    img1 = cv2.add(img1, img2)
                    cv2.imwrite(os.path.join(Mask_cwd, image1), img1)
                    os.remove(os.path.join(Mask_cwd, image2))
                    file_list.pop(j)
                    print("removed duplicate") 
def onlyTruths(folder): 
    mask_cwd = os.path.join(os.getcwd(), folder, "masks")
    rgb_cwd = os.path.join(os.getcwd(), folder, "RGB")  
    cwd = os.path.join(os.getcwd(), folder)
    mask_list = os.listdir(mask_cwd)
    rgb_list = os.listdir(rgb_cwd)
    mask_set = set(mask_list)
    not_in_mask = [img for img in rgb_list if img not in mask_set]
    for img in not_in_mask:
        os.remove(os.path.join(rgb_cwd, img))

def split_images(folder, train_ratio=0.8):
    # Create new folders for train and test data
    base_path = os.getcwd()
    train_mask_folder = os.path.join(base_path, folder, "train_mask")
    train_rgb_folder = os.path.join(base_path, folder, "train_rgb")
    test_mask_folder = os.path.join(base_path, folder, "test_mask")
    test_rgb_folder = os.path.join(base_path, folder, "test_rgb")
    os.makedirs(train_mask_folder, exist_ok=True)
    os.makedirs(train_rgb_folder, exist_ok=True)
    os.makedirs(test_mask_folder, exist_ok=True)
    os.makedirs(test_rgb_folder, exist_ok=True)
    
    # Get list of mask and RGB images
    mask_folder = os.path.join(base_path, folder, "masks")
    rgb_folder = os.path.join(base_path, folder, "rgb")
    mask_list = os.listdir(mask_folder)
    rgb_list = os.listdir(rgb_folder)
    
    # Split the images randomly into train and test sets
    mask_train, mask_test, rgb_train, rgb_test = train_test_split(
        mask_list, rgb_list, train_size=train_ratio, random_state=42)
    
    # Move the train images to the train folders and save as PNG
    for mask_file, rgb_file in tqdm(zip(mask_train, rgb_train), desc='Moving train images'):
        mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(os.path.join(rgb_folder, rgb_file))
        cv2.imwrite(os.path.join(train_mask_folder, mask_file), mask)
        cv2.imwrite(os.path.join(train_rgb_folder, rgb_file), rgb)
        
    # Move the test images to the test folders and save as PNG
    for mask_file, rgb_file in tqdm(zip(mask_test, rgb_test), desc='Moving test images'):
        mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(os.path.join(rgb_folder, rgb_file))
        cv2.imwrite(os.path.join(test_mask_folder, mask_file), mask)
        cv2.imwrite(os.path.join(test_rgb_folder, rgb_file), rgb)

        
split_images("src/functions/Merge_anntotes")
    

        


