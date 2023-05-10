import sys
import os
sys.path.append('/workspaces/P6-Automated-Hazard-Detection')

from src.Preprocess.prep import PreProcess
from src.Data_acquisition.Image_fetcher import ImageFetcher
import pandas as pd
import cv2

def alterate_set(path2imgs, path2excl):
    """[summary]
    Returns a dataframe with the images and the corresponding labels.
    """
    excl = pd.read_csv(path2excl)

    # Add a new column "filenames" to the existing DataFrame
    excl['filenames'] = None

    # Fetching the images
    img_fetcher = ImageFetcher(path2imgs)
    imglib = img_fetcher.get_rgb_depth_images()

    # Group the filenames into sets of four
    filenames_list = list(imglib.keys())
    grouped_filenames = [filenames_list[i:i + 4] for i in range(0, len(filenames_list), 4)]

    for index, row in excl.iterrows():
        # Fetch the image filenames related to the current row based on the chronological order
        try:
            filenames = grouped_filenames[index]
        except IndexError:
            print(f"No more sets of 4 images available for row {index}")
            continue

        # Update the "filenames" column in the DataFrame with the list of filenames
        excl.at[index, 'filenames'] = filenames
    return excl

list=alterate_set((os.getcwd() + "/images"), (os.getcwd() + "/Dangerous_goods_list_for_testing.csv"))
print(list.loc[0])