import sys
import os
sys.path.append('/workspaces/Automated-Hazard-Detection')

from src.Preprocess.prep import PreProcess
from src.Data_acquisition.Image_fetcher import ImageFetcher
import pandas as pd
import cv2 

def alterate_set(path2imgs,path2excl):
    """[summary]
    Returns a dataframe with the images and the corresponding labels.
    """
    # Fetching the images
    img_fetcher = ImageFetcher(path2imgs)
    imglib = img_fetcher.get_rgb_depth_images()
    excl = pd.read_csv(path2excl)
    # Create a new dataframe with existing headers plus a new "filenames" column
    pandas_frame = pd.DataFrame(columns=list(excl.columns) + ['filenames'])
    print(pandas_frame)
    
alterate_set((os.getcwd() + "/images"), (os.getcwd() + "/Dangerous _goods_list_for_testing.csv"))




        
           
    
    



