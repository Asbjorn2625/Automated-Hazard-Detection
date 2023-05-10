import sys
sys.path.append('/workspaces/Automated-Hazard-Detection')

from src.Segmentation.segmentation import Segmentation
from src.Preprocess.prep import PreProcess
from src.Data_acquisition.Image_fetcher import ImageFetcher

from src.Text_reader.ReaderClass import ReadText
import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import csv
import pandas as pd
import Levenshtein



def levensthein_distance(color):
    Reading_test_output = "/workspaces/Automated-Hazard-Detection/testing/reading_test_output.csv"
    color = color.replace("  ", "")
    # read the CSV file into a DataFrame
    df = pd.read_csv(Reading_test_output)
    levenshtein_distance_list = []

    testing_df = pd.DataFrame(columns=['picturename', 'size', 'orientation','background_color','read_text','ground_truth_text'])

    # Select the rows with the current picture name
    selected_rows = df.loc[df['background_color'] == color]

    # Use the selected rows in your code

    for index, row in selected_rows.iterrows():
        # Do something with the row data
        current_row = {'picturename': row['picturename'], 'size': row['size'], 'orientation': row['orientation'],'background_color': row['background_color'],'read_text': row['read_text'],'ground_truth_text': row['ground_truth_text'] }
        testing_df = pd.concat([testing_df, pd.DataFrame(current_row, index=[0])], ignore_index=True)

    for index, row in testing_df.iterrows():
        read_string = row["read_text"]
        ground_truth = row["ground_truth_text"]
        
        if isinstance(read_string, str) and len(read_string) > 0:
            read_string = read_string.replace(" ", "")
            ground_truth = ground_truth.replace(" ", "")
            levenshtein_distance = Levenshtein.distance(read_string, ground_truth)
            levenshtein_distance_list.append(levenshtein_distance)
        else:
            levenshtein_distance = 17   # worst case for the ground truth
            levenshtein_distance_list.append(levenshtein_distance)  

    levenshtein_distance_mean = sum(levenshtein_distance_list) / len(levenshtein_distance_list)       
    print(color, "Levenshtein's distance is", levenshtein_distance_mean)
    return levenshtein_distance_mean


levensthein_distance("yellow")
levensthein_distance("brown")
levensthein_distance("blue")
levensthein_distance("green")
levensthein_distance("black")
levensthein_distance("red")
levensthein_distance("white")
levensthein_distance("orange")

"""
    # Define custom weights for different characters or transformations
    weights = {
        ("4", "A"): 0.2,  # Cost of changing 4 to A is 0.2
        ("D", "0"): 0.2, # Cost of changing D to D is 0.2
        ("/", "I"): 0.5,
        ("/", "1"): 0.5,
        ("S", "3"): 0.2,
        ("Y", "V"): 0.2,
        "default": 1  # Default cost for any other character change is 1
        }
"""