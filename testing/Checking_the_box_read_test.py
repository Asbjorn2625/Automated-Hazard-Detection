import sys
sys.path.append('/workspaces/Automated-Hazard-Detection')
sys.path.append('/workspaces/P6-Automated-Hazard-Detection')

from src.Segmentation.segmentation import Segmentation
from src.Preprocess.prep import PreProcess
#from src.Data_acquisition.Image_fetcher import ImageFetcher

from src.Text_reader.ReaderClass import ReadText
import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import csv
import pandas as pd
import Levenshtein



def levensthein_distance(read_string, ground_truth = "4G/Y30/S/22/D/BAM"):

    if isinstance(read_string, str) and len(read_string) > 0:
        read_string = read_string.replace(" ", "")
        ground_truth = ground_truth.replace(" ", "")
        levenshtein_distance = Levenshtein.distance(read_string, ground_truth)
    else:
        levenshtein_distance = 17   # worst case for the ground truth
    
    
    
    return levenshtein_distance



#Folder path
current_file_path = os.path.abspath(__file__)
# Get the folder containing the current file
current_folder = os.path.dirname(current_file_path)


Reading_test_output = os.path.join(current_folder, "reading_test_output.csv")
# read the CSV file into a DataFrame
df = pd.read_csv(Reading_test_output)

size_test_df = pd.read_csv(Reading_test_output)

size_test_df = size_test_df.drop(["picturename", "ground_truth_text","background_color","read_text"], axis= 1 )

size_test_df.dropna(subset=['detected_text_size'], inplace=True)

print(size_test_df)

#size_test_df = size_test_df.groupby(["orientation", "size"]).mean()


# Use pivot_table to group by 'name' and 'color', then pivot on 'size' and 'value'
size_test_pivot_df = size_test_df.pivot_table(index='orientation', columns='size', values='detected_text_size', aggfunc='mean').reset_index()

# Rename the columns to make them more understandable
#size_test_pivot_df.columns.name = None

print(size_test_pivot_df)

print(size_test_pivot_df.keys())


# Round the numeric columns to 3 decimal places
rounded_df_size_test = size_test_pivot_df.round(3)

# Convert the DataFrame to a LaTeX table with controlled float format
rounded_df_size_test = rounded_df_size_test.to_latex(index=False, float_format="%.3f")

# Print the LaTeX table
print(rounded_df_size_test)



testing_df = pd.DataFrame(columns=['picturename', 'size', 'orientation','background_color','read_text','ground_truth_text'])


sizes = [5.5,5.9,6.1,6.9,7.1,7.9]
# Now group by 'mask' and 'model', and take the mean of the other columns

df.drop("detected_text_size", axis=1)

df["read_text"] = df["read_text"].apply(levensthein_distance)


# Use pivot_table to group by 'name' and 'color', then pivot on 'size' and 'value'
df_pivot = df.pivot_table(index=['orientation', 'background_color'], columns='size', values='read_text', aggfunc='sum').reset_index()

# Rename the columns to make them more understandable
df_pivot.columns.name = None

print(df_pivot)

print(df_pivot.keys())



df_color = df_pivot.drop("orientation", axis=1)


df_orientation = df_pivot.drop('background_color', axis=1)


df_color = df_color.groupby('background_color').mean()

df_orientation = df_orientation.groupby('orientation').mean()


print(df_color)

print(df_orientation)

# Round the numeric columns to 3 decimal places
rounded_df_color = df_color.round(3)

# Convert the DataFrame to a LaTeX table with controlled float format
color_latex_table = rounded_df_color.to_latex(index=False, float_format="%.3f")

# Print the LaTeX table
print(color_latex_table)


# Round the numeric columns to 3 decimal places
rounded_df_orientation = df_orientation.round(3)

# Convert the DataFrame to a LaTeX table with controlled float format
orientaion_latex_table = rounded_df_orientation.to_latex(index=False, float_format="%.3f")

# Print the LaTeX table
print(orientaion_latex_table)

"""
print(df_pivot)
for i in enumerate(size):       
    selected_rows = orientation_grouped_df[orientation_grouped_df['size'] == size[i]]


    # Use the selected rows in your code

    for index, row in selected_rows.iterrows():
        # Do something with the row data
        current_row = {'size': row['size'],'color_and_size': orientation_grouped_df,'read_text': row['read_text'],'ground_truth_text': row['ground_truth_text'] }
        testing_df = pd.concat([testing_df, pd.DataFrame(current_row, index=[0])], ignore_index=True)

    for index, row in testing_df.iterrows():
        read_string = row["read_text"]
        ground_truth = row["ground_truth_text"]
            """

"""
for metric in metrics:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each model
    for i, model in enumerate(models):
        model_data = grouped_df[grouped_df['model'] == model]
        ax.bar(x + i*bar_width, model_data[metric], width=bar_width, label=model)

    # Fix the x-axes
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(masks)

    ax.set_xlabel('Masks')
    ax.set_ylabel(f'Average {metric.capitalize()}')
    ax.set_title(f'Average {metric.capitalize()} for each Mask-Model pair')
    ax.legend()

    plt.show()

# Round the numeric columns to 3 decimal places
rounded_grouped_df = grouped_df.round(3)

# Convert the DataFrame to a LaTeX table with controlled float format
latex_table = rounded_grouped_df.to_latex(index=False, float_format="%.3f")

# Print the LaTeX table
print(latex_table)





levensthein_distance("yellow")
levensthein_distance("brown")
levensthein_distance("blue")
levensthein_distance("green")
levensthein_distance("black")
levensthein_distance("red")
levensthein_distance("white")
levensthein_distance("orange")"""

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