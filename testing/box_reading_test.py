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
from tqdm import tqdm
import re
from itertools import islice



def display_depth_image(depth_image, title='Depth Image'):
    plt.imshow(depth_image, cmap=plt.cm.viridis)
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def transform_list(image_list, image_path, pre_processer):
    # Create transformed list folder
    folder_path = os.path.join(image_path, "transformed_images")
    os.makedirs(folder_path, exist_ok=True)
    
    # Create csv list
    csv_list = []
    for image_path in tqdm(image_list):
        original_img = cv2.imread(image_path)
    
        depth = np.fromfile(image_path.replace("rgb_image", "depth_image").replace("png", "raw"), dtype=np.uint16)
        # Reconstruct the depth map
        depth = depth.reshape(int(1080), int(1920))
        
        # undistort the image
        original_img = pre_processer.undistort_images(original_img)
        
        depth = pre_processer.undistort_images(depth)
        
        trans_img, homography = pre_processer.retrieve_transformed_plane(original_img, depth)
        # Save path
        save_path = os.path.join(folder_path, image_path.split("/")[-1])
        # Save the image into the folder
        cv2.imwrite(save_path, trans_img)
        
        # Save the transformed image and homography into the csv list
        csv_list.append({"image_path": image_path, "transformed_image_path": save_path, "homography": homography})
    # save the csv list
    csv_path = os.path.join(folder_path, "transformed_images.csv")
    with open(csv_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=csv_list[0].keys())
        writer.writeheader()
        writer.writerows(csv_list)
    return
        

pp = PreProcess()

read = ReadText()

i = 0

k = 0

page_count = 1



NEW_SET = False

#Folder path
current_file_path = os.path.abspath(__file__)
# Get the folder containing the current file
current_folder = os.path.dirname(current_file_path)

image_folder = os.path.join(current_folder, "Final_reading_test")

# Image path

# Create list of image filenames
rgb_images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.startswith("rgb_image")]

csv_file_path = os.path.join(current_folder, "color_test.csv")


csv_file_path_output = os.path.join(current_folder, "reading_test_output.csv")

# Create a list to store the label strings for each image
label_list = []
label_data_from_csv = []       


# read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# convert the DataFrame to a dictionary
my_dict = df.to_dict('list')

output_list = []  


succesful_black = 0
succesful_red = 0
succesful_yellow = 0
succesful_brown = 0
succesful_white = 0
succesful_blue = 0
succesful_orange = 0
succesful_green = 0

def calc_success_rate(color, color_name):
    success_rate = color/66
    print(color_name, success_rate)
    return success_rate    

if NEW_SET:
    transform_list(rgb_images, image_folder, pp)

# Load the csv file
csv_file_path = os.path.join(image_folder, "transformed_images/transformed_images.csv")
df_imges = pd.read_csv(csv_file_path)

# Loop through the images
for path, transform_path, homography in tqdm(islice(zip(df_imges["image_path"], df_imges["transformed_image_path"], df_imges["homography"]), 0, None), "Reading images"):

    # Load the images
    original_img = cv2.imread(path)
    trans_img = cv2.imread(transform_path)
    

    # Remove brackets and newlines, and replace multiple spaces with single spaces
    homography = re.sub('\s+', ' ', homography.replace('[', '').replace(']', '').replace('\n', ''))

    # Use np.fromstring to convert the string to a 1D array
    homography = np.fromstring(homography, sep=' ')
    # Fix the homography
    homography = homography.reshape(-1, 3)

    if i == 11:
        k = k + 1
        i = 0
        page_count += 1
    

    
    box_text= read.findText(trans_img)
    
    resized_depth_img = cv2.resize(trans_img, (960, 540))
    
    resized_img = cv2.resize(original_img, (960, 540))
    
    current_orientation = [0,170,160,150,140,135,10,20,30,40,45] 
    
    image_filename = os.path.basename(path)

    # Remove the file extension from the filename
    image_filename_without_extension = os.path.splitext(image_filename)[0]
    
    
    size = [5.5, 5.5, 5.9, 5.9, 6.1, 6.1, 6.9, 6.9, 7.1, 7.1, 7.9, 7.9]
    

    # Set the font properties
    font = {'family': 'sans-serif',
            'color':  'black',
            'weight': 'normal',
            'size': 12}
    
    
    #plt.figure(figsize=(7, 5))  # Set the figure size to 10x10 inches
    #plt.title(image_filename_without_extension)
    #plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    #plt.show()
    
    current_set_df = pd.DataFrame(columns=['picturename', 'size', 'x', 'y', 'orientation', 'text', 'color '])
    
    
    
    #print(image_filename_without_extension)
    #print(df['picturename'] == image_filename_without_extension)
    #print(df.loc[df['picturename'] == image_filename_without_extension])
    
    # Select the rows with the current picture name
    selected_rows = df.loc[df['picturename'] == image_filename_without_extension]
    
    #print(selected_rows)

    # Use the selected rows in your code
    positions = []
    for index, row in selected_rows.iterrows():
     # Do something with the row data
        current_row = {'picturename': row['picturename'], 'size': row['size'], 'x': row['x'], 'y': row['y'],'orientation': row['orientation'],'text': row['text'], 'color': row['color']}
        positions.append([int(row['x']),int(row['y'])])
        current_set_df = pd.concat([current_set_df, pd.DataFrame(current_row, index=[0])], ignore_index=True)
  
    for count, box in enumerate(box_text):
        
        center = np.sum(box[:,0])/len(box),np.sum(box[:,1])/len(box)
        text = read.readText(trans_img, box, display=False)
        #print("\n text was found: " , text)
        # Define the values you want to write to the CSV file
        picturename = image_filename_without_extension
        size_value = size[k]
        coordinates = center
        #real_coordinates = pp.transformed_to_original_pixel(center, homography)
        orientation_value = current_orientation[i]
        text_on_box = text
        real_box = list(map(lambda x: list(pp.transformed_to_original_pixel(trans_img,x, homography)), box))
        

        # Extract the bounding area
        x1, y1 = real_box[0]
        x2, y2 = real_box[1]
        x3, y3 = real_box[2]
        x4, y4 = real_box[3]
        
        xmin = min(x1, x2, x3, x4)
        xmax = max(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        ymax = max(y1, y2, y3, y4)
        
        smallest_point = [xmin, ymin]
        
        biggest_point = [xmax, ymax]


        for pos, (x, y) in enumerate(positions):
            if (smallest_point[0] < x)  and (biggest_point[0] > x) and  (smallest_point[1] < y)  and (biggest_point[1] > y):
                
                the_color = current_set_df.loc[pos, "color"]
                the_color = the_color.replace("  ","")
                
                if text_on_box == "4G/Y30/S/22/D/BAM":
                    if the_color == "yellow":
                        succesful_yellow += 1
                    if the_color == "brown":
                        succesful_brown += 1
                    if the_color == "red":
                        succesful_red += 1
                    if  the_color == "blue":
                         succesful_blue += 1
                    if the_color == "green":
                        succesful_green += 1
                    if the_color == "black":
                        succesful_black += 1
                    if  the_color == "white":
                        succesful_white += 1
                    if  the_color == "orange":
                        succesful_orange += 1                    

                csv_output_string = f"{image_filename_without_extension},{size_value},{orientation_value},{the_color},{text_on_box}, 4G/Y30/S/22/D/BAM "
                current_set_df = current_set_df[(current_set_df['color'] != current_set_df.loc[pos, "color"])]
                
                output_list.append(csv_output_string)
            
    for index, row in current_set_df.iterrows():
        the_color = row["color"]
        the_color = the_color.replace("  ", "")

        csv_output_string = f"{image_filename_without_extension},{size_value},{orientation_value},{the_color},nan, 4G/Y30/S/22/D/BAM"
        output_list.append(csv_output_string)
            

    i = i + 1 
   
    # Open the CSV file in append mode
with open(csv_file_path_output, mode='w', newline='') as file:
    file.write("picturename,size,orientation,background_color,read_text,ground_truth_text\n")    
    for string in output_list:
        file.write(string + "\n")
        

red_rate = calc_success_rate(succesful_red, "red")   
green_rate = calc_success_rate(succesful_green, "green")  
blue_rate = calc_success_rate(succesful_blue, "blue")       
orange_rate = calc_success_rate(succesful_orange, "orange")  
black_rate = calc_success_rate(succesful_black, "black")  
white_rate = calc_success_rate(succesful_white, "white")  
brown_rate = calc_success_rate(succesful_brown, "brown") 
yellow_rate = calc_success_rate(succesful_yellow, "yellow")   

print("CSV file updated successfully.")
