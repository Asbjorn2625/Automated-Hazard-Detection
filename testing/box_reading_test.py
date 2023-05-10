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





def display_depth_image(depth_image, title='Depth Image'):
    plt.imshow(depth_image, cmap=plt.cm.viridis)
    plt.title(title)
    plt.axis('off')
    plt.show()
    

        
        

pp = PreProcess()

read = ReadText()

#segment = Segmentation()

#imma_pre = PreProcess()

i = 0

k = 0

page_count = 1



New_set = False

# Create list of image filenames
rgb_images = [f'./testing/Final_reading_test/{img}' for img in os.listdir("./testing/Final_reading_test/") if img.startswith("rgb_image")]

csv_file_path = "/workspaces/Automated-Hazard-Detection/testing/color_test.csv"


csv_file_path_output = "/workspaces/Automated-Hazard-Detection/testing/reading_test_output.csv"

# Create a list to store the label strings for each image
label_list = []

label_data_from_csv = []

#with open(csv_file_path, newline='') as csvfile:
#    reader = csv.reader(csvfile, delimiter=',')
#    for row in reader:
#        label_data_from_csv.append(row)
        
#print(label_data_from_csv)        


# read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# convert the DataFrame to a dictionary
my_dict = df.to_dict('list')

#print the dictionary
#print(my_dict)  

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
    



# Loop through the images
for image_path in rgb_images:
    
    
    if i == 11:
        k = k + 1
        i = 0
        page_count += 1
    
    
    original_img = cv2.imread(image_path)
    
    depth = np.fromfile(image_path.replace("rgb_image", "depth_image").replace("png", "raw"), dtype=np.uint16)
    # Reconstruct the depth map
    depth = depth.reshape(int(1080), int(1920))
     
    # undistort the image
    original_img = pp.undistort_images(original_img)
    
    depth = pp.undistort_images(depth)
    
    width, length = pp.get_pixelsize(depth)
    
    
    depth_blurred = cv2.medianBlur(depth, 5)
    
    trans_img, homography = pp.retrieve_transformed_plane(original_img, depth_blurred)

    
    
    
    
    #preds =  segment.locateHazard(trans_img)
    
    #Roi = pp.segmentation_to_ROI(preds)
    
    #for bounds in Roi:
    #        cropped = preds[bounds[1]:bounds[3], bounds[0]:bounds[2]]
            
    #cv2.imshow("crop", cropped) 
    #cv2.waitKey(0)
    


            

    box_text= read.findText(trans_img)
    
    resized_depth_img = cv2.resize(trans_img, (960, 540))
    
    resized_img = cv2.resize(original_img, (960, 540))
    
    current_orientation = [0,170,160,150,140,135,10,20,30,40,45] 
    
    image_filename = os.path.basename(image_path)

    # Remove the file extension from the filename
    image_filename_without_extension = os.path.splitext(image_filename)[0]
    
    
    size = [5.5, 5.5, 5.9, 5.9, 6.1, 6.1, 6.9, 6.9, 7.1, 7.1, 7.9, 7.9]
    

    # Set the font properties
    font = {'family': 'sans-serif',
            'color':  'black',
            'weight': 'normal',
            'size': 12}
    
    
    
    #plt.figure(figsize=(10, 5))  # Set the figure size to 10x10 inches
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
        text = read.readText(trans_img, box, False)
        print("text was found: " , text)
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
        
        print(smallest_point, biggest_point)
        print(positions)

    
        
        for pos, (x, y) in enumerate(positions):
            if (smallest_point[0] < x)  and (biggest_point[0] > x) and  (smallest_point[1] < y)  and (biggest_point[1] > y):
                
                #print("New_picture: ", picturename, text, current_set_df.loc[pos, "color"], orientation_value, size_value)
                the_color = current_set_df.loc[pos, "color"]
                the_color = the_color.replace("  ","")
                
                
                print(the_color)
                
                if text_on_box == " 4G/Y30/S/22/D/BAM": #remember to change 0 to D
                    print("I read good")
                if the_color == "  yellow":
                    print("im yellow")    
                
                #label = f" {picturename},  {size_value},  {x}, {y},  {orientation_value}, 4GY30S22DBAM, {the_color}"
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
            
    #csv_output_string = f"{image_filename_without_extension},{size_value},{orientation_value},{the_color},{text_on_box},4GY30S22DBAM "

    #print(output_list)          
    #print(csv_output_string)      
   

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



                

                

        



        #label = f"New_image: , {picturename},  {size_value},  {real_coordinates},  {orientation_value}, {text_on_box};"
        #label_list.append(label)
        #print(label)
        
        
        


    
    

      


    

# Set plot title and axis labels
#ax.set_title("Image Data")
#ax.set_xlabel("Size")
#ax.set_ylabel("Orientation")

 







  
     
    #cv2.imshow("img", resized_depth_img)
    #cv2.imshow("img1", resized_img)
    #for box in box_text:
        
       
    #    text = read.readText(trans_img, box, False, True)
    #    print("text was found: " , text)
        
        
    #cv2.imshow("rgb", trans_img)

    
    #cv2.waitKey(0)

"""
# define color ranges for each category
brown_range = np.array([[125, 0, 120], [180, 20, 150]])
other_brown_range = np.array([[0, 20, 120], [10, 50, 150]])
red_range = np.array([[0, 170, 100], [5, 255, 150]])
yellow_range = np.array([[20, 100, 100], [30, 200, 200]])
green_range = np.array([[45, 140, 40], [90, 255, 80]])
blue_range = np.array([[85, 100, 100], [130, 255, 170]])
orange_range = np.array([[5, 180, 130], [10, 220, 160]])
black_range = np.array([[0, 0, 0], [180, 255, 30]])
white_range = np.array([[0, 0, 200], [180, 50, 255]])


# convert image to HSV color space
hsv = cv2.cvtColor(trans_img, cv2.COLOR_BGR2HSV)

# create a window and set mouse callback function to get HSV values
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", get_hsv_values, param=hsv)

cv2.imshow("hsv", hsv)



# apply color filters to isolate colors
brown_mask = cv2.inRange(hsv, brown_range[0], brown_range[1])
other_brown_mask = cv2.inRange(hsv, other_brown_range[0], other_brown_range[1])
red_mask = cv2.inRange(hsv, red_range[0], red_range[1])
yellow_mask = cv2.inRange(hsv, yellow_range[0], yellow_range[1])
green_mask = cv2.inRange(hsv, green_range[0], green_range[1])
blue_mask = cv2.inRange(hsv, blue_range[0], blue_range[1])
orange_mask = cv2.inRange(hsv, orange_range[0], orange_range[1])
black_mask = cv2.inRange(hsv, black_range[0], black_range[1])
white_mask = cv2.inRange(hsv, white_range[0], white_range[1])

# apply masks to original image to show only pixels in each color category
brown_img = cv2.bitwise_and(trans_img, trans_img, mask=brown_mask)
other_brown_img = cv2.bitwise_and(trans_img, trans_img, mask=other_brown_mask)
red_img = cv2.bitwise_and(trans_img, trans_img, mask=red_mask)
yellow_img = cv2.bitwise_and(trans_img, trans_img, mask=yellow_mask)
green_img = cv2.bitwise_and(trans_img, trans_img, mask=green_mask)
blue_img = cv2.bitwise_and(trans_img, trans_img, mask=blue_mask)
orange_img = cv2.bitwise_and(trans_img, trans_img, mask=orange_mask)
black_img = cv2.bitwise_and(trans_img, trans_img, mask=black_mask)
white_img = cv2.bitwise_and(trans_img, trans_img, mask=white_mask)

# display the images
cv2.imshow("Brown", brown_img)
cv2.imshow("other_Brown", other_brown_img)
cv2.imshow("Red", red_img)
cv2.imshow("Yellow", yellow_img)
cv2.imshow("Green", green_img)
cv2.imshow("Blue", blue_img)
cv2.imshow("Orange", orange_img)
cv2.imshow("Black", black_img)
cv2.imshow("White", white_img)

# wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
    
       
        
