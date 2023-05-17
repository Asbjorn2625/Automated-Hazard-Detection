import cv2
import os 
import numpy as np
import sys
import pandas as pd

sys.path.append('/workspaces/Automated-Hazard-Detection')
from src.Preprocess.prep import PreProcess
from src.Classification.Classy import classifier

pp = PreProcess()
classy = classifier()
def format_degree(value):
        return f"{value}$^\circ$"

def detect__package_orientation(image,Display=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    ret, thresh = cv2.threshold(image,20,255, cv2.THRESH_BINARY)
    cnt, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    largest_extent = float("-inf")
    for c in cnt:
        #measure the extent of the contour
        cnt_area = cv2.contourArea(c)
        # find the rotated rectangle that encloses the contour
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = [[int(point[0]), int(point[1])] for point in box]
    
        box_area = cv2.contourArea(np.array(box))
        extent = cnt_area/box_area
        
        if extent > largest_extent:
            largest_extent = extent
            best_box = box
            box = np.array(box)
            # get the longest side of the box
            long_side_index = np.argmax([np.linalg.norm(box[i] - box[(i+1)%4]) for i in range(4)])
            
            # Get the angel of the longest side
            dx = box[long_side_index][0] - box[(long_side_index+1)%4][0]
            dy = box[long_side_index][1] - box[(long_side_index+1)%4][1]
            angle = np.arctan2(dy, dx) * 180 / np.pi
            #angle = 90-angle
            if angle < 0:
                angle += 180
            angle = (angle)%180
            if angle > 90:
                angle = (angle)-180
            angle = abs(angle)
           
            
    cv2.drawContours(image, [np.array(best_box)], 0, (0, 0, 255), 2)
    # Resize image
    image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    # Display the image
    if Display:
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    return round(angle,2)

list = os.listdir(os.getcwd() + "/Dataset")

columns = { "Real angle":[], "Predicted angle": [],"difference": []}
real_angles = []

for i in range(0,100,10):
    real_angles.append(i)
for i in range(10,100,10):
    real_angles.append(i)


for i,img in enumerate(list):
    image = cv2.imread(os.getcwd() + "/Dataset/" + img)
    real = real_angles[i]
    prediction = detect_orientation(image)
    differnece = round(abs(real-prediction),2)
    columns["Real angle"].append(format_degree(real))
    columns["Predicted angle"].append(format_degree(detect_orientation(image)))
    columns["difference"].append("\\textbf{"+format_degree(differnece)+"}")
df = pd.DataFrame(columns)
print(df.to_latex(index=False, float_format="%.2f"))
    

