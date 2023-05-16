import os 
import cv2
import numpy as np
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append('/workspaces/Automated-Hazard-Detection')
from src.Preprocess.prep import PreProcess
from src.Text_reader.ReaderClass import ReadText

def unpack_boxes(image,box, padding=10):
    x1, y1 = box[0]
    x2, y2 = box[1]
    x3, y3 = box[2]
    x4, y4 = box[3]
    xmin = min(x1, x2, x3, x4)
    xmax = max(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    ymax = max(y1, y2, y3, y4)

    # Padding is to make sure we get the entire text
    xmin = int(xmin) - padding
    xmax = int(xmax) + padding
    ymin = int(ymin) - padding
    ymax = int(ymax) + padding

    (h, w) = image.shape[:2]
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)
    
    cropped_image = image[ymin:ymax, xmin:xmax]
    return cropped_image


list = os.listdir(os.getcwd() + "/Dataset")

RT = ReadText()
PP = PreProcess()
columns = {"Filename":[], "Real text":[], "Predicted text":[]}

ground_truth = ["Flammable Liquid","Flammable Solid","Toxic Gas","Spontaneosly Combustible","Toxic","Oxidizing Agent","Corrosive","Explosive","Infectous Substance"]
length = len(ground_truth)*4
for items in ground_truth:
    for i in range(4):
        columns["Real text"].append(items.upper())

for filename in tqdm(list):
    text1=[]
    if filename.endswith("_MASK.png"):
        columns["Filename"].append(filename.replace("_MASK.png",""))
        mask_img=cv2.imread(os.getcwd()+ "/Dataset/" + filename, 0)
        mask_img = cv2.threshold(mask_img, 10, 255, cv2.THRESH_BINARY)[1]
        rgb_img=cv2.imread(os.getcwd()+ "/Dataset/" + filename.replace("_MASK",""))
        masked=cv2.bitwise_and(rgb_img, rgb_img, mask=mask_img)
        ROI=PP.segmentation_to_ROI(mask_img)
        for bounds in ROI:
            cropped = masked[bounds[1]:bounds[3], bounds[0]:bounds[2]]
            RT.findText(cropped)
            bounding = RT.findText(cropped)
            config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\ ./- --psm 7 --oem 3'
            for boxes in bounding:
                predtext= RT.readText(cropped, boxes,config=config)
                text1.append(predtext)
                
                            
        if len(text1) == 0:
            columns["Predicted text"].append("No text found")
        else:
            columns["Predicted text"].append(text1)
    else:
        continue



df = pd.DataFrame(columns)
print(df)