import sys
import os
sys.path.append('/workspaces/Automated-Hazard-Detection')

from src.Preprocess.prep import PreProcess
from src.Data_acquisition.Image_fetcher import ImageFetcher
from src.Classification.Classy import Classifier
from src.Segmentation.segmentation import Segmentation
import pandas as pd
import cv2
import numpy as np
from src.Text_reader.ReaderClass import ReadText

read = ReadText()
pp= PreProcess()
classy = Classifier(read, pp)
Imagefetch = ImageFetcher(os.getcwd() + "/Dataset")
imglib = Imagefetch._Orientation_test()

def format_degree(value):
        return f"{value}$^\circ$"


columns = {"Real angle":[], "label1 Pred angle": [],"label2 Pred angle": [],"Difference": []}

list = os.listdir(os.getcwd() + "/Dataset")
real_angles= []
pred1 = []
pred2 = []
for masks in list :
    mask = cv2.imread(os.getcwd()+ "/Dataset/" + masks)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
    
    image = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    resized = cv2.resize(image, (int(image.shape[1]*0.5),int(image.shape[0]*0.5)), interpolation = cv2.INTER_AREA)

    if "Explosive" in masks:
        if "+" in masks:
            real_angle = float(masks.split("+")[1].replace(".png",""))
            columns["Real angle"].append(format_degree(round(real_angle,2)))
            prediction= abs(classy.Orientation(image))
            pred1.append(prediction)
            print(prediction)
            cv2.imshow("Explosive+", resized)
            cv2.waitKey(0)
            columns["label1 Pred angle"].append(format_degree(round(prediction,2)))
            
        elif "-" in masks:
            real_angle = float(masks.split("-")[1].replace(".png",""))
            columns["Real angle"].append(format_degree(round(real_angle,2)))
            prediction= abs(classy.Orientation(image))
            pred1.append(prediction)
            print(prediction)
            cv2.imshow("Explosive-", resized)
            cv2.waitKey(0)
            columns["label1 Pred angle"].append(format_degree(round(prediction,2)))
        else:
            continue
    if "Oxidant" in masks:
        if "+" in masks:
            print(masks)
            real_angle = float(masks.split("+")[1].replace(".png",""))
            columns["Real angle"].append(format_degree(round(real_angle,2)))
            prediction= abs(classy.Orientation(image))
            pred2.append(prediction)
            print(prediction)
            cv2.imshow("Oxidant+", resized)
            cv2.waitKey(0)
            columns["label2 Pred angle"].append(format_degree(round(prediction,2)))
            
        elif "-" in masks:
            print(masks)
            real_angle = float(masks.split("-")[1].replace(".png",""))
            columns["Real angle"].append(format_degree(round(real_angle,2)))
            prediction= abs(classy.Orientation(image))
            pred2.append(prediction)
            print(prediction)
            cv2.imshow("Oxidant-", resized)
            cv2.waitKey(0)
            columns["label2 Pred angle"].append(format_degree(round(prediction,2)))
        else:
            continue
real = [value.replace("$^\circ$", "") for value in columns["Real angle"][:len(columns["Real angle"]) // 2]]


for i in range(len(real)):
    columns["Difference"].append(format_degree(round(abs(((float(real[i]) - pred1[i])+(float(real[i])-pred2[i])/2)),2)))
    
columns["Real angle"] = columns["Real angle"][:len(columns["Real angle"]) // 2]

df = pd.DataFrame(columns)
print(df.to_latex(index=False,escape=False))
