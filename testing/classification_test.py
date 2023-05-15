import cv2
import numpy as np
import sys
import json
import csv
sys.path.append("/workspaces/Automated-Hazard-Detection")
from src.Text_reader.ReaderClass import ReadText
from src.Preprocess.prep import PreProcess
from testing.classifier import Classifier
from src.Data_acquisition.Image_fetcher import ImageFetcher

classifi = Classifier()

def check_list(row_nr, item):
    with open('Dangerous_goods_list_for_testing.csv', 'r') as file:
        reader = csv.reader(file)
        counter = 0
        for row in reader:
            if counter == row_nr:
                listed = row[item]
                break
            counter += 1
    return listed
handel = False
UN = False
Haz = False
count1 = 1
count = 1
row_nr = 1
while(count <= 128):
    if count < 10:
        img = cv2.imread("images/rgb_image_000" + str(count) + ".png")
        depth = np.fromfile("images/depth_image_000" + str(count) +".raw", dtype=np.uint16)
    elif count < 100:
        img = cv2.imread("images/rgb_image_00" + str(count) + ".png")
        depth = np.fromfile("images/depth_image_00" + str(count) +".raw", dtype=np.uint16)
    else:
        img = cv2.imread("images/rgb_image_0" + str(count) + ".png")
        depth = np.fromfile("images/depth_image_0" + str(count) +".raw", dtype=np.uint16)
    
    trans = classifi.image_prep(img , depth)
    #Package type
    #check_list(row_nr, 0)
    
    if count1 == 4:
        print("package type = N/A")
        data1 = ["N/A"]
        
    # Minimum testing
    #check_list(row_nr, 1)
   
    if count1 == 4: 
        print ("Minimum testing = N/A")
        data2 = ["N/A"]
    # UN number
    
    res, text1 = classifi.classif_PS(trans)
    print(text1)
    if text1[-4:] == check_list(row_nr, 2):
        UN = True
    if count1 == 4 and UN == True:
        print("UN number = pass")
        data3 = ["pass"]
        UN = False
    elif count1 == 4:
        print("UN number = fail")
        data3 = ["fail"]
    
    
    # Proper shipping name
    
    check_list(row_nr, 3)
    if count1 == 4:
        print("propper shipping name = N/A")
        data4 = ["N/A"]
    
    # Name and adress
    check_list(row_nr, 4)
    if count1 == 4:
        print("Name and adress = N/A")
        data5 = ["N/A"]
    
    # Hazard Labels
    res, text2  = classifi.classifi_hazard(img)
    print(text2)
    if check_list(row_nr, 5) in text2:
        Haz = True
    if Haz == True and count1 == 4:
        print("Hazard = pass")
        data6 = ["pass"]
        Haz = False
    elif count1 == 4:
        print("Hazard = failed")
        data6 = ["fail"]
    # Handling Labels
    hand = check_list(row_nr, 6)
    TSU, CAO, LIT, UNnr = classifi.classifi_Handeling(trans)
    print(UN)
    if hand == "Lithium bat" and LIT == True and CAO != True:
        if UNnr == text2:
            handel = True
    if hand == "CARGO CRAFT only" and LIT != True and CAO == True:
        handel = True 
    if hand == "NONE" and LIT != True and CAO != True:
        handel = True         
    if count1 == 4 and handel == True:
        print("Handling = pass")
        data7 = ["pass"]
        handel = False
    elif count1 == 4:
        print("Handeling = fail")
        data7 = ["fail"]
    
    if count1 == 4:
        data = data1 + data2 + data3 + data4 + data5 + data6 + data7
        with open('output.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
        row_nr += 1
        count1 = 0
    count += 1
    print(count1)
    count1 += 1
    
    print("loop")

#classifier = Classifier()
#pp = PreProcess()
















