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
from src.Segmentation.segmentation import Segmentation
import re

classifi = Classifier()
open('fulloutput.csv', 'w').close()
open('UN_nr.csv', 'w').close()
open('Hazard.csv', 'w').close()
open('Handel.csv', 'w').close()
open('cert.csv', 'w').close()
open('TSU.csv', 'w').close()

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

cert_XYZ = []
count = 1
row_nr = 1
count1 = 1
UN_data, Haz_data, hand_data, cert_data = [], [], [], []
data1, data3, data6, data7 = "N/A","N/A","N/A", "N/A"
UN_nr = None
UN = None
haz_test = True
hand_test = True
handel = True
cert_pak = False
cert_class = False
TSU_data = []
cert_XYZ_true = False

#UN certificate


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
    

    #UN certificate
    if count1 == 1:
        cert_package = check_list(row_nr, 0)
        cert_test = check_list(row_nr, 1)
        if "Fibreboard Box" == cert_package:
            cert_package = "4G"
        
    rotated, text3 = classifi.classifi_UN(trans)
    cert_data = cert_data + [text3]
    
    if count1 == 4:
        list_of_strings = [' '.join(sublist) for sublist in cert_data]
        print(list_of_strings)
        for string in list_of_strings:
            if cert_package in string:
                cert_pak = True
            x_values = re.findall('X\d+(\.\d+)?', string)
            print(x_values)
            y_values = re.findall('Y\d+(\.\d+)?', string)
            print(y_values)
            z_values = re.findall('Z\d+(\.\d+)?', string)
            print(z_values)
            if "X" in cert_test and len(x_values) > 0: 
                cert_XYZ_true = True
            if "Y" in cert_test and len(y_values)> 0:
                cert_XYZ_true = True
            if "Z" in cert_test and len(z_values) > 0:
                cert_XYZ_true = True            
                        
            
        if cert_XYZ_true == True:
            data2 = "pass"
            cert_XYZ_true = False
        else:
            data2 = "fail"
            
        if cert_pak == True:
            data1 = "pass"
            cert_pak = False
        else:
            data1 = "fail"
        
        cert_data = cert_data + ["ground truth package:"] + [cert_package] + ["ground truth: "] + [cert_test]
        with open('cert.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(cert_data)
        cert_data = []
    
    #UN_nr test
    res, text1 = classifi.classif_PS(trans)
            
    UN_data = UN_data + [text1]
    
    if count1 == 1:
        UN_nr = check_list(row_nr, 2)
                
    if text1[-4:] == UN_nr:
        UN = 1
    if count1 == 4 and UN == 1:
        data3 = "pass"
        UN = 0
    elif count1 == 4:
        data3 = "fail"
    #stores results from UN number reading as an csv
    if count1 == 4:
        UN_data = UN_data + ["ground truth:"] + [UN_nr]
        with open('UN_nr.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(UN_data)
        UN_data = []

    
    #Hazard_label test
    if haz_test == True:
        res, text2  = classifi.classifi_hazard(trans)
        if count1 == 1:
            haz_list = check_list(row_nr, 5)
        Haz_data = Haz_data + [text2]
        
        if count1 == 4:
            haz_split = haz_list.split()
            # Flatten Haz_data
            flat_Haz_data = [item for sublist in Haz_data for item in sublist]
            # Make it into a set
            haz1_set = set(flat_Haz_data)

            if "or" in haz_list:
                while "or" in haz_split:  # Remove all occurrences of "or"
                    haz_split.remove("or")
                haz2_set = set(haz_split)  # Then convert the list to a set
                if len(haz1_set.intersection(haz2_set)) > 0:
                    data6 = "pass" 
                else:
                    data6 = "fail"   

            elif "," in haz_list:
                while "," in haz_split:  # Remove all occurrences of ","
                    haz_split.remove(",")
                haz2_set = set(haz_split)  # Then convert the list to a set
                if len(haz2_set.intersection(haz1_set)) == len(haz2_set):
                    data6 = "pass"
                else:
                    data6 = "fail"
            
            elif haz_list in Haz_data:
                data6 = "pass"
                
            else:
                data6 = "fail"
            
            Haz_data = Haz_data + ["ground truth:"] + [haz_list]
            with open('Hazard.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(Haz_data)
            Haz_data = []


    #Handeling label test
    if hand_test == True:
        if count1 == 1:
            hand = check_list(row_nr, 6)
        LIT, CAO, TSU,  UNnr = classifi.classifi_Handeling(trans)
        TSU_data = TSU_data + [TSU]
        if True in TSU_data:
            if TSU_data == [True,False,True,False]:
                TSU_res = "pass"
            elif TSU_data == [False, True, False, True]:
                TSU_res = "pass"
            elif count1 == 4:
                TSU_res = "fail"
        elif count1 == 4:
            TSU_res = "N/A" 
            
        if count1 == 4:
            with open('TSU.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(TSU_data)
            TSU_data = [] 
        
        if "Lithium bat" in hand:
            hand_data = hand_data + [UNnr]
            if LIT == True:            
                if UNnr == Haz_data:
                    handel = True
        else:
            hand_data = hand_data + ["N/A"]
            
        if "CARGO CRAFT only" in hand and CAO == True:
            handel = True 
        if hand == "NONE" and LIT != True and CAO != True:
            handel = True    
                 
        if count1 == 4:
                hand_data = hand_data + ["ground truth:"] + [hand]
                if handel == True:
                    data7 = "pass" 
                    handel = False 
                else:
                    data7 = "fail"  
                     
                with open('Handel.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(hand_data)
                hand_data = []
                   
    
    


    
    
    #loop wrap up
    if count1 == 4:
        with open('fulloutput.csv', 'a', newline='') as f:
            full_data = ["Package type = " + data1] +["weight test = " + data2] + ["UN_nr = " + data3] + ["Hazard =" + data6] + ["Handeling =" + data7] + ["TSU = " + TSU_res]
            writer = csv.writer(f)
            writer.writerow(full_data)
            print("full data saved", full_data)
        count1 = 0
        row_nr += 1
        full_data = []
    count1 += 1
    count += 1
    cv2.destroyAllWindows()