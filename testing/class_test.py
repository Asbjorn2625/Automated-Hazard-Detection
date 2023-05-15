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

classifi = Classifier()
open('fulloutput.csv', 'w').close()
open('UN_nr.csv', 'w').close()
open('Hazard.csv', 'w').close()
open('Handel.csv', 'w').close()
open('cert.csv', 'w').close()

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

count = 1
row_nr = 1
count1 = 1
UN_data, Haz_data, hand_data, cert_data = [], [], [], []
data3, data6, data7 = "N/A","N/A","N/A"
UN_nr = None
UN = None
haz_test = True
hand_test = True
handel = False
TSU_data = []

#UN certificate



def UN_nr_test(count1, trans, row_nr, UN_data, data3, UN_nr, UN):
        res, text1 = classifi.classif_PS(trans)
        
        UN_data = UN_data + [text1]
        print(UN)
        if count1 == 1:
            UN_nr = check_list(row_nr, 2)
            print(UN_nr)
            
        if text1[-4:] == UN_nr:
            print("here")
            UN = 1
        if count1 == 4 and UN == 1:
            print("here to")
            data3 = "pass"
            UN = 0
        elif count1 == 4:
            print("somehow")
            data3 = "fail"
        #stores results from UN number reading as an csv
        if count1 == 4:
            UN_data = UN_data + ["ground truth:"] + [UN_nr]
            with open('UN_nr.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(UN_data)
            count1 = 0
            row_nr += 1
            UN_data = []
        return data3, UN_data, UN_nr, UN




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
        cert_list = check_list(row_nr, 0)
        if cert_list == ["Fibreboard Box"]:
            cert_list == ["4G"]
    
    rotated, text3 = classifi.classifi_UN(trans)
    trans_res = cv2.resize(rotated, [520, 520]) 
    #cv2.imshow("image", trans_res)
    #cv2.waitKey(1000)
    cert_data = cert_data + [text3]
    if count1 == 4:
            cert_data = cert_data + ["ground truth:"] + [cert_list]
            with open('cert.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(cert_data)
            cert_data = []
    
    #UN_nr test
    res, text1 = classifi.classif_PS(trans)
            
    UN_data = UN_data + [text1]
    print(UN)
    print(text1)
    if count1 == 1:
        UN_nr = check_list(row_nr, 2)
        print(UN_nr)
                
    if text1[-4:] == UN_nr:
        print("here")
        UN = 1
    if count1 == 4 and UN == 1:
        print("here to")
        data3 = "pass"
        UN = 0
    elif count1 == 4:
        print("somehow")
        data3 = "fail"
    #stores results from UN number reading as an csv
    if count1 == 4:
        UN_data = UN_data + ["ground truth:"] + [UN_nr]
        with open('UN_nr.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(UN_data)
        count1 = 0
        row_nr += 1
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
            
            UN_data = Haz_data + ["ground truth:"] + [haz_list]
            with open('Hazard.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(Haz_data)
            Haz_data = []


    #Handeling label test
    if hand_test == True:
        if count1 == 1:
            hand = check_list(row_nr, 6)
        TSU = False
        TSU, CAO, LIT, UNnr = classifi.classifi_Handeling(trans)
        TSU_data = TSU_data + [TSU]
        if True in TSU_data:
            if TSU_data == [True,False,True,False]:
                TSU_res = "pass"
            if TSU_data == [False, True, False, True]:
                TSU_res = "pass"
            else:
                TSU_res = "fail"
        else:
            TSU_res = ["N/A"]    
        
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
                with open('Handel.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(hand_data)
                hand_data = []
                if handel == True:
                    data7 = "pass"  
                else:
                    data7 = "fail"      
            


    
    
    #loop wrap up
    if count1 == 4:
        with open('fulloutput.csv', 'a', newline='') as f:
            full_data = ["UN_nr = " + data3] + ["Hazard =" + data6] + ["Handeling =" + data7] + ["TSU = " + TSU_res]
            writer = csv.writer(f)
            writer.writerow(full_data)
        count1 = 0
        row_nr += 1
        full_data = []
    count1 += 1
    count += 1
    cv2.destroyAllWindows()