import sys
import numpy as np
import cv2
import csv
sys.path.append("/workspaces/Automated-Hazard-Detection")
from src.Text_reader.ReaderClass import ReadText
from src.Preprocess.prep import PreProcess
from src.Package.DGcheck import Package
#open('fulloutput1.csv', 'w').close()
#open('mask_truth.csv', 'w').close()
#open('cert_text.csv', 'w').close()
#open('PS.csv', 'w').close()
#open('CAO.csv', 'w').close()
#open('TSU.csv', 'w').close()
#open('Haz.csv', 'w').close()
reader = ReadText()
PreProd = PreProcess()
pack = Package(reader, PreProd)
count = 96
row_nr = 25
image_nr = 1
while count <= 232:
    print(count)
    print(row_nr)
    if count < 10:
        img = cv2.imread("images/Testing_data/rgb_image_000" + str(count) + ".png")
        depth = np.fromfile("images/Testing_data/depth_image_000" + str(count) +".raw", dtype=np.uint16)
    elif count < 100:
        img = cv2.imread("images/Testing_data/rgb_image_00" + str(count) + ".png")
        depth = np.fromfile("images/Testing_data/depth_image_00" + str(count) +".raw", dtype=np.uint16)
    elif count <= 128:
        img = cv2.imread("images/Testing_data/rgb_image_0" + str(count) + ".png")
        depth = np.fromfile("images/Testing_data/depth_image_0" + str(count) +".raw", dtype=np.uint16)
    elif count <= 137:
        new_count = count - 128
        img = cv2.imread("images/Bad_set/rgb_image_000" + str(new_count) + ".png")
        depth = np.fromfile("images/Bad_set/depth_image_000" + str(new_count) +".raw", dtype=np.uint16)
    elif count < 228:
        new_count = count - 128
        img = cv2.imread("images/Bad_set/rgb_image_00" + str(new_count) + ".png")
        depth = np.fromfile("images/Bad_set/depth_image_00" + str(new_count) +".raw", dtype=np.uint16)
    elif count >= 228:
        new_count = count - 128
        img = cv2.imread("images/Bad_set/rgb_image_0" + str(new_count) + ".png")
        depth = np.fromfile("images/Bad_set/depth_image_0" + str(new_count) +".raw", dtype=np.uint16)
    depth = depth.reshape(int(1080), int(1920))

    if row_nr not in [32, 39, 43, 51, 71]:
        pack.main(img, depth, row_nr)
    
    if image_nr == 4:
        if row_nr in [32, 39, 43, 51, 71]:
            with open('fulloutput1.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow("bad data")
        else: pack.log_full()
        image_nr = 1
        row_nr += 1
        count += 1
    else:
        count += 1
        image_nr += 1
    
    







 
