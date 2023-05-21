import sys
import numpy as np
import cv2
sys.path.append("/workspaces/Automated-Hazard-Detection")
from src.Text_reader.ReaderClass import ReadText
from src.Preprocess.prep import PreProcess
from src.Package.DGcheck import Package
open('fulloutput.csv', 'w').close()
open('mask_truth.csv', 'w').close()
open('cert_text.csv', 'w').close()
open('PS.csv', 'w').close()
open('CAO.csv', 'w').close()
open('TSU.csv', 'w').close()
open('Haz.csv', 'w').close()
reader = ReadText()
PreProd = PreProcess()
pack = Package(reader, PreProd)
count = 1
row_nr = 1
image_nr = 1
while count <= 128:
    if count < 10:
        img = cv2.imread("images/rgb_image_000" + str(count) + ".png")
        depth = np.fromfile("images/depth_image_000" + str(count) +".raw", dtype=np.uint16)
    elif count < 100:
        img = cv2.imread("images/rgb_image_00" + str(count) + ".png")
        depth = np.fromfile("images/depth_image_00" + str(count) +".raw", dtype=np.uint16)
    else:
        img = cv2.imread("images/rgb_image_0" + str(count) + ".png")
        depth = np.fromfile("images/depth_image_0" + str(count) +".raw", dtype=np.uint16)
    
    depth = depth.reshape(int(1080), int(1920))

    pack.main(img, depth, row_nr)
    
    if image_nr == 4:
        pack.log_full()
        image_nr = 1
        row_nr += 1
    print(count)
    count += 1
    image_nr += 1
    
    







 
