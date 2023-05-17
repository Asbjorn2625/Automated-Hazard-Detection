import os 
import cv2
import numpy as np
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import Levenshtein

sys.path.append('/workspaces/Automated-Hazard-Detection')
from src.Preprocess.prep import PreProcess
from src.Text_reader.ReaderClass import ReadText


def find_closest_match(input_list, DGRlist):
    scores = []
    for key, value in DGRlist.items():
        similarity = 0
        for word in value:
            for input_word in input_list:
                if ((word != "" and input_word != "") and ((not input_word.replace(".","").isdigit()) and (not word.replace(".","").isdigit()))) and len(input_word) > 2:
                    if Levenshtein.distance(word, input_word) <= 2:
                        """
                        print("_______________________________________________________________")
                        print("Key: ", key)
                        print("DGR word: ", word, "Input word: ", input_word)
                        print("Levenshtein distance: ", Levenshtein.distance(word, input_word))
                        print("_______________________________________________________________")
                        """
                        similarity += 1
                        
                elif (word != "" and input_word != "") and (input_word.replace(".","").isdigit() and word.replace(".","").isdigit()):
                    if word == input_word:
                        """ 
                        print("word: ", word, "input_word: ", input_word)
                        """
                        similarity += 1
        if similarity > 0:
            scores.append((key, similarity))
           

    if scores:
        max_score = max(scores, key=lambda x: x[1])
        max_score_value = max_score[1]
        max_score_indices = [i for i, score in enumerate(scores) if score[1] == max_score_value]
        keys_with_highest_scores = [scores[i][0] for i in max_score_indices]

        return scores, keys_with_highest_scores
    else:
        return scores, ""

list = os.listdir(os.getcwd() + "/Dataset")
RT = ReadText()
PP = PreProcess()
columns = {"Real text":[],"prediction":[]}

test = {"Explosives": ["EXPLOSIVE","1", "1.1","1.2","1.3","1.4","1.5","1.6"],
        "Flammable Gas": ["FLAMMABLE","GAS","FLAMMABLE GAS","2","2.1"],
        "Non-flammable gas": ["NON-FLAMMABLE","GAS","2","2.1"],
        "Toxic Gas": ["TOXIC","GAS","TOXIC GAS","2","2.3"],
        "Flammable Liquid": ["Flammable Liquid","Flammable", "Liquid","3"],
        "Flammable Solid": ["FLAMMABLE", "SOLID", "FLAMMABLE SOLID","4","4.1"],
        "Spontaneosly Combustible": ["SPONTANEOUSLY COMBUSTIBLE","SPONTANEOUSLY", "COMBUSTIBLE","4","4.2"],
        "Dangerous When Wet": ["DANGEROUS","WHEN", "WET","DANGEROUS WHEN WET", "4","4.3"],
        "Oxidizing Agent": ["OXIDISING","AGENT","OXIDISING AGENT","5","5.1"],
        "Organic Peroxides": ["ORGANIC", "PEROXIDES","ORGANIC PEROXIDES","5","5.2"],
        "Toxic": ["TOXIC","6","6.1"],
        "Infectous Substance": ["INFECTIOUS","SUBSTANCE","INFECTIOUS SUBSTANCE", "6","6.2"],
        "Corrosive": ["CORROSIVE","8"],
        "Miscellanous": ["MISCELLANEOUS","9"],
        "Lithium Batteries": ["LITHIUM BATTERIES", "9"]}

ground_truth = ["Flammable Liquid","Flammable Solid","Toxic Gas","Spontaneosly Combustible","Toxic","Oxidizing Agent","Corrosive","Explosives","Infectous Substance"]
length = len(ground_truth)*4
for items in ground_truth:
    for i in range(4):
        columns["Real text"].append(items)

for filename in tqdm(list):
    text1=[]
    if filename.endswith("_MASK.png"):
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
            columns["prediction"].append(find_closest_match(text1,test)[1])
    else:
        continue
for i, value in enumerate(columns["prediction"]):
    if value != "":
        if len(value) >= 2:
            columns["prediction"][i] = ""
        else:
            columns["prediction"][i] = value[0]
            
cm=confusion_matrix(columns["Real text"],columns["prediction"],labels=ground_truth)
print(cm)


class_labels= ["FL","FS","TG","SC","TX","OA","CO","EX","IS"]
# Plot confusion matrix
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.grid(True) 
plt.show()

df = pd.DataFrame(columns)
print(df.to_html(index=False))