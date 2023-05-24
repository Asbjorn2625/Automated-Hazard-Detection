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
from src.Segmentation.segmentation import Segmentation
from src.Text_reader.ReaderClass import ReadText
from src.Classification.Classy import Classifier


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

RT = ReadText()
PP = PreProcess()
seg= Segmentation()
classy = Classifier(RT,PP)
columns = {"Real text":[],"prediction":[]}

test = {"Explosives": ["EXPLOSIVE","1", "1.1","1.2","1.3","1.4","1.5","1.6"],
        "Flammable Gas": ["FLAMMABLE","GAS","FLAMMABLE GAS","2","2.1"],
        "Non-Flammable Gas": ["NON-FLAMMABLE","GAS","2","2.1"],
        "Toxic Gas": ["TOXIC","GAS","TOXIC GAS","2","2.3"],
        "Flammable Liquid": ["Flammable Liquid","Flammable", "Liquid","3"],
        "Flammable Solid": ["FLAMMABLE", "SOLID", "FLAMMABLE SOLID","4","4.1"],
        "Spontaneosly Combustible": ["SPONTANEOUSLY COMBUSTIBLE","SPONTANEOUSLY", "COMBUSTIBLE","4","4.2"],
        "Dangerous When Wet": ["DANGEROUS","WHEN", "WET","DANGEROUS WHEN WET", "4","4.3"],
        "Oxidizing Agent": ["OXIDISING","AGENT","OXIDISING AGENT","5","5.1"],
        "Organic Peroxides": ["ORGANIC", "PEROXIDES","ORGANIC PEROXIDES","5","5.2"],
        "Toxic": ["TOXIC","6","6.1"],
        "Infectious Substance": ["INFECTIOUS","SUBSTANCE","INFECTIOUS SUBSTANCE", "6","6.2"],
        "Corrosive": ["CORROSIVE","8"],
        "Miscellanous": ["MISCELLANEOUS","9"],
        "Lithium Batteries": ["LITHIUM BATTERIES", "9"]}

ground_truth = []
image_folder = os.getcwd() + "/Dataset"

image_paths = [os.path.join(image_folder, img) for img in os.listdir(os.getcwd() + "/Dataset") if img.endswith("_MASK.png")]


for path in tqdm(image_paths):
    text1=[]
    imgpath = path.replace("_MASK","")
    ground_truth.append(path.split("/")[-1].split("_")[0])
    img = cv2.imread(imgpath)
    mask = cv2.imread(path, 0)
    mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    masked=cv2.bitwise_and(img, img, mask=mask)
    if classy.labels_on_edge(mask):
        columns["prediction"].append("")
    else:
        ROI=PP.segmentation_to_ROI(mask)
        for bounds in ROI:
            cropped = masked[bounds[1]:bounds[3], bounds[0]:bounds[2]]
            bounding = RT.findText(cropped)
            for boxes in bounding:
                if "Explosives" in imgpath:
                    predtext= RT.readText(cropped, boxes,display= True)
                    print(path)
                    print(predtext)
                else:
                    predtext= RT.readText(cropped, boxes)
                text1.append(predtext)
        if len(text1) == 0:
            columns["prediction"].append("No text found")
        else:
            columns["prediction"].append(find_closest_match(text1,test)[1])
columns["Real text"] = ground_truth

for i, value in enumerate(columns["prediction"]):
    if value != "":
        if len(value) >= 2:
            columns["prediction"][i] = ""
        else:
            columns["prediction"][i] = value[0]
for i, value in enumerate(columns["prediction"]):
    if value == "":
        columns["prediction"][i] = "Bent"


if len(columns["Real text"]) == len(columns["prediction"]):
    df = pd.DataFrame(columns)
    print(df.to_html(index=False))
else:
    print(columns)

unique_labels = set(ground_truth)
shortening = {"Bent": "F","Corrosive": "Co", "Dangerous When Wet": "DW", "Explosives": "EX", "Flammable Gas": "FG", "Flammable Liquid": "FL", "Flammable Solid": "FS", "Infectious Substance": "IS", "Lithium Batteries": "LB", "Miscellanous": "M", "Non-Flammable Gas": "NF", "Organic Peroxides": "OP", "Oxidizing Agent": "OA", "Spontaneously Combustible": "SC", "Toxic": "T", "Toxic Gas": "TG"}
class_labels = [shortening.get(item, item) for item in unique_labels]

unique_labels = list(set(columns["Real text"] + columns["prediction"]))
if "Bent" not in unique_labels:
    unique_labels.append("Bent")

cm = confusion_matrix(columns["Real text"], columns["prediction"], labels=unique_labels)

# Find the index of the "Fail" class label
fail_index = unique_labels.index("Bent")

# Rearrange the class labels, moving the "Fail" label to the last position
class_labels_reordered = class_labels[:fail_index] + class_labels[fail_index+1:] + [class_labels[fail_index]]

# Rearrange the rows of the confusion matrix
cm_reordered = np.concatenate((cm[:fail_index], cm[fail_index+1:], cm[fail_index:fail_index+1]), axis=0)

# Rearrange the columns of the confusion matrix
cm_reordered = np.concatenate((cm_reordered[:,:fail_index], cm_reordered[:,fail_index+1:], cm_reordered[:,fail_index:fail_index+1]), axis=1)

# Plot the confusion matrix with the reordered class labels and matrix
plt.imshow(cm_reordered, cmap='Blues', interpolation='nearest')

# Add text annotations for each cell in the confusion matrix
for i in range(len(class_labels_reordered)):
    for j in range(len(class_labels_reordered)):
        plt.text(j, i, str(cm_reordered[i][j]), ha='center', va='center', color='white')

# Set the tick marks and labels
tick_marks = np.arange(len(class_labels_reordered))
plt.xticks(tick_marks, class_labels_reordered, rotation='vertical')
plt.yticks(tick_marks, class_labels_reordered)

# Add labels and title
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Display the color bar
plt.colorbar()

# Draw vertical grid lines
for i in range(len(class_labels)-1):
    plt.axvline(x=i+0.5, color='black', linewidth=1)

# Draw horizontal grid lines
for i in range(len(class_labels)-1):
    plt.axhline(y=i+0.5, color='black', linewidth=1)

# Show the plot
plt.show()
