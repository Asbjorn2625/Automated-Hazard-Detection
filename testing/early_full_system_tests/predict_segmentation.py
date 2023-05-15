import numpy as np
import cv2
import os
import sys
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, jaccard_score
import pandas as pd
import csv

sys.path.append("/workspaces/P6-Automated-Hazard-Detection")
from src.Preprocess.prep import PreProcess
from src.Segmentation.segmentation import Segmentation
import matplotlib.pyplot as plt
from Libs.final_unet.UNET_trainer import  *

def evaluate_metrics(y_true, y_pred):
    # Convert masks to binary format
    y_true = np.where(y_true > 0, 1, 0)
    y_pred = np.where(y_pred > 0, 1, 0)

    # Flatten the arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate metrics
    precision = precision_score(y_true_flat, y_pred_flat)
    recall = recall_score(y_true_flat, y_pred_flat)
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    dice = f1_score(y_true_flat, y_pred_flat)
    iou = jaccard_score(y_true_flat, y_pred_flat)

    return precision, recall, accuracy, dice, iou

def display_images(images):
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(num_images*5, 5))
    for i, img in enumerate(images):
        # Change image to RGB and rotate it 90 degrees
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.show()

types_of_models = ['CombinedLoss', 'Dice_loss', 'IoU','CrossEntropy']
#mask_types = ['hazard', 'UN_circle', 'CAO', 'Shipping', 'this_side_up']
mask_types = ['this_side_up']
if __name__ == "__main__":
    # get the parent directory of the current module
    current_file_path = os.path.abspath(__file__)
    # Get the folder containing the current file
    current_folder = os.path.dirname(current_file_path)
    # Get the base folder
    parent_folder = os.path.dirname(os.path.dirname(current_folder))
    
    base_folder = os.path.join(parent_folder,"Libs/final_unet/Full_set")
    rgb_folder = os.path.join(base_folder, "rgb_depth")
    
    # Initialize a dictionary to store the extreme images
    extreme_images = {}
    results = []
    
    # start the models
    for mask in mask_types:
        # Set up the test set
        mask_folder = os.path.join(base_folder, f"masks_{mask}")
        trainer = UNETTrainer(base_folder, rgb_folder, mask_folder, NEW_SET=True)
        for type in types_of_models:
            segmentor = Segmentation(model_type=type)
            pp = PreProcess()
            
            # load the images 
            paths = [[os.path.join(base_folder+"/test_rgb", img), os.path.join(base_folder+"/test_mask", img)] for img in os.listdir(base_folder+"/test_mask") if img.startswith("rgb_image")]
            
            # Initialize the variables to store the extreme images for the current type and mask
            best_dice = 0
            worst_dice = 1
            best_img_path = None
            worst_img_path = None

            # Loop through the images
            for img_path, mask_path in paths:
                img = cv2.imread(img_path)
                ground_truth = cv2.imread(mask_path)
                if mask == "hazard":
                    predicted_mask = segmentor.locateHazard(img)
                elif mask == "UN_circle":
                    predicted_mask = segmentor.locateUN(img)
                elif mask == "CAO":
                    predicted_mask = segmentor.locateCao(img)
                elif mask == "Shipping":
                    predicted_mask = segmentor.locatePS(img)
                elif mask == "this_side_up":
                    predicted_mask = segmentor.locateTSU(img)

                # Get the precision of the predicted model compared to the ground truth
                precision, recall, accuracy, dice, iou = evaluate_metrics(ground_truth, predicted_mask)

                # Append to results
                results.append({"image":img_path.split("/")[-1],
                                "model":type,
                                "mask":mask,
                                "precision":precision, 
                                "recall":recall,
                                "accuracy":accuracy,
                                "dice":dice,
                                "iou":iou})
                
                if dice > best_dice:
                    best_dice = dice
                    best_img_path = img_path

                if dice < worst_dice:
                    worst_dice = dice
                    worst_img_path = img_path
                    
            extreme_images[(mask, type)] = {
                'best': cv2.imread(best_img_path),
                'worst': cv2.imread(worst_img_path),
            }

    # Writing results to a CSV file
    csv_columns = ["image", "model", "mask", "precision", "recall", "accuracy", "dice", "iou"]
    csv_file = "results2.csv"

    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Save images for the report
    for (mask, type), images in extreme_images.items():
        cv2.imwrite(f'best_{mask}_{type}.png', images['best'])
        cv2.imwrite(f'worst_{mask}_{type}.png', images['worst'])