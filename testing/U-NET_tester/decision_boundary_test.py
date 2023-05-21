from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
import sys
import os
sys.path.append("/workspaces/P6-Automated-Hazard-Detection")
sys.path.append("/workspaces/Automated-Hazard-Detection")
from Libs.final_unet.UNET_trainer import UNETTrainer
from Libs.final_unet.utils.model import UNET
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm


album_transform = A.Compose([
    A.Resize(height=1080, width=1920),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Temp data set
class CustomDataset(Dataset):
    def __init__(self, images_folder, masks_folder, transform=None):
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.transform = transform
        self.image_files = os.listdir(images_folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_folder, img_name)
        mask_path = os.path.join(self.masks_folder, img_name)
        with Image.open(img_path) as image, Image.open(mask_path) as mask:  # use 'with' statement to automatically close the file after use
            image = np.array(image.convert("RGB"))
            mask = np.array(mask.convert("L"), dtype=np.float32)
            mask[mask > 0] = 1.0
            if self.transform:
                augmented = self.transform(image=np.array(image), mask=np.array(mask))
                image = augmented['image']
                mask = augmented['mask']
        return image, mask

def evaluate_model(model, dataloader):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Placeholder lists to store the ground truths and the model's predicted probabilities
    all_outputs = []
    all_targets = []

    # Iterate over the test dataset
    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = torch.sigmoid(model(inputs))

        # Store the ground truths and the model's predicted probabilities
        all_outputs.extend(outputs.detach().cpu().numpy().flatten())
        all_targets.extend(targets.cpu().numpy().flatten())
        
        # Free up memory
        del inputs
        del targets
        del outputs
        torch.cuda.empty_cache()
    print("Done evaluating")
    # Compute precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(all_targets, all_outputs)
    average_precision = average_precision_score(all_targets, all_outputs)

    # Compute F1-score for different threshold values
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    
    # Get the threshold that gives the maximum F1-score
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f'Best threshold: {best_threshold}')
    
    return precisions, recalls, average_precision




if __name__ == "__main__":
    mask_types = [ 'UN_circle', 'CAO', 'Shipping', 'this_side_up', 'Lithium']
    #'models/UNET_hazardCombinedLoss.pth''hazard',
    models = ['models/UNET_UN_circleCrossEntropy.pth', 'models/UNET_CAOCrossEntropy.pth', 'models/UNET_ShippingCrossEntropy.pth', 'models/UNET_this_side_upDice_loss.pth', 'models/UNET_LithiumDice_loss.pth']
    # get the parent directory of the current module
    current_file_path = os.path.abspath(__file__)
    # Get the folder containing the current file
    current_folder = os.path.dirname(current_file_path)
    # Get the base folder
    parent_folder = os.path.dirname(os.path.dirname(current_folder))
    
    base_folder = os.path.join(parent_folder,"Libs/final_unet/Full_set")
    rgb_folder = os.path.join(base_folder, "rgb_depth")
    model_folder = os.path.join(parent_folder, "src/Segmentation")
    
    # start the models
    for mask, model_name in zip(mask_types, models):
        # Set up the test set
        mask_folder = os.path.join(base_folder, f"masks_{mask}")
        
        # Load the model
        model = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to('cuda')

        model.load_state_dict(torch.load(os.path.join(model_folder, model_name))["state_dict"])
        model.eval()
        
        trainer = UNETTrainer(base_folder, rgb_folder, mask_folder, NEW_SET=True)
        
        mask_folder = os.path.join(base_folder+"/test_mask")
        img_folder = os.path.join(base_folder+"/test_rgb")
        
        dataset = CustomDataset(images_folder=img_folder, masks_folder=mask_folder, transform=album_transform)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        
        # Evaluate the model
        precisions, recalls, average_precision = evaluate_model(model, dataloader)

        # Plot the Precision-Recall curve
        plt.figure()
        plt.plot(recalls, precisions, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))
        plt.legend(loc="lower right")
        # Save the figure
        plt.savefig(os.path.join(model_folder, f'prc_{mask}.png'))
            
            