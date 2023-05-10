from UNET_trainer import  *
from utils.lossModels import CombinedLoss, IoULoss
import torch.nn as nn
import os

#mask_types = ["CAO", "hazard", "Lithium", "Shipping", "this_side_up", "UN_circle"]
mask_types = ["UN_circle"]
folder_name = "rgb_depth"

if __name__ == "__main__":
    # get the parent directory of the current module
    current_file_path = os.path.abspath(__file__)
    # Get the folder containing the current file
    current_folder = os.path.dirname(current_file_path)

    base_folder = os.path.join(current_folder, "Full_set")
    rgb_folder = os.path.join(base_folder, folder_name)
    
    for type in mask_types:
        mask_folder = os.path.join(base_folder, f"masks_{type}")
        model_name = f"UNET_{type}"
        print(f"Training {model_name}")
        # Go through the different models and save them individually
        trainer = UNETTrainer(base_folder, rgb_folder, mask_folder, worker_threads=6,
                              batch_size=2, model_name=model_name+"Dice_loss"+".pth", NEW_SET=True)
        trainer.train()
        
        trainer = UNETTrainer(base_folder, rgb_folder, mask_folder, worker_threads=6,
                              batch_size=2, model_name=model_name+"IoU"+".pth", loss_model=IoULoss(),
                              NEW_SET=False)
        trainer.train()
        
        trainer = UNETTrainer(base_folder, rgb_folder, mask_folder, worker_threads=6,
                              batch_size=2, model_name=model_name+"CrossEntropy"+".pth", loss_model=nn.BCEWithLogitsLoss(),
                              NEW_SET=False)
        trainer.train()
        
        trainer = UNETTrainer(base_folder, rgb_folder, mask_folder, worker_threads=6,
                              batch_size=2, model_name=model_name+"CombinedLoss"+".pth", loss_model=CombinedLoss(),
                              NEW_SET=False)
        trainer.train()
        
        
        
        # Might try without augmentation later

