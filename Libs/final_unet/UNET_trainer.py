import os
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn.modules import loss
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from utils.model import UNET
from utils.utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
from sklearn.model_selection import train_test_split
from utils.lossModels import DiceLoss, CombinedLoss, FocalLoss
# Get the preprocessing functions
import sys
sys.path.append('/workspaces/P6-Automated-Hazard-Detection')
from src.Preprocess.prep import PreProcess


class UNETTrainer:
    def __init__(self, base_folder, rgb_folder, mask_folder, image_size=[1080,1920], model_name="model.pth", 
                 batch_size=1, epochs=70, learning_rate=1e-4, worker_threads = 2, loss_model=DiceLoss(),
                 AUGMENT_SET=True,NEW_SET=True, DEBUG_PLOT = False):
        """
        U-NET TRAINER, used for training the U-NET model.

        Args:
            base_folder (str): folder path to the base folder
            rgb_folder (str): folder path to the rgb images
            mask_folder (str): folder path to the mask images
            image_size (list, optional): image sizes. Defaults to [1080,1920].
            model_name (str, optional): model name. Defaults to "model.pth".
            batch_size (int, optional):batch size, set according to hardware. Defaults to 1.
            epochs (int, optional): Amount of epocs. Defaults to 70.
            learning_rate (float, optional): learning rate. Defaults to 1e-4.
            worker_threads (int, optional): Worker threads for multithreading, set according to hardware. Defaults to 2.
            loss_model (Class object, optional): Loss model used to train. Defaults to DiceLoss().
            AUGMENT_SET (bool, optional): Set to True to augment the training set. Defaults to True.
            NEW_SET (bool, optional): If the data has to be sorted. Defaults to True.
            DEBUG_PLOT (bool, optional): Plot a random set of images. Defaults to False.
        """
        if NEW_SET:
            # Fix the folder for the UNET
            rgb_list = [f'{img}' for img in os.listdir(rgb_folder) if img.startswith("rgb_image")]
            mask_list = [f'{img}' for img in os.listdir(mask_folder)]

            mask_list = self._merge_masks(mask_folder, mask_list)
            rgb_list = self._onlyTruths(rgb_list, mask_list)
            train_folders, test_folders = self._split_images(base_folder, rgb_folder, mask_folder, rgb_list, mask_list)
            # Preprocess the images
            self._preprocess_images(train_folders, test_folders, rgb_folder)
        else:
            train_folders = [os.path.join(base_folder, "train_mask"), os.path.join(base_folder, "train_rgb")]
            test_folders = [os.path.join(base_folder, "test_mask"), os.path.join(base_folder, "test_rgb")]

        if DEBUG_PLOT:
            visualize_samples(train_folders[1], train_folders[0])
        
        if AUGMENT_SET:
            augmentations = A.Compose([
                                        A.Resize(height=image_size[0], width=image_size[1]),
                                        A.Rotate(limit=35, p=1.0),
                                        A.HorizontalFlip(p=0.5),
                                        A.VerticalFlip(p=0.1),
                                        A.RandomBrightnessContrast(p=0.3),
                                        A.Normalize(
                                            # ImageNet std values
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225],
                                            max_pixel_value=255.0,
                                        ),
                                        ToTensorV2(),
                                      ],)
        else:
            augmentations = A.Compose([
                                        A.Resize(height=image_size[0], width=image_size[1]),
                                        A.Rotate(limit=35, p=1.0),
                                        A.HorizontalFlip(p=0.5),
                                        A.VerticalFlip(p=0.1),
                                        A.Normalize(
                                            # ImageNet std values
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225],
                                            max_pixel_value=255.0,
                                        ),
                                        ToTensorV2(),
                                      ],)
        
        os.makedirs(os.path.join(base_folder, "Images"),exist_ok=True)
        self.config = {
                "learning_rate": learning_rate,
                "device": "cuda",
                "batch_size": batch_size,
                "num_epochs": epochs,
                "num_workers": worker_threads,
                "image_height": int(image_size[0]),
                "image_width": int(image_size[1]),
                "pin_memory": True,
                "load_model": False,
                "train_img_dir": train_folders[1],
                "train_mask_dir": train_folders[0],
                "val_img_dir": test_folders[1],
                "val_mask_dir": test_folders[0],
                "model_name": os.path.join(base_folder, model_name),
                "save_folder": os.path.join(base_folder, "Images"),
                "train_transform": augmentations,
                "val_transforms": A.Compose(
                    [
                        A.Resize(height=image_size[0], width=image_size[1]),
                        A.Normalize(
                            # ImageNet std values
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                            max_pixel_value=255.0,
                        ),
                        ToTensorV2(),
                    ],
                ),
            }
        self.model = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(self.config["device"])
        self.loss_fn = loss_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.scaler = torch.cuda.amp.GradScaler()
        self.tb = SummaryWriter()
        self.max_score = 0
        self.train_loader, self.val_loader = get_loaders(
            self.config["train_img_dir"],
            self.config["train_mask_dir"],
            self.config["val_img_dir"],
            self.config["val_mask_dir"],
            self.config["batch_size"],
            self.config["train_transform"],
            self.config["val_transforms"],
            self.config["num_workers"],
            self.config["pin_memory"],
        )

    def train_fn(self, loader):
        loop = tqdm(loader)
        total_loss = 0
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=self.config["device"])
            targets = targets.float().unsqueeze(1).to(device=self.config["device"])
            
            with torch.cuda.amp.autocast():
                predictions = self.model(data)
                loss = self.loss_fn(predictions, targets)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loop.set_postfix(loss=loss.item())
            total_loss += loss.item()
        return total_loss

    def train(self):
        if self.config["load_model"]:
            load_checkpoint(torch.load("model/TrainingModel.pth"), self.model)

        check_accuracy(self.val_loader, self.model, device=self.config["device"])

        for epoch in range(0, self.config["num_epochs"]):
            loss = self.train_fn(self.train_loader)

            checkpoint = {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            IoU, F1, acc = check_accuracy(self.val_loader, self.model, device=self.config["device"])

            if IoU > self.max_score:
                print("Best model found => saving")
                self.max_score = IoU
                save_checkpoint(checkpoint, self.config["model_name"])

                save_predictions_as_imgs(
                    self.val_loader, self.model, folder=self.config["save_folder"], device=self.config["device"]
                )

            print(f"EPOCH: {epoch}/{self.config['num_epochs']}")
            self.tb.add_scalar('Accuracy', acc, epoch)
            self.tb.add_scalar('Loss', loss / len(self.train_loader), epoch)
            self.tb.add_scalar('F1-score', F1, epoch)
            self.tb.add_scalar('IoU', IoU, epoch)
            
            # Next step in the scheduler
            self.scheduler.step()
        self.tb.close()
    
    def _merge_masks(self, mask_folder, mask_list):
        # start finding duplicates
        for i, image1 in enumerate(mask_list):
            imag1_split = image1.split("_")
            for j, image2 in enumerate(mask_list):
                imag2_split = image2.split("_")
                if i != j and imag1_split[-2] == imag2_split[-2]:
                    if imag1_split[-1] != imag2_split[-1]:
                        img1 = cv2.imread(os.path.join(mask_folder, image1))
                        img2 = cv2.imread(os.path.join(mask_folder, image2))
                        img1 = cv2.add(img1, img2)
                        cv2.imwrite(os.path.join(mask_folder, image1), img1)
                        os.remove(os.path.join(mask_folder, image2))
                        mask_list.pop(j)
        return mask_list
    
    def _empty_folder(self, folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            # Check if it's a file and not a directory
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                # If it's a directory, ignore it
                continue
    
    def _split_images(self, folder, rgb_folder, mask_folder, rgb_list, mask_list, train_ratio=0.8):
        # Create new folders for train and test data
        train_mask_folder = os.path.join(folder, "train_mask")
        train_rgb_folder = os.path.join(folder, "train_rgb")
        test_mask_folder = os.path.join(folder, "test_mask")
        test_rgb_folder = os.path.join(folder, "test_rgb")
        os.makedirs(train_mask_folder, exist_ok=True)
        os.makedirs(train_rgb_folder, exist_ok=True)
        os.makedirs(test_mask_folder, exist_ok=True)
        os.makedirs(test_rgb_folder, exist_ok=True)
        # Make sure the folders are empty
        self._empty_folder(train_mask_folder)
        self._empty_folder(train_rgb_folder)
        self._empty_folder(test_mask_folder)
        self._empty_folder(test_rgb_folder)
 
        # Split the images randomly into train and test sets
        mask_train, mask_test, rgb_train, rgb_test = train_test_split(mask_list, rgb_list, train_size=train_ratio, random_state=42)

        # Move the train images to the train folders and save as PNG
        for mask_file, rgb_file in tqdm(zip(mask_train, rgb_train), desc='Moving train images'):
            mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
            rgb = cv2.imread(os.path.join(rgb_folder, rgb_file))

            cv2.imwrite(os.path.join(train_mask_folder, mask_file[:-10]+".png"), mask)
            cv2.imwrite(os.path.join(train_rgb_folder, rgb_file), rgb)
            
        # Move the test images to the test folders and save as PNG
        for mask_file, rgb_file in tqdm(zip(mask_test, rgb_test), desc='Moving test images'):
            mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
            rgb = cv2.imread(os.path.join(rgb_folder, rgb_file))
            cv2.imwrite(os.path.join(test_mask_folder, mask_file[:-10]+".png"), mask)
            cv2.imwrite(os.path.join(test_rgb_folder, rgb_file), rgb)
        return([train_mask_folder, train_rgb_folder], [test_mask_folder, test_rgb_folder])
    
    def _onlyTruths(self, rgb_list, mask_list):
        mask_list = [mask[:-10]+".png" for mask in mask_list]
        mask_set = set(mask_list)
        not_in_mask = [img for img in rgb_list if img not in mask_set]
        for img in not_in_mask:
            rgb_list.remove(img)
        return rgb_list
    
    # Preprocess the images
    def _preprocess_images(self, train_folders, test_folders, depth_folder):
        pp = PreProcess()
        # Get a list of the images
        train_mask_list = os.listdir(train_folders[0])
        train_rgb_list = os.listdir(train_folders[1])
        test_mask_list = os.listdir(test_folders[0])
        test_rgb_list = os.listdir(test_folders[1])
        # Run through the images and preprocess them
        for mask_file, rgb_file in tqdm(zip(train_mask_list, train_rgb_list), desc='Preprocessing train images'):
            # Retrtieve the depth image
            depth_file = os.path.join(depth_folder, rgb_file.replace("rgb_image", "depth_image").replace("png", "raw"))
            depth = np.fromfile(depth_file, dtype=np.uint16)
            # Reconstruct the depth map
            depth = depth.reshape(1080, 1920)
            depth = cv2.medianBlur(depth, 5)
            # load the images
            mask = cv2.imread(os.path.join(train_folders[0], mask_file), cv2.IMREAD_GRAYSCALE)
            rgb = cv2.imread(os.path.join(train_folders[1], rgb_file))
            
            # undistort the image
            rgb = pp.undistort_images(rgb)
            mask = pp.undistort_images(mask)
            depth = pp.undistort_images(depth)
            
            # Warp the images
            trans_img, _, trans_mask = pp.retrieve_transformed_plane(rgb, depth, mask=mask)
            
            if np.any(trans_mask != 0):
                # Save the images
                cv2.imwrite(os.path.join(train_folders[0], mask_file), trans_mask)
                cv2.imwrite(os.path.join(train_folders[1], rgb_file), trans_img)
            else:
                # Remove the images
                os.remove(os.path.join(train_folders[0], mask_file))
                os.remove(os.path.join(train_folders[1], rgb_file))
        for mask_file, rgb_file in tqdm(zip(test_mask_list, test_rgb_list), desc='Preprocessing test images'):
            # Retrtieve the depth image
            depth_file = os.path.join(depth_folder, rgb_file.replace("rgb_image", "depth_image").replace("png", "raw"))
            depth = np.fromfile(depth_file, dtype=np.uint16)
            # Reconstruct the depth map
            depth = depth.reshape(int(1080), int(1920))
            # Blue the depth image
            depth = cv2.medianBlur(depth, 5)
            # load the images
            mask = cv2.imread(os.path.join(test_folders[0], mask_file), cv2.IMREAD_GRAYSCALE)
            rgb = cv2.imread(os.path.join(test_folders[1], rgb_file))
            
            # undistort the image
            rgb = pp.undistort_images(rgb)
            mask = pp.undistort_images(mask)
            depth = pp.undistort_images(depth)
            
            # Warp the images
            trans_img, _, trans_mask = pp.retrieve_transformed_plane(rgb, depth, mask=mask)

            if np.any(trans_mask != 0):
                # Save the images
                cv2.imwrite(os.path.join(test_folders[0], mask_file), trans_mask)
                cv2.imwrite(os.path.join(test_folders[1], rgb_file), trans_img)
            else:
                # Remove the images
                os.remove(os.path.join(test_folders[0], mask_file))
                os.remove(os.path.join(test_folders[1], rgb_file))


def visualize_samples(image_folder, mask_folder, num_samples=5):
    import random
    image_files = os.listdir(image_folder)
    mask_files = os.listdir(mask_folder)
    
    # Genrate random samples
    random_samples = random.sample(range(0, len(image_files)), num_samples)
    
    for i in random_samples:
        image_file = os.path.join(image_folder, image_files[i])
        mask_file = os.path.join(mask_folder, mask_files[i])

        # Load the images
        image = cv2.imread(image_file)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Plot the images side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(image)
        ax1.set_title("Image")
        ax1.axis("off")

        ax2.imshow(mask, cmap="gray")
        ax2.set_title("Mask")
        ax2.axis("off")

        plt.show()


def main():
    # get the parent directory of the current module
    current_file_path = os.path.abspath(__file__)
    # Get the folder containing the current file
    current_folder = os.path.dirname(current_file_path)
    # Get the parent folder of the current folder
    parent_folder = os.path.dirname(current_folder)

    base_folder = os.path.join(parent_folder, "Initial_unet/Training")
    rgb_folder = os.path.join(base_folder, "original/rgb_depth")
    mask_folder = os.path.join(base_folder, "original/masks/masks")

    #print(rgb_folder, mask_folder)
    trainer = UNETTrainer(base_folder, rgb_folder, mask_folder, worker_threads=0, batch_size=1, NEW_SET=False, DEBUG_PLOT=True)
    trainer.train()


if __name__ == "__main__":
    main()