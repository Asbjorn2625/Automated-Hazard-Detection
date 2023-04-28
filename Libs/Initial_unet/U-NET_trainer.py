import os
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn.modules import loss
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from utils.model import UNET
from utils.utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
from utils.Dice import DiceLoss
from sklearn.model_selection import train_test_split

import sys
sys.path.append('/workspaces/P6-Automated-Hazard-Detection')
from Libs.functions.Merge_anntotes import Merge


class UNETTrainer:
    def __init__(self, rgb_folder, mask_folder, image_size=[1920,1080], batch_size=1, epochs=70, learning_rate=1e-4):
        # Fix the folder for the UNET
        
        self.config = {
                "learning_rate": learning_rate,
                "device": "cuda",
                "batch_size": batch_size,
                "num_epochs": epochs,
                "num_workers": 4,
                "image_height": int(image_size[0]),
                "image_width": int(image_size[1]),
                "pin_memory": True,
                "load_model": False,
                "train_img_dir": rgb_folder,
                "train_mask_dir": mask_folder,
                "val_img_dir": rgb_folder,
                "val_mask_dir": mask_folder,
                "model_name": os.getcwd().replace("\\", "/") + "UNmodel.pth",
                "save_folder": os.getcwd().replace("\\", "/") + "/Initial_unet/Training/Images",
                "train_transform": A.Compose(
                    [
                        A.Resize(height=image_size[0], width=image_size[1]),
                        A.Rotate(limit=35, p=1.0),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.1),
                        A.RandomBrightnessContrast(p=0.3),
                        A.Normalize(
                            mean=[0.0, 0.0, 0.0],
                            std=[1.0, 1.0, 1.0],
                            max_pixel_value=255.0,
                        ),
                        ToTensorV2(),
                    ],
                ),
                "val_transforms": A.Compose(
                    [
                        A.Resize(height=image_size[0], width=image_size[1]),
                        A.Normalize(
                            mean=[0.0, 0.0, 0.0],
                            std=[1.0, 1.0, 1.0],
                            max_pixel_value=255.0,
                        ),
                        ToTensorV2(),
                    ],
                ),
            }
        self.model = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(self.config["device"])
        self.loss_fn = DiceLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
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

            if epoch == 30:
                print("Changing learning rate to 1e-5")
                self.optimizer.param_groups[0]['lr'] = 1e-5
            if epoch == 60:
                print("Changing learning rate to 1e-6")
                self.optimizer.param_groups[0]['lr'] = 1e-6

            print(f"EPOCH: {epoch}/{self.config['num_epochs']}")
            self.tb.add_scalar('Accuracy', acc, epoch)
            self.tb.add_scalar('Loss', loss / len(self.train_loader), epoch)
            self.tb.add_scalar('F1-score', F1, epoch)
            self.tb.add_scalar('IoU', IoU, epoch)
            self.tb.close()
    
    def _merge_masks(self, folder):
        # use os library to search for all files in a masks directory
        Mask_cwd = os.path.join(os.getcwd(), folder)
        file_list = os.listdir(Mask_cwd)

        # start finding duplicates
        for i, image1 in enumerate(file_list):
            imag1_split = image1.split("_")
            for j, image2 in enumerate(file_list):
                imag2_split = image2.split("_")
                if i != j and imag1_split[2] == imag2_split[2]:
                    print("found duplicate")
                    if imag1_split[3] != imag2_split[3]:
                        img1 = cv2.imread(os.path.join(Mask_cwd, image1))
                        img2 = cv2.imread(os.path.join(Mask_cwd, image2))
                        img1 = cv2.add(img1, img2)
                        cv2.imwrite(os.path.join(Mask_cwd, image1), img1)
                        os.remove(os.path.join(Mask_cwd, image2))
                        file_list.pop(j)
                        print("removed duplicate")
    
    def _split_images(folder, rgb_folder, mask_folder, train_ratio=0.8):
        # Create new folders for train and test data
        base_path = os.getcwd()
        train_mask_folder = os.path.join(base_path, folder, "train_mask")
        train_rgb_folder = os.path.join(base_path, folder, "train_rgb")
        test_mask_folder = os.path.join(base_path, folder, "test_mask")
        test_rgb_folder = os.path.join(base_path, folder, "test_rgb")
        os.makedirs(train_mask_folder, exist_ok=True)
        os.makedirs(train_rgb_folder, exist_ok=True)
        os.makedirs(test_mask_folder, exist_ok=True)
        os.makedirs(test_rgb_folder, exist_ok=True)
        
        # Get list of mask and RGB images
        mask_list = os.listdir(mask_folder)
        rgb_list = os.listdir(rgb_folder)
        
        # Split the images randomly into train and test sets
        mask_train, mask_test, rgb_train, rgb_test = train_test_split(mask_list, rgb_list, train_size=train_ratio, random_state=42)
        
        # Move the train images to the train folders and save as PNG
        for mask_file, rgb_file in tqdm(zip(mask_train, rgb_train), desc='Moving train images'):
            mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
            rgb = cv2.imread(os.path.join(rgb_folder, rgb_file))
            cv2.imwrite(os.path.join(train_mask_folder, mask_file), mask)
            cv2.imwrite(os.path.join(train_rgb_folder, rgb_file), rgb)
            
        # Move the test images to the test folders and save as PNG
        for mask_file, rgb_file in tqdm(zip(mask_test, rgb_test), desc='Moving test images'):
            mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
            rgb = cv2.imread(os.path.join(rgb_folder, rgb_file))
            cv2.imwrite(os.path.join(test_mask_folder, mask_file), mask)
            cv2.imwrite(os.path.join(test_rgb_folder, rgb_file), rgb)
    
    def _killTheExtraNumber(self, folder):
        # use os library to search for all files in a masks directory
        Mask_cwd = os.path.join(os.getcwd(), folder)
        file_list = os.listdir(Mask_cwd)

        # start finding duplicates
        for i, image1 in enumerate(file_list):
            imag1_split = image1.split("_")
            new_name=os.path.join(Mask_cwd, "%s_%s_%s.png" % (imag1_split[0], imag1_split[1], imag1_split[2])).replace("\\", "/")
            os.rename(os.path.join(Mask_cwd, image1).replace("\\", "/"), new_name)
    
    def _onlyTruths(self, rgb_folder, mask_folder): 
        mask_list = os.listdir(mask_folder)
        rgb_list = os.listdir(rgb_folder)
        mask_set = set(mask_list)
        not_in_mask = [img for img in rgb_list if img not in mask_set]
        for img in not_in_mask:
            os.remove(os.path.join(rgb_folder, img))

def main():
    base_folder = os.getcwd().replace("\\", "/") + "/Libs/Initial_unet/Training"
    rgb_folder = os.path.join(base_folder, "train_rgb_depth")
    mask_folder = os.path.join(base_folder, "train_mask")
    #print(rgb_folder, mask_folder)
    trainer = UNETTrainer(rgb_folder, mask_folder)
    trainer.train()


if __name__ == "__main__":
    main()