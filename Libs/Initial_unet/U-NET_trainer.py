import os
import cv2
import matplotlib.pyplot as plt
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

class UNETTrainer:
    def __init__(self, base_folder, rgb_folder, mask_folder, image_size=[1920,1080], batch_size=1, epochs=70, learning_rate=1e-5, worker_threads = 2, NEW_SET=True, DEBUG_PLOT = False):
        if NEW_SET:
            # Fix the folder for the UNET
            rgb_list = [f'{img}' for img in os.listdir(rgb_folder) if img.startswith("rgb_image")]
            depth_list = [path.replace("rgb_image", "depth_image").replace("png", "raw") for path in rgb_list]
            mask_list = [f'{img}' for img in os.listdir(mask_folder)]

            mask_list = self._merge_masks(mask_folder, mask_list)
            rgb_list = self._onlyTruths(rgb_list, mask_list)
            train_folders, test_folders = self._split_images(base_folder, rgb_folder, mask_folder, rgb_list, mask_list)
        else:
            train_folders = [os.path.join(base_folder, "train_mask"), os.path.join(base_folder, "train_rgb")]
            test_folders = [os.path.join(base_folder, "test_mask"), os.path.join(base_folder, "test_rgb")]

        if DEBUG_PLOT:
            visualize_samples(train_folders[1], train_folders[0])
    
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
                "model_name": os.path.join(base_folder, "UNmodel.pth"),
                "save_folder": os.path.join(base_folder, "Images"),
                "train_transform": A.Compose(
                    [
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
                    ],
                ),
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


def visualize_samples(image_folder, mask_folder, num_samples=5):
    image_files = os.listdir(image_folder)
    mask_files = os.listdir(mask_folder)

    for i in range(num_samples):
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
    base_folder = os.getcwd().replace("\\", "/") + "/Libs/Initial_unet/Training"
    rgb_folder = os.path.join(base_folder, "original/rgb_depth")
    mask_folder = os.path.join(base_folder, "original/masks/masks")
    #print(rgb_folder, mask_folder)
    trainer = UNETTrainer(base_folder, rgb_folder, mask_folder, NEW_SET=False, DEBUG_PLOT=False)
    trainer.train()


if __name__ == "__main__":
    main()