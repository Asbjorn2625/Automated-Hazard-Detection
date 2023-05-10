import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
    def __len__(self):
        assert len(self.images)/len(self.masks) == 1
        return len(self.images)

    def __getitem__(self, index):
        img_path = "%s/%s" % (self.image_dir, self.images[index])
        mask_path = "%s/%s" % (self.mask_dir, self.masks[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask > 0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

    