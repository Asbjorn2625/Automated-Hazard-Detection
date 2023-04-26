from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
import cv2
import torch.utils
from utils.model import UNET



class Segmentation:

    def __init__(self):
        DEVICE = "cuda"
        print("loading model 1/1")
        self.model1 = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(DEVICE)
        self.model1.load_state_dict(torch.load("TrainingModel.pth")["state_dict"])
        print("done")

    def locateHazard(self, image_np, folder="saved_images/", device="cuda", out_x=1920, out_y=1080):
        # Ensure the input is a NumPy array with the correct data type
        assert isinstance(image_np, np.ndarray), "Input must be a numpy.ndarray"
        assert image_np.dtype == np.uint8, "Input array must have dtype 'uint8'"

        transform = T.Compose([
            T.Resize((562, 562), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        ])

        # Convert the BGR NumPy array to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Convert the NumPy array to a PIL image
        image = Image.fromarray(image_np)

        x = transform(image)

        # Prepare the image tensor for the model
        x = x.unsqueeze(0).to(device=device)

        # Set the model to evaluation mode
        self.model1.eval()

        # Perform a forward pass and create a binary segmentation mask
        with torch.no_grad():
            preds = torch.sigmoid(self.model1(x))
            preds = (preds > 0.6).float()

        preds = preds.cpu().numpy()

        if preds.shape[0] == 1:
            preds = preds.squeeze(0)

        preds = np.stack([preds] * 3, axis=-1)
        preds = (preds * 255).astype(np.uint8)
        if preds.shape[0] == 1:
            preds = preds.squeeze(0)

        preds = cv2.resize(preds, (out_x,out_y))

        return preds
