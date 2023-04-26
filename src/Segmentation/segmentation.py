from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
import cv2
import torch.utils
from Libs.Initial_unet.utils.model import UNET



class Segmentation:
    def __init__(self):
        DEVICE = "cuda"
        img = cv2.imread("example.png")
        transform = T.Compose([
            T.Resize((562, 562), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        ])
        print("loading model 1/6")
        self.model1 = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(DEVICE)
        self.model1.load_state_dict(torch.load("Models/TrainingModel.pth")["state_dict"])
        # Set the model to evaluation mode
        self.model1.eval()

        print("loading model 2/6")
        self.model2 = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(DEVICE)
        self.model2.load_state_dict(torch.load("Models/TrainingModelUN.pth")["state_dict"])
        self.model2.eval()

        print("loading model 3/6")
        self.model3 = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(DEVICE)
        self.model3.load_state_dict(torch.load("Models/TrainingModelCAO.pth")["state_dict"])
        self.model3.eval()

        print("loading model 4/6")
        self.model4 = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(DEVICE)
        self.model4.load_state_dict(torch.load("Models/TrainingModelPS.pth")["state_dict"])
        self.model4.eval()

        print("loading model 5/6")
        self.model5 = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(DEVICE)
        self.model5.load_state_dict(torch.load("Models/TrainingModelTSU.pth")["state_dict"])
        self.model5.eval()

        print("loading model 6/6")
        self.model6 = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(DEVICE)
        self.model6.load_state_dict(torch.load("Models/TrainingModelLithium.pth")["state_dict"])
        self.model6.eval()

        print("loading model done")

        print("Warming up torch(please ignore user warning)")
        # Convert the BGR NumPy array to RGB
        image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert the NumPy array to a PIL image
        image = Image.fromarray(image_np)
        x = transform(image)
        # Prepare the image tensor for the model
        x = x.unsqueeze(0).to(device=DEVICE)
        # Perform a forward pass and create a binary segmentation mask
        with torch.no_grad():
            preds = torch.sigmoid(self.model1(x))
        print("The torch is now lit")
        print("ready for use")

    def locateHazard(self, image_np, device="cuda", out_x=1920, out_y=1080):
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

    def locateUN(self, image_np, device="cuda", out_x=1920, out_y=1080):
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
        # Perform a forward pass and create a binary segmentation mask
        with torch.no_grad():
            sig = torch.sigmoid(self.model2(x))
            preds = (sig > 0.6).float()
        preds = preds.cpu().numpy()
        if preds.shape[0] == 1:
            preds = preds.squeeze(0)
        preds = np.stack([preds] * 3, axis=-1)
        preds = (preds * 255).astype(np.uint8)
        if preds.shape[0] == 1:
            preds = preds.squeeze(0)
        preds = cv2.resize(preds, (out_x,out_y))

        return preds

    def locateCAO(self, image_np, device="cuda", out_x=1920, out_y=1080):
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
        # Perform a forward pass and create a binary segmentation mask
        with torch.no_grad():
            sig = torch.sigmoid(self.model3(x))
            preds = (sig > 0.6).float()
        preds = preds.cpu().numpy()
        if preds.shape[0] == 1:
            preds = preds.squeeze(0)
        preds = np.stack([preds] * 3, axis=-1)
        preds = (preds * 255).astype(np.uint8)
        if preds.shape[0] == 1:
            preds = preds.squeeze(0)
        preds = cv2.resize(preds, (out_x,out_y))

        return preds

    def locateShipping(self, image_np, device="cuda", out_x=1920, out_y=1080):
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
        # Perform a forward pass and create a binary segmentation mask
        with torch.no_grad():
            sig = torch.sigmoid(self.model4(x))
            preds = (sig > 0.6).float()
        preds = preds.cpu().numpy()
        if preds.shape[0] == 1:
            preds = preds.squeeze(0)
        preds = np.stack([preds] * 3, axis=-1)
        preds = (preds * 255).astype(np.uint8)
        if preds.shape[0] == 1:
            preds = preds.squeeze(0)
        preds = cv2.resize(preds, (out_x,out_y))

        return preds

    def locateTSU(self, image_np, device="cuda", out_x=1920, out_y=1080):
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
        # Perform a forward pass and create a binary segmentation mask
        with torch.no_grad():
            sig = torch.sigmoid(self.model5(x))
            preds = (sig > 0.6).float()
        preds = preds.cpu().numpy()
        if preds.shape[0] == 1:
            preds = preds.squeeze(0)
        preds = np.stack([preds] * 3, axis=-1)
        preds = (preds * 255).astype(np.uint8)
        if preds.shape[0] == 1:
            preds = preds.squeeze(0)
        preds = cv2.resize(preds, (out_x,out_y))

        return preds

    def locateLithium(self, image_np, device="cuda", out_x=1920, out_y=1080):
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
        # Perform a forward pass and create a binary segmentation mask
        with torch.no_grad():
            sig = torch.sigmoid(self.model6(x))
            preds = (sig > 0.6).float()
        preds = preds.cpu().numpy()
        if preds.shape[0] == 1:
            preds = preds.squeeze(0)
        preds = np.stack([preds] * 3, axis=-1)
        preds = (preds * 255).astype(np.uint8)
        if preds.shape[0] == 1:
            preds = preds.squeeze(0)
        preds = cv2.resize(preds, (out_x,out_y))

        return preds
