from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
import cv2
import torch.utils
import sys
import os
sys.path.append(os.getcwd().replace("\\", "/") + "/Libs")
from Initial_unet.utils.model import UNET

class Segmentation:
    def __init__(self):
        self.DEVICE = "cuda"
        self.Hazardmodel= None
        self.UNmodel= None
        self.CAOmodel= None
        self.PSmodel= None
        self.TSUmodel= None
        self.Lithiummodel= None
        self._load_models()

    def _load_models(self):
            # Load all the model and set it to evaluation mode
            #First up is the Hazard model
            self.Hazardmodel = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(self.DEVICE)
            self.Hazardmodel.load_state_dict(torch.load("Models/TrainingModel.pth")["state_dict"])
            self.Hazardmodel.eval()
            #Next up is the UN model
            self.UNmodel = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(self.DEVICE)
            self.UNmodel.load_state_dict(torch.load("Models/TrainingModelUN.pth")["state_dict"])
            self.UNmodel.eval()
            #CAO model
            self.CAOmodel = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(self.DEVICE)
            self.CAOmodel.load_state_dict(torch.load("Models/TrainingModelCAO.pth")["state_dict"])
            self.CAOmodel.eval()
            #Proper shipping NAme
            self.PSmodel = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(self.DEVICE)
            self.PSmodel.load_state_dict(torch.load("Models/TrainingModelPS.pth")["state_dict"])
            self.PSmodel.eval()
            #this side up
            self.TSUmodel = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(self.DEVICE)
            self.TSUmodel.load_state_dict(torch.load("Models/TrainingModelTSU.pth")["state_dict"])
            self.TSUmodel.eval()
            #Lithium
            self.Lithiummodel = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(self.DEVICE)
            self.Lithiummodel.load_state_dict(torch.load("Models/TrainingModelLithium.pth")["state_dict"])
            self.Lithiummodel.eval()
            
            
    def _image_segment(self,model,image_np, out_x=1920, out_y=1080):
        assert modeltype in ["Hazard","UN","CAO","PS","TSU","Lithium"], "Modeltype does not exist"
        # Ensure the input is a NumPy array with the correct data type
        assert isinstance(image_np, np.ndarray), "Input must be a numpy.ndarray"
        assert image_np.dtype == np.uint8, "Input array must have dtype 'uint8'"
        assert model != None, "Model not loaded, or mispell of the modeltype"
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
        x = x.unsqueeze(0).to(device=self.DEVICE)
        # Perform a forward pass and create a binary segmentation mask
        with torch.no_grad():
            preds = torch.sigmoid(model)
            preds = (preds > 0.5).float()
        preds = preds.cpu().numpy()
        if preds.shape[0] == 1:
            preds = preds.squeeze(0)
        preds = np.stack([preds] * 3, axis=-1)
        preds = (preds * 255).astype(np.uint8)
        if preds.shape[0] == 1:
            preds = preds.squeeze(0)
        preds = cv2.resize(preds, (out_x,out_y))
        return preds
    
    def locateHazard(self,model,image_np, out_x=1920, out_y=1080):
        model = self.Hazardmodel
        return self._image_segment(model,image_np, out_x=1920, out_y=1080)
    def locateUN(self,model,image_np, out_x=1920, out_y=1080):
        model = self.UNmodel
        return self._image_segment(model,image_np, out_x=1920, out_y=1080)
    def locateCao(self,model,image_np, out_x=1920, out_y=1080):
        model = self.CAOmodel
        return self._image_segment(model,image_np, out_x=1920, out_y=1080)
    def locatePS(self,model,image_np, out_x=1920, out_y=1080):
        model = self.PSmodel
        return self._image_segment(model,image_np, out_x=1920, out_y=1080)
    def locateTSU(self,model,image_np, out_x=1920, out_y=1080):
        model = self.TSUmodel
        return self._image_segment(model,image_np, out_x=1920, out_y=1080)
    def locateLithium(self,model,image_np, out_x=1920, out_y=1080):
        model = self.Lithiummodel
        return self._image_segment(model,image_np, out_x=1920, out_y=1080)
    