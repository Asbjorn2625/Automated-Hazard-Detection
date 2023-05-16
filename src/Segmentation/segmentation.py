from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
import cv2
import torch.utils
import sys
import os
sys.path.append("/workspaces/P6-Automated-Hazard-Detection")
from Libs.final_unet.utils.model import UNET

class Segmentation:
    def __init__(self,model_type="Dice_Loss", DEVICE="cuda" if torch.cuda.is_available() else "cpu"):
        self.DEVICE = DEVICE
        self.Hazardmodel= None
        self.UNmodel= None
        self.CAOmodel= None
        self.PSmodel= None
        self.TSUmodel= None
        self.Lithiummodel= None
        self._load_models(model_type=model_type)

    def _load_models(self, model_type=None):
            # Load all the model and set it to evaluation mode
            dir_folder = os.path.dirname(os.path.abspath(__file__))
            #First up is the Hazard model
            self.Hazardmodel = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(self.DEVICE)
            if model_type != None:
                self.Hazardmodel.load_state_dict(torch.load(os.path.join(dir_folder, f"models/UNET_hazard{model_type}.pth"))["state_dict"])
            else:
                self.Hazardmodel.load_state_dict(torch.load(os.path.join(dir_folder, f"models/UNET_hazardCombinedLoss.pth"))["state_dict"])
            self.Hazardmodel.eval()
            #Next up is the UN model
            self.UNmodel = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(self.DEVICE)
            if model_type != None:
                self.UNmodel.load_state_dict(torch.load(os.path.join(dir_folder, f"models/UNET_UN_circle{model_type}.pth"))["state_dict"])
            else:
                self.UNmodel.load_state_dict(torch.load(os.path.join(dir_folder, f"models/UNET_UN_circleCrossEntropy.pth"))["state_dict"])
            self.UNmodel.eval()
            #CAO model
            self.CAOmodel = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(self.DEVICE)
            if model_type != None:
                self.CAOmodel.load_state_dict(torch.load(os.path.join(dir_folder, f"models/UNET_CAO{model_type}.pth"))["state_dict"])
            else:
                self.CAOmodel.load_state_dict(torch.load(os.path.join(dir_folder, f"models/UNET_CAOCrossEntropy.pth"))["state_dict"])
            self.CAOmodel.eval()
            #Proper shipping NAme
            self.PSmodel = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(self.DEVICE)
            if model_type != None:
                self.PSmodel.load_state_dict(torch.load(os.path.join(dir_folder, f"models/UNET_Shipping{model_type}.pth"))["state_dict"])
            else:
                self.PSmodel.load_state_dict(torch.load(os.path.join(dir_folder, f"models/UNET_ShippingCrossEntropy.pth"))["state_dict"])
            self.PSmodel.eval()
            #this side up
            self.TSUmodel = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(self.DEVICE)
            if model_type != None:
                self.TSUmodel.load_state_dict(torch.load(os.path.join(dir_folder, f"models/UNET_this_side_up{model_type}.pth"))["state_dict"])
            else:
                self.TSUmodel.load_state_dict(torch.load(os.path.join(dir_folder, f"models/UNET_this_side_upDice_loss.pth"))["state_dict"])
            self.TSUmodel.eval()
            #Lithium
            self.Lithiummodel = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(self.DEVICE)
            if model_type != None:
                self.Lithiummodel.load_state_dict(torch.load(os.path.join(dir_folder, f"models/UNET_Lithium{model_type}.pth"))["state_dict"])
            else:
                self.Lithiummodel.load_state_dict(torch.load(os.path.join(dir_folder, f"models/UNET_LithiumDice_loss.pth"))["state_dict"])
            self.Lithiummodel.eval()
            print("Models loaded")
            
            
    def _image_segment(self,model,image_np, out_x=1920, out_y=1080):
        # Ensure the input is a NumPy array with the correct data type
        assert isinstance(image_np, np.ndarray), "Input must be a numpy.ndarray"
        assert image_np.dtype == np.uint8, "Input array must have dtype 'uint8'"
        assert model != None, "Model not loaded"
        transform = T.Compose([
            T.Resize((out_y, out_x), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],)
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
            preds = torch.sigmoid(model(x))
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
    
    def locateHazard(self,image_np, out_x=1920, out_y=1080):
        return self._image_segment(self.Hazardmodel, image_np, out_x, out_y)
    def locateUN(self, image_np, out_x=1920, out_y=1080):
        return self._image_segment(self.UNmodel, image_np, out_x, out_y)
    def locateCao(self,image_np, out_x=1920, out_y=1080):
        return self._image_segment(self.CAOmodel, image_np, out_x, out_y)
    def locatePS(self,image_np, out_x=1920, out_y=1080):
        return self._image_segment(self.PSmodel, image_np, out_x, out_y)
    def locateTSU(self,image_np, out_x=1920, out_y=1080):
        return self._image_segment(self.TSUmodel,image_np, out_x, out_y)
    def locateLithium(self,image_np, out_x=1920, out_y=1080):
        return self._image_segment(self.Lithiummodel,image_np, out_x, out_y)
    