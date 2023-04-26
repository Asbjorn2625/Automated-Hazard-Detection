import os
from PIL import Image
import torchvision.transforms as T
import torch
import torchvision.utils
import numpy as np
import albumentations
import cv2
from albumentations.pytorch import ToTensorV2
import torch.utils
from utils.model import UNET

HEIGHT = int(562)
WIDTH = int(562)
m = "Labels.pth"
DEVICE = "cuda"
model = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(DEVICE)
print("loading model")
model.load_state_dict(torch.load(m)["state_dict"])


def save_single_image_prediction(image_path, model, folder="saved_images/", device="cuda"):
    # Create the output folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Load the image and apply the necessary transformations
    image = Image.open(image_path)
    transform = T.Compose([
        T.Resize((HEIGHT, WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ])
    x = transform(image)

    # Prepare the image tensor for the model
    x = x.unsqueeze(0).to(device=device)

    # Set the model to evaluation mode
    model.eval()

    # Perform a forward pass and create a binary segmentation mask
    with torch.no_grad():
        preds = torch.sigmoid(model(x))
        preds = (preds > 0.9).float()

    # Save the predicted mask and input image
    filename = os.path.splitext(os.path.basename(image_path))[0]
    torchvision.utils.save_image(preds, f"{folder}/{filename}_pred.png")
    torchvision.utils.save_image(x, f"{folder}/{filename}_original.png")

    preds = preds.cpu().numpy()
    if preds.shape[0] == 1:
        preds = preds.squeeze(0)
    preds = np.stack([preds] * 3, axis=-1)
    preds = (preds * 255).astype(np.uint8)
    if preds.shape[0] == 1:
        preds = preds.squeeze(0)
    print("Shape of preds_np_rgb:", preds.shape)

    return preds


image_path = "rgb_image_0102.png"
preds_np_rgb = save_single_image_prediction(image_path, model , folder="yobo", device="cuda")

pred = preds_np_rgb[:, :, 0]
contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img = cv2.imread(image_path)
img = cv2.resize(img, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)
cv2.drawContours(img, contours, -1, (0, 255, 255), 1)
result = cv2.bitwise_and(img, img, mask=pred)

display = np.hstack((result, img))
cv2.namedWindow('result', cv2.WINDOW_AUTOSIZE)
cv2.imshow("result", display)
cv2.waitKey(0)
cv2.imwrite("cutOut_2.png", result)
