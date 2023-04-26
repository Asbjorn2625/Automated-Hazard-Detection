import torch
import torchvision
from utils.model import UNET
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(1, 2, 3)) + self.eps
        den = (probs + targets).sum(dim=(1, 2, 3)) + self.eps
        dice_coeff = num / den
        return 1 - dice_coeff.mean()
