import torch
import sys
sys.path.append('/workspaces/P6-Automated-Hazard-Detection')
sys.path.append('/workspaces/Automated-Hazard-Detection')
from Libs.final_unet.utils.model import UNET
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

class IoULoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = (probs * targets).sum(dim=(1, 2, 3)) + self.eps
        den = (probs + targets).sum(dim=(1, 2, 3)) - num + self.eps
        iou = num / den
        return 1 - iou.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.alpha * self.dice_loss(pred, target) + (1 - self.alpha) * self.bce_loss(pred, target)