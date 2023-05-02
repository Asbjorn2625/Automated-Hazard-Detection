import torch
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

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.alpha * self.dice_loss(pred, target) + (1 - self.alpha) * self.bce_loss(pred, target)

# Loss model for used to focus on hard examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # Convert the target to one-hot encoding
        target_one_hot = torch.zeros_like(input)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)

        # Calculate the probability of each class using the softmax function
        prob = torch.nn.functional.softmax(input, dim=1)

        # Calculate the focal loss
        focal_weight = (1.0 - prob) ** self.gamma
        focal_loss = -self.alpha * focal_weight * target_one_hot * torch.log(prob + 1e-6)

        # Reduce the loss based on the reduction method
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        else:
            raise ValueError("Invalid reduction method. Choose 'mean' or 'sum'.")

        return focal_loss