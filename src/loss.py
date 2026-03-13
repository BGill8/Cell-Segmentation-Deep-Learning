import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice

class MultiTaskLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, mse_weight=10.0):
        """
        Combined loss for instance segmentation.
        Args:
            dice_weight: Weight for Dice Loss (Semantic)
            bce_weight: Weight for Binary Cross Entropy (Semantic)
            mse_weight: Weight for Mean Squared Error (Distance Map)
        """
        super(MultiTaskLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        """
        predictions: [Batch, 2, H, W]
        targets: [Batch, 2, H, W]
        """
        # Split channels
        pred_semantic = predictions[:, 0, :, :]
        pred_distance = predictions[:, 1, :, :]

        true_semantic = targets[:, 0, :, :]
        true_distance = targets[:, 1, :, :]

        # 1. Semantic Loss (BCE + Dice)
        # We use both because Dice handles class imbalance (small cells) 
        # while BCE provides smoother gradients early in training.
        loss_bce = self.bce(pred_semantic, true_semantic)
        loss_dice = self.dice(pred_semantic, true_semantic)
        semantic_loss = (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)

        # 2. Distance Map Loss (MSE)
        # We only calculate MSE on the distance map regression
        loss_mse = self.mse(pred_distance, true_distance)

        # Total Loss
        total_loss = semantic_loss + (self.mse_weight * loss_mse)

        return total_loss, {
            "bce": loss_bce.item(),
            "dice": loss_dice.item(),
            "mse": loss_mse.item(),
            "total": total_loss.item()
        }