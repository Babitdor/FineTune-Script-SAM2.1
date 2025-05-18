import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        weights=(0.5, 0.3, 0.2),
        smooth=1e-6,
        alpha: float = 0.25,
        gamma: float = 2,
    ):
        super().__init__()
        self.weights = weights
        self.smooth = smooth
        self.focal_alpha = alpha
        self.focal_gamma = gamma

    def dice_loss(self, pred_mask, gt_mask, smooth=1e-6):

        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)
        return dice_loss

    def iou_score(self, prd_mask, gt_mask):
        # Score Loss calculation (IOU)
        pred_binary = (prd_mask > 0.5).float()
        inter = (gt_mask * pred_binary).flatten(1).sum(dim=1)
        union = gt_mask.flatten(1).sum(dim=1) + pred_binary.flatten(1).sum(dim=1)
        iou = inter / (union - inter + 1e-6)
        return iou

    def score_loss(self, pred_score, actual_iou):
        return torch.abs(pred_score - actual_iou).mean()

    def sigmoid_focal_loss(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.focal_gamma)

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (
                1 - targets
            )
            loss = alpha_t * loss

        return loss.mean()

    def forward(self, pred_mask, gt_mask, pred_scores):
        focal = self.sigmoid_focal_loss(pred_mask, gt_mask)
        dice = self.dice_loss(torch.sigmoid(pred_mask), gt_mask)
        iou = self.iou_score(pred_mask.sigmoid(), gt_mask).detach()
        score = self.score_loss(torch.sigmoid(pred_scores), iou)
        total = (
            focal * self.weights[0] + dice * self.weights[1] + score * self.weights[2]
        )
        return total, focal, dice, score, iou
