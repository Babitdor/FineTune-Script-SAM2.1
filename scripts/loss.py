import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        weights=(0.3, 0.5, 0.2),
        smooth=1e-6,
        alpha: float = 0.1,
        gamma: float = 2,
        score_loss_type="combo",
    ):
        super().__init__()
        self.weights = weights
        self.smooth = smooth
        self.focal_alpha = alpha
        self.focal_gamma = gamma
        self.score_loss_type = score_loss_type

    # def dice_loss(self, pred_mask, gt_mask, smooth=1e-6):
    #     # Normal Dice Loss calculation
    #     intersection = (pred_mask * gt_mask).sum()
    #     union = pred_mask.sum() + gt_mask.sum()
    #     dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)
    #     return dice_loss

    def generalized_dice_loss(self, pred_mask, gt_mask, smooth=1e-6):
        w = 1.0 / (gt_mask.flatten(1).sum(dim=1) ** 2 + smooth)
        numerator = (w * (pred_mask * gt_mask).flatten(1).sum(dim=1)).sum()
        denominator = (w * (pred_mask + gt_mask).flatten(1).sum(dim=1)).sum()
        return 1 - (2.0 * numerator + smooth) / (denominator + smooth)

    def iou_score(self, prd_mask, gt_mask, epsilon=1e-6):
        # Score Loss calculation (IOU)
        pred_binary = (prd_mask > 0.5).float()
        inter = (gt_mask * pred_binary).flatten(1).sum(dim=1)
        union = (
            gt_mask.flatten(1).sum(dim=1) + pred_binary.flatten(1).sum(dim=1) - inter
        )
        iou = (inter + epsilon) / (union + epsilon)
        return iou

    def score_loss(self, pred_score, actual_iou):
        pred_score = pred_score.squeeze()  # Flatten to 1D if needed
        actual_iou = actual_iou.squeeze()  # Flatten to 1D

        if self.score_loss_type == "l1":
            return F.l1_loss(pred_score, actual_iou)
        elif self.score_loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_score, actual_iou)
        else:  # combo loss
            return 0.7 * F.l1_loss(pred_score, actual_iou) + 0.3 * F.mse_loss(
                pred_score, actual_iou
            )

    def sigmoid_focal_loss(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.focal_gamma)

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (
                1 - targets
            )
            loss = alpha_t * loss

        return loss.mean(1).mean()

    def forward(self, pred_mask, gt_mask, pred_scores):

        assert (
            pred_mask.shape == gt_mask.shape
        ), "Shape mismatch between prediction and target"

        focal = self.sigmoid_focal_loss(pred_mask, gt_mask)
        dice = self.generalized_dice_loss(pred_mask, gt_mask)
        iou = self.iou_score(pred_mask, gt_mask, epsilon=1e-6).detach()
        score = self.score_loss(torch.sigmoid(pred_scores), iou)
        total = (
            focal * self.weights[0] + dice * self.weights[1] + score * self.weights[2]
        )
        return total, iou
