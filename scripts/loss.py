import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, score_loss_weight=0.05):
        super().__init__()
        self.score_loss_weight = score_loss_weight

    def cross_entropy_loss(self, pred_mask, gt_mask):
        return (
            -gt_mask * torch.log(pred_mask + 1e-5)
            - (1 - gt_mask) * torch.log((1 - pred_mask) + 1e-5)
        ).mean()

    def iou_score(self, pred_mask, gt_mask):
        # Using threshold of 0.5 for binary prediction
        pred_binary = (pred_mask > 0.5).float()
        inter = (gt_mask * pred_binary).flatten(1).sum(dim=1)
        union = (
            gt_mask.flatten(1).sum(dim=1) + pred_binary.flatten(1).sum(dim=1) - inter
        )
        return inter / (union + 1e-6)

    def score_loss(self, pred_scores, actual_iou):
        return torch.abs(pred_scores[:, 0] - actual_iou).mean()

    def forward(self, pred_mask, gt_mask, pred_scores):
        assert (
            pred_mask.shape == gt_mask.shape
        ), "Shape mismatch between prediction and target"

        # Calculate losses
        ce_loss = self.cross_entropy_loss(pred_mask, gt_mask)
        iou = self.iou_score(pred_mask, gt_mask).detach()
        sc_loss = self.score_loss(pred_scores, iou)

        # Combine losses
        total_loss = ce_loss + sc_loss * self.score_loss_weight

        return total_loss, iou
