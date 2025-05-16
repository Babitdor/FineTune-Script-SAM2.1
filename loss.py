import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, weights=(1.0, 0.3, 0.05), smooth=1e-6):
        super().__init__()
        self.weights = weights
        self.smooth = smooth

    def dice_loss(self, pred_mask, gt_mask, smooth=1e-6):

        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)
        return dice_loss

    def iou_score(self, prd_mask, gt_mask):
        # Score Loss calculation (IOU)
        pred_binary = (prd_mask > 0.5).float()
        inter = (gt_mask * pred_binary).sum()
        union = gt_mask.sum() + pred_binary.sum()
        iou = inter / (union - inter + 1e-6)
        return iou

    def score_loss(self, pred_score, actual_iou):
        return torch.abs(pred_score - actual_iou).mean()

    def forward(self, pred_mask, gt_mask, pred_scores):
        if pred_mask.dim() == 4:
            pred_mask = pred_mask[:, 0]  # Reduce to [B, H, W]

        bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
        dice = self.dice_loss(pred_mask, gt_mask)
        iou = self.iou_score(pred_mask.sigmoid(), gt_mask).detach()  # For score loss
        score = self.score_loss(pred_scores, iou)

        total = self.weights[0] * bce + self.weights[1] * dice + self.weights[2] * score
        return total, bce, dice, score, iou


# class ImageMaskPointLoss(nn.Module):
#     def __init__(self, weight_dict, focal_alpha=0.25, focal_gamma=2):
#         """
#         Loss function for images with masks and point prompts.
#         Args:
#             weight_dict: Dictionary containing weights for different loss components.
#             focal_alpha: Alpha for sigmoid focal loss.
#             focal_gamma: Gamma for sigmoid focal loss.
#         """
#         super().__init__()
#         self.weight_dict = weight_dict
#         self.focal_alpha = focal_alpha
#         self.focal_gamma = focal_gamma

#     def forward(self, outputs, targets):
#         """
#         Compute the loss for images, masks, and point prompts.
#         Args:
#             outputs: Dictionary containing model outputs:
#                 - "pred_masks": Predicted masks (tensor of shape [N, H, W]).
#             targets: Dictionary containing ground truth:
#                 - "gt_masks": Ground truth masks (tensor of shape [N, H, W]).
#                 - "positive_points": List of positive point coordinates for each image.
#                 - "negative_points": List of negative point coordinates for each image.
#         Returns:
#             Total loss as a scalar tensor.
#         """
#         pred_masks = outputs["pred_masks"]  # [N, H, W]
#         gt_masks = targets["gt_masks"]  # [N, H, W]
#         positive_points = targets["positive_points"]  # List of [x, y] coordinates
#         negative_points = targets["negative_points"]  # List of [x, y] coordinates

#         # Compute mask loss (Dice + Focal Loss)
#         loss_mask = self.compute_mask_loss(pred_masks, gt_masks)

#         # Compute point prompt loss
#         loss_points = self.compute_point_loss(
#             pred_masks, positive_points, negative_points
#         )

#         # Combine losses
#         total_loss = (
#             self.weight_dict["loss_mask"] * loss_mask
#             + self.weight_dict["loss_points"] * loss_points
#         )
#         return total_loss

#     def compute_mask_loss(self, pred_masks, gt_masks):
#         """
#         Compute the loss between predicted masks and ground truth masks.
#         Uses Dice loss and Focal loss.
#         """
#         # Dice loss
#         dice_loss = self.dice_loss(pred_masks, gt_masks)

#         # Focal loss
#         focal_loss = self.sigmoid_focal_loss(pred_masks, gt_masks)

#         return dice_loss + focal_loss

#     def compute_point_loss(self, pred_masks, positive_points, negative_points):
#         """
#         Compute the loss for point prompts.
#         Args:
#             pred_masks: Predicted masks (tensor of shape [N, H, W]).
#             positive_points: List of positive point coordinates for each image.
#             negative_points: List of negative point coordinates for each image.
#         Returns:
#             Point prompt loss as a scalar tensor.
#         """
#         batch_size, height, width = pred_masks.shape
#         device = pred_masks.device
#         point_loss = 0.0

#         for i in range(batch_size):
#             # Positive points
#             for x, y in positive_points[i]:
#                 x, y = int(x), int(y)
#                 point_loss += F.binary_cross_entropy_with_logits(
#                     pred_masks[i, y, x], torch.tensor(1.0, device=device)
#                 )

#             # Negative points
#             for x, y in negative_points[i]:
#                 x, y = int(x), int(y)
#                 point_loss += F.binary_cross_entropy_with_logits(
#                     pred_masks[i, y, x], torch.tensor(0.0, device=device)
#                 )

#         return point_loss / batch_size

#     def sigmoid_focal_loss(self, inputs, targets):
#         """
#         Compute the sigmoid focal loss.
#         """
#         prob = inputs.sigmoid()
#         ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
#         p_t = prob * targets + (1 - prob) * (1 - targets)
#         loss = ce_loss * ((1 - p_t) ** self.focal_gamma)

#         if self.focal_alpha >= 0:
#             alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (
#                 1 - targets
#             )
#             loss = alpha_t * loss

#         return loss.mean()

#     def iou_loss(
#         inputs,
#         targets,
#         pred_ious,
#         num_objects,
#         loss_on_multimask=False,
#         use_l1_loss=False,
#     ):
#         """
#         Args:
#             inputs: A float tensor of arbitrary shape.
#                     The predictions for each example.
#             targets: A float tensor with the same shape as inputs. Stores the binary
#                     classification label for each element in inputs
#                     (0 for the negative class and 1 for the positive class).
#             pred_ious: A float tensor containing the predicted IoUs scores per mask
#             num_objects: Number of objects in the batch
#             loss_on_multimask: True if multimask prediction is enabled
#             use_l1_loss: Whether to use L1 loss is used instead of MSE loss
#         Returns:
#             IoU loss tensor
#         """
#         assert inputs.dim() == 4 and targets.dim() == 4
#         pred_mask = inputs.flatten(2) > 0
#         gt_mask = targets.flatten(2) > 0
#         area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
#         area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
#         actual_ious = area_i / torch.clamp(area_u, min=1.0)

#         if use_l1_loss:
#             loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
#         else:
#             loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
#         if loss_on_multimask:
#             return loss / num_objects
#         return loss.sum() / num_objects

# Segmentaion Loss caclulation
# gt_mask = torch.from_numpy(mask.astype(np.float32)).cuda().unsqueeze(0)
# prd_mask = torch.sigmoid(prd_masks[:, 0])

#     seg_loss = (
#         -gt_mask * torch.log(prd_mask + 1e-6)
#         - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-6)
#     ).mean()

#     # Score Loss calculation (IOU)
#     inter = (gt_mask * (prd_mask > 0.5)).sum()
#     iou = inter / (gt_mask.sum() + (prd_mask > 0.5).sum() - inter + 1e-6)
#     score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

#     # Dice Loss
#     smooth = 1e-6
#     intersection = (prd_mask * gt_mask).sum()
#     dice_loss = 1 - (2.0 * intersection + smooth) / (
#         prd_mask.sum() + gt_mask.sum() + smooth
#     )
#     loss = seg_loss + 0.3 * dice_loss + 0.05 * score_loss
