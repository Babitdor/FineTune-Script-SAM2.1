import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageMaskPointLoss(nn.Module):
    def __init__(self, weight_dict, focal_alpha=0.25, focal_gamma=2):
        """
        Loss function for images with masks and point prompts.
        Args:
            weight_dict: Dictionary containing weights for different loss components.
            focal_alpha: Alpha for sigmoid focal loss.
            focal_gamma: Gamma for sigmoid focal loss.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, outputs, targets):
        """
        Compute the loss for images, masks, and point prompts.
        Args:
            outputs: Dictionary containing model outputs:
                - "pred_masks": Predicted masks (tensor of shape [N, H, W]).
            targets: Dictionary containing ground truth:
                - "gt_masks": Ground truth masks (tensor of shape [N, H, W]).
                - "positive_points": List of positive point coordinates for each image.
                - "negative_points": List of negative point coordinates for each image.
        Returns:
            Total loss as a scalar tensor.
        """
        pred_masks = outputs["pred_masks"]  # [N, H, W]
        gt_masks = targets["gt_masks"]  # [N, H, W]
        positive_points = targets["positive_points"]  # List of [x, y] coordinates
        negative_points = targets["negative_points"]  # List of [x, y] coordinates

        # Compute mask loss (Dice + Focal Loss)
        loss_mask = self.compute_mask_loss(pred_masks, gt_masks)

        # Compute point prompt loss
        loss_points = self.compute_point_loss(pred_masks, positive_points, negative_points)

        # Combine losses
        total_loss = (
            self.weight_dict["loss_mask"] * loss_mask +
            self.weight_dict["loss_points"] * loss_points
        )
        return total_loss

    def compute_mask_loss(self, pred_masks, gt_masks):
        """
        Compute the loss between predicted masks and ground truth masks.
        Uses Dice loss and Focal loss.
        """
        # Dice loss
        dice_loss = self.dice_loss(pred_masks, gt_masks)

        # Focal loss
        focal_loss = self.sigmoid_focal_loss(pred_masks, gt_masks)

        return dice_loss + focal_loss

    def compute_point_loss(self, pred_masks, positive_points, negative_points):
        """
        Compute the loss for point prompts.
        Args:
            pred_masks: Predicted masks (tensor of shape [N, H, W]).
            positive_points: List of positive point coordinates for each image.
            negative_points: List of negative point coordinates for each image.
        Returns:
            Point prompt loss as a scalar tensor.
        """
        batch_size, height, width = pred_masks.shape
        device = pred_masks.device
        point_loss = 0.0

        for i in range(batch_size):
            # Positive points
            for x, y in positive_points[i]:
                x, y = int(x), int(y)
                point_loss += F.binary_cross_entropy_with_logits(
                    pred_masks[i, y, x], torch.tensor(1.0, device=device)
                )

            # Negative points
            for x, y in negative_points[i]:
                x, y = int(x), int(y)
                point_loss += F.binary_cross_entropy_with_logits(
                    pred_masks[i, y, x], torch.tensor(0.0, device=device)
                )

        return point_loss / batch_size

    def dice_loss(self, inputs, targets):
        """
        Compute the Dice loss.
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(1) + targets.sum(1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()

    def sigmoid_focal_loss(self, inputs, targets):
        """
        Compute the sigmoid focal loss.
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.focal_gamma)

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()