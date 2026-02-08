"""
Focal Loss implementation for handling class imbalance in object detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection.

    Focal Loss = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
        - p_t = p if y=1 else (1-p)
        - α_t = α if y=1 else (1-α)

    This down-weights easy examples and focuses on hard negatives.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        class_weights: torch.Tensor = None
    ):
        """
        Args:
            alpha: Weighting factor for positive class (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean', 'sum', or 'none'
            class_weights: Optional per-class weights of shape (num_classes,)
                          for handling class imbalance
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('class_weights', class_weights)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits of shape (N, C) or (N, C, H, W)
            targets: Ground truth class indices of shape (N,) or (N, H, W)
                     Values: 0 = background, 1-C = classes, -1 = ignore

        Returns:
            Focal loss value
        """
        # Flatten inputs if needed
        if inputs.dim() == 4:
            # (N, C, H, W) -> (N*H*W, C)
            n, c, h, w = inputs.shape
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, c)
            targets = targets.reshape(-1)
        elif inputs.dim() == 2:
            pass
        else:
            raise ValueError(f"Unexpected input shape: {inputs.shape}")

        num_classes = inputs.shape[1]

        # Create mask for valid targets (not ignored)
        valid_mask = targets >= 0
        valid_inputs = inputs[valid_mask]
        valid_targets = targets[valid_mask]

        if valid_inputs.numel() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # Convert to probabilities
        p = valid_inputs.sigmoid()

        # Create one-hot encoding (including background class 0)
        # Note: targets are 0 for background, 1-C for classes
        target_one_hot = F.one_hot(valid_targets, num_classes + 1)
        # Remove background class from one-hot (we only predict foreground)
        # Use same dtype as inputs for AMP compatibility
        target_one_hot = target_one_hot[:, 1:].to(dtype=valid_inputs.dtype)  # (N, C)

        # Compute focal loss for all classes
        # p_t = p if target=1, else (1-p)
        p_t = p * target_one_hot + (1 - p) * (1 - target_one_hot)

        # α_t = α if target=1, else (1-α)
        alpha_t = self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot)

        # Focal weight: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.gamma

        # Binary cross-entropy loss per class
        bce = F.binary_cross_entropy_with_logits(
            valid_inputs, target_one_hot, reduction='none'
        )

        # Apply focal weighting
        loss = alpha_t * focal_weight * bce

        # Apply class weights if provided
        if self.class_weights is not None:
            # Get class weight for each sample (targets are 0=background, 1-C=classes)
            # For background (target=0), use weight of 1.0
            class_weight_per_sample = torch.ones(valid_targets.shape[0], device=loss.device)
            positive_mask = valid_targets > 0
            if positive_mask.any():
                # Map target indices (1-C) to class indices (0-C-1)
                class_indices = valid_targets[positive_mask] - 1
                class_weight_per_sample[positive_mask] = self.class_weights[class_indices]
            # Expand to match loss shape (N, C) and apply
            class_weight_per_sample = class_weight_per_sample.unsqueeze(1).expand_as(loss)
            loss = loss * class_weight_per_sample

        # Sum over classes, then reduce
        loss = loss.sum(dim=1)

        if self.reduction == 'mean':
            # Normalize by number of positive samples
            num_pos = (valid_targets > 0).sum().float()
            return loss.sum() / num_pos.clamp(min=1)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    # Test focal loss
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # Test with 2D input
    batch_size = 4
    num_classes = 24
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(-1, num_classes + 1, (batch_size,))

    loss = loss_fn(logits, targets)
    print(f"2D input loss: {loss.item():.4f}")

    # Test with 4D input (feature map)
    h, w = 135, 240
    logits = torch.randn(batch_size, num_classes, h, w)
    targets = torch.randint(-1, num_classes + 1, (batch_size, h, w))

    loss = loss_fn(logits, targets)
    print(f"4D input loss: {loss.item():.4f}")

    # Test gradient flow
    logits.requires_grad = True
    loss = loss_fn(logits, targets)
    loss.backward()
    print(f"Gradient computed: {logits.grad is not None}")
