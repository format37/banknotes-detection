"""
Combined FCOS loss with classification, regression, and centerness losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

from .focal_loss import FocalLoss
from .giou_loss import GIoULoss


class FCOSLoss(nn.Module):
    """
    Combined loss function for FCOS.

    Total Loss = L_cls + λ_reg * L_reg + λ_ctr * L_ctr

    Where:
        - L_cls: Focal loss for classification
        - L_reg: GIoU loss for bounding box regression
        - L_ctr: Binary cross-entropy for centerness
    """

    def __init__(
        self,
        num_classes: int = 24,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        lambda_reg: float = 1.0,
        lambda_ctr: float = 1.0,
        strides: List[int] = None
    ):
        """
        Args:
            num_classes: Number of object classes
            focal_alpha: Alpha for focal loss
            focal_gamma: Gamma for focal loss
            lambda_reg: Weight for regression loss
            lambda_ctr: Weight for centerness loss
            strides: FPN strides for each level
        """
        super().__init__()

        self.num_classes = num_classes
        self.lambda_reg = lambda_reg
        self.lambda_ctr = lambda_ctr
        self.strides = strides or [8, 16, 32, 64, 128]

        # Loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.giou_loss = GIoULoss(reduction='sum')

    def forward(
        self,
        cls_outputs: List[torch.Tensor],
        reg_outputs: List[torch.Tensor],
        ctr_outputs: List[torch.Tensor],
        targets: List[Dict],
        compute_fcos_targets_fn
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute FCOS loss.

        Args:
            cls_outputs: List of (B, C, Hi, Wi) classification predictions
            reg_outputs: List of (B, 4, Hi, Wi) regression predictions (l, t, r, b)
            ctr_outputs: List of (B, 1, Hi, Wi) centerness predictions
            targets: List of target dicts per image with 'boxes' and 'labels'
            compute_fcos_targets_fn: Function to compute FCOS targets from boxes/labels

        Returns:
            Tuple of (total_loss, cls_loss, reg_loss, ctr_loss)
        """
        device = cls_outputs[0].device
        batch_size = cls_outputs[0].shape[0]

        # Collect all losses
        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        total_ctr_loss = torch.tensor(0.0, device=device)
        total_num_pos = 0

        for batch_idx in range(batch_size):
            # Get targets for this image
            boxes = targets[batch_idx]['boxes'].to(device)
            labels = targets[batch_idx]['labels'].to(device)

            # Compute FCOS targets
            fcos_targets = compute_fcos_targets_fn(boxes, labels, device)

            for level_idx, stride in enumerate(self.strides):
                # Get predictions for this level
                cls_pred = cls_outputs[level_idx][batch_idx]  # (C, H, W)
                reg_pred = reg_outputs[level_idx][batch_idx]  # (4, H, W)
                ctr_pred = ctr_outputs[level_idx][batch_idx]  # (1, H, W)

                # Get targets for this level
                cls_target = fcos_targets[level_idx]['cls_targets']  # (H, W)
                reg_target = fcos_targets[level_idx]['reg_targets']  # (H, W, 4)
                ctr_target = fcos_targets[level_idx]['ctr_targets']  # (H, W)

                h, w = cls_pred.shape[1:]

                # Flatten predictions
                cls_pred_flat = cls_pred.permute(1, 2, 0).reshape(-1, self.num_classes)
                reg_pred_flat = reg_pred.permute(1, 2, 0).reshape(-1, 4)
                ctr_pred_flat = ctr_pred.reshape(-1)

                # Flatten targets
                cls_target_flat = cls_target.reshape(-1)
                reg_target_flat = reg_target.reshape(-1, 4)
                ctr_target_flat = ctr_target.reshape(-1)

                # Positive mask (locations assigned to objects)
                pos_mask = cls_target_flat > 0
                num_pos = pos_mask.sum().item()
                total_num_pos += num_pos

                # Classification loss (for all locations)
                cls_loss = self._compute_cls_loss(
                    cls_pred_flat, cls_target_flat
                )
                total_cls_loss = total_cls_loss + cls_loss

                if num_pos > 0:
                    # Regression loss (only for positive locations)
                    pos_reg_pred = reg_pred_flat[pos_mask]
                    pos_reg_target = reg_target_flat[pos_mask]

                    # Create grid locations for decoding
                    shifts_x = torch.arange(0, w, device=device) * stride + stride // 2
                    shifts_y = torch.arange(0, h, device=device) * stride + stride // 2
                    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
                    locations = torch.stack([shift_x, shift_y], dim=-1).reshape(-1, 2)
                    pos_locations = locations[pos_mask]

                    # Decode predicted and target boxes
                    pred_boxes = self._decode_boxes(pos_reg_pred, pos_locations)
                    target_boxes = self._decode_boxes(pos_reg_target, pos_locations)

                    reg_loss = self.giou_loss(pred_boxes, target_boxes)
                    total_reg_loss = total_reg_loss + reg_loss

                    # Centerness loss (only for positive locations)
                    pos_ctr_pred = ctr_pred_flat[pos_mask]
                    pos_ctr_target = ctr_target_flat[pos_mask]

                    ctr_loss = F.binary_cross_entropy_with_logits(
                        pos_ctr_pred, pos_ctr_target, reduction='sum'
                    )
                    total_ctr_loss = total_ctr_loss + ctr_loss

        # Normalize by total number of positive samples
        num_pos_avg = max(total_num_pos, 1)

        total_cls_loss = total_cls_loss / num_pos_avg
        total_reg_loss = total_reg_loss / num_pos_avg
        total_ctr_loss = total_ctr_loss / num_pos_avg

        # Total loss
        total_loss = (
            total_cls_loss +
            self.lambda_reg * total_reg_loss +
            self.lambda_ctr * total_ctr_loss
        )

        return total_loss, total_cls_loss, total_reg_loss, total_ctr_loss

    def _compute_cls_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification loss using focal loss.

        Args:
            pred: (N, C) predicted logits
            target: (N,) target class indices (0=bg, 1-C=classes)

        Returns:
            Classification loss (sum, not averaged)
        """
        num_classes = pred.shape[1]
        device = pred.device

        # Mask for valid locations (not ignored)
        valid_mask = target >= 0

        if not valid_mask.any():
            return torch.tensor(0.0, device=device)

        valid_pred = pred[valid_mask]
        valid_target = target[valid_mask]

        # Convert to probabilities
        p = valid_pred.sigmoid()

        # Create one-hot target (0=bg, 1-C=classes -> 0-C-1 for one-hot)
        # Background (0) means all zeros in one-hot
        target_one_hot = torch.zeros_like(valid_pred)
        pos_mask = valid_target > 0
        if pos_mask.any():
            pos_indices = valid_target[pos_mask] - 1  # Convert to 0-indexed
            target_one_hot[pos_mask] = F.one_hot(
                pos_indices, num_classes
            ).float()

        # Compute focal loss
        alpha = 0.25
        gamma = 2.0

        p_t = p * target_one_hot + (1 - p) * (1 - target_one_hot)
        alpha_t = alpha * target_one_hot + (1 - alpha) * (1 - target_one_hot)
        focal_weight = (1 - p_t) ** gamma

        bce = F.binary_cross_entropy_with_logits(
            valid_pred, target_one_hot, reduction='none'
        )

        loss = (alpha_t * focal_weight * bce).sum()

        return loss

    def _decode_boxes(
        self,
        reg_pred: torch.Tensor,
        locations: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode regression predictions to xyxy boxes.

        Args:
            reg_pred: (N, 4) predictions in (l, t, r, b) format
            locations: (N, 2) grid locations (x, y)

        Returns:
            (N, 4) boxes in xyxy format
        """
        l, t, r, b = reg_pred.unbind(dim=1)
        x, y = locations.unbind(dim=1)

        x1 = x - l
        y1 = y - t
        x2 = x + r
        y2 = y + b

        return torch.stack([x1, y1, x2, y2], dim=1)


def compute_centerness(reg_target: torch.Tensor) -> torch.Tensor:
    """
    Compute centerness from regression targets.

    centerness = sqrt((min(l,r) / max(l,r)) * (min(t,b) / max(t,b)))

    Args:
        reg_target: (N, 4) or (H, W, 4) in (l, t, r, b) format

    Returns:
        Centerness values in same shape as input (without last dim)
    """
    if reg_target.dim() == 2:
        l, t, r, b = reg_target.unbind(dim=1)
    else:
        l = reg_target[..., 0]
        t = reg_target[..., 1]
        r = reg_target[..., 2]
        b = reg_target[..., 3]

    lr_min = torch.min(l, r)
    lr_max = torch.max(l, r)
    tb_min = torch.min(t, b)
    tb_max = torch.max(t, b)

    centerness = torch.sqrt(
        (lr_min / (lr_max + 1e-6)) * (tb_min / (tb_max + 1e-6))
    )

    return centerness


if __name__ == "__main__":
    # Test FCOS loss
    from dataset import BanknoteDataset

    # Create dummy predictions
    batch_size = 2
    num_classes = 24
    strides = [8, 16, 32, 64, 128]
    h, w = 1080, 1920

    cls_outputs = []
    reg_outputs = []
    ctr_outputs = []

    for stride in strides:
        fh, fw = h // stride, w // stride
        cls_outputs.append(torch.randn(batch_size, num_classes, fh, fw))
        reg_outputs.append(torch.randn(batch_size, 4, fh, fw).abs())  # positive values
        ctr_outputs.append(torch.randn(batch_size, 1, fh, fw))

    # Create dummy targets
    targets = [
        {
            'boxes': torch.tensor([[100, 100, 200, 200], [300, 300, 500, 500]], dtype=torch.float32),
            'labels': torch.tensor([1, 5], dtype=torch.long)
        },
        {
            'boxes': torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
            'labels': torch.tensor([3], dtype=torch.long)
        }
    ]

    # Create dataset for target computation
    class DummyDataset:
        def __init__(self):
            self.strides = strides
            self.image_size = (w, h)
            self.size_ranges = [
                (0, 64), (64, 128), (128, 256), (256, 512), (512, float('inf'))
            ]

        def compute_fcos_targets(self, boxes, labels, device):
            from dataset import BanknoteDataset
            ds = BanknoteDataset.__new__(BanknoteDataset)
            ds.strides = self.strides
            ds.image_size = self.image_size
            ds.size_ranges = self.size_ranges
            return ds.compute_fcos_targets(boxes, labels, device)

    dummy_ds = DummyDataset()

    # Test loss computation
    loss_fn = FCOSLoss(num_classes=num_classes, strides=strides)

    total_loss, cls_loss, reg_loss, ctr_loss = loss_fn(
        cls_outputs, reg_outputs, ctr_outputs, targets,
        dummy_ds.compute_fcos_targets
    )

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Classification loss: {cls_loss.item():.4f}")
    print(f"Regression loss: {reg_loss.item():.4f}")
    print(f"Centerness loss: {ctr_loss.item():.4f}")

    # Test gradient flow
    for outputs in [cls_outputs, reg_outputs, ctr_outputs]:
        for o in outputs:
            o.requires_grad = True

    total_loss, _, _, _ = loss_fn(
        cls_outputs, reg_outputs, ctr_outputs, targets,
        dummy_ds.compute_fcos_targets
    )
    total_loss.backward()
    print(f"Gradient computed: {cls_outputs[0].grad is not None}")
