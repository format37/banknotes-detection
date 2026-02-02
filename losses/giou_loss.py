"""
Generalized IoU (GIoU) Loss implementation for bounding box regression.
"""

import torch
import torch.nn as nn


class GIoULoss(nn.Module):
    """
    Generalized Intersection over Union Loss.

    GIoU = IoU - |C \ (A ∪ B)| / |C|

    Where:
        - IoU = |A ∩ B| / |A ∪ B|
        - C = smallest enclosing box of A and B

    Loss = 1 - GIoU

    GIoU has better gradient properties for non-overlapping boxes
    compared to standard IoU loss.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute GIoU loss.

        Args:
            pred_boxes: Predicted boxes (N, 4) in xyxy format
            target_boxes: Target boxes (N, 4) in xyxy format
            weights: Optional weights per box (N,)

        Returns:
            GIoU loss value
        """
        if pred_boxes.numel() == 0:
            return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)

        giou = self._compute_giou(pred_boxes, target_boxes)
        loss = 1 - giou

        if weights is not None:
            loss = loss * weights

        if self.reduction == 'mean':
            if weights is not None:
                return loss.sum() / weights.sum().clamp(min=1)
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def _compute_giou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GIoU between pairs of boxes.

        Args:
            boxes1: (N, 4) boxes in xyxy format
            boxes2: (N, 4) boxes in xyxy format

        Returns:
            (N,) GIoU values in range [-1, 1]
        """
        # Ensure valid boxes (x2 > x1, y2 > y1)
        boxes1 = self._ensure_valid_boxes(boxes1)
        boxes2 = self._ensure_valid_boxes(boxes2)

        # Compute areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Compute intersection
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # left-top
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # right-bottom

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]

        # Compute union
        union = area1 + area2 - inter

        # Compute IoU
        iou = inter / (union + 1e-7)

        # Compute enclosing box
        enclose_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
        enclose_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

        enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]

        # Compute GIoU
        giou = iou - (enclose_area - union) / (enclose_area + 1e-7)

        return giou

    def _ensure_valid_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Ensure boxes have x2 > x1 and y2 > y1."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Ensure ordering
        x1_new = torch.min(x1, x2)
        x2_new = torch.max(x1, x2)
        y1_new = torch.min(y1, y2)
        y2_new = torch.max(y1, y2)

        return torch.stack([x1_new, y1_new, x2_new, y2_new], dim=1)


def compute_giou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor
) -> torch.Tensor:
    """
    Compute GIoU between pairs of boxes (standalone function).

    Args:
        boxes1: (N, 4) boxes in xyxy format
        boxes2: (N, 4) boxes in xyxy format

    Returns:
        (N,) GIoU values in range [-1, 1]
    """
    loss_fn = GIoULoss(reduction='none')
    return 1 - loss_fn(boxes1, boxes2)


if __name__ == "__main__":
    # Test GIoU loss
    loss_fn = GIoULoss(reduction='mean')

    # Test with overlapping boxes
    pred = torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=torch.float32)
    target = torch.tensor([[5, 5, 15, 15], [0, 0, 10, 10]], dtype=torch.float32)

    loss = loss_fn(pred, target)
    print(f"Overlapping boxes loss: {loss.item():.4f}")

    # Test with non-overlapping boxes
    pred = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    target = torch.tensor([[20, 20, 30, 30]], dtype=torch.float32)

    loss = loss_fn(pred, target)
    print(f"Non-overlapping boxes loss: {loss.item():.4f}")

    # Test with perfect match
    pred = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    target = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)

    loss = loss_fn(pred, target)
    print(f"Perfect match loss: {loss.item():.4f}")

    # Test gradient flow
    pred.requires_grad = True
    loss = loss_fn(pred, target)
    loss.backward()
    print(f"Gradient computed: {pred.grad is not None}")

    # Test compute_giou function
    giou = compute_giou(
        torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
        torch.tensor([[5, 5, 15, 15]], dtype=torch.float32)
    )
    print(f"GIoU value: {giou.item():.4f}")
