"""
Box utilities for object detection: IoU, GIoU, NMS, encoding/decoding.
"""

import torch
from typing import Tuple


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) boxes in xyxy format
        boxes2: (M, 4) boxes in xyxy format

    Returns:
        (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-6)


def box_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute GIoU between two sets of boxes.

    Args:
        boxes1: (N, 4) boxes in xyxy format
        boxes2: (M, 4) boxes in xyxy format

    Returns:
        (N, M) GIoU matrix
    """
    # Standard IoU
    iou = box_iou(boxes1, boxes2)

    # Compute enclosing box
    enclose_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1]  # (N, M)

    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter

    # GIoU = IoU - (enclose_area - union) / enclose_area
    giou = iou - (enclose_area - union) / (enclose_area + 1e-6)

    return giou


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    Non-Maximum Suppression.

    Args:
        boxes: (N, 4) boxes in xyxy format
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of kept boxes
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    # Sort by score (descending)
    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0].item())
            break

        i = order[0].item()
        keep.append(i)

        # Compute IoU with remaining boxes
        iou = box_iou(boxes[i:i+1], boxes[order[1:]])[0]

        # Keep boxes with IoU below threshold
        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    Batched NMS (NMS per class).

    Args:
        boxes: (N, 4) boxes in xyxy format
        scores: (N,) confidence scores
        labels: (N,) class labels
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of kept boxes
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    # Offset boxes by class to avoid cross-class suppression
    max_coordinate = boxes.max()
    offsets = labels.float() * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]

    return nms(boxes_for_nms, scores, iou_threshold)


def encode_boxes(
    boxes: torch.Tensor,
    locations: torch.Tensor
) -> torch.Tensor:
    """
    Encode boxes as FCOS regression targets (l, t, r, b).

    Args:
        boxes: (N, 4) boxes in xyxy format
        locations: (N, 2) grid locations (x, y)

    Returns:
        (N, 4) encoded targets in (l, t, r, b) format
    """
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    cx, cy = locations.unbind(dim=1)

    l = cx - x1
    t = cy - y1
    r = x2 - cx
    b = y2 - cy

    return torch.stack([l, t, r, b], dim=1)


def decode_boxes(
    reg_pred: torch.Tensor,
    locations: torch.Tensor
) -> torch.Tensor:
    """
    Decode FCOS regression predictions to xyxy boxes.

    Args:
        reg_pred: (N, 4) predictions in (l, t, r, b) format
        locations: (N, 2) grid locations (x, y)

    Returns:
        (N, 4) boxes in xyxy format
    """
    l, t, r, b = reg_pred.unbind(dim=1)
    cx, cy = locations.unbind(dim=1)

    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b

    return torch.stack([x1, y1, x2, y2], dim=1)


def clip_boxes(
    boxes: torch.Tensor,
    image_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Clip boxes to image boundaries.

    Args:
        boxes: (N, 4) boxes in xyxy format
        image_size: (width, height) of image

    Returns:
        Clipped boxes
    """
    w, h = image_size
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(0, w)
    boxes[:, 1] = boxes[:, 1].clamp(0, h)
    boxes[:, 2] = boxes[:, 2].clamp(0, w)
    boxes[:, 3] = boxes[:, 3].clamp(0, h)
    return boxes


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute area of boxes.

    Args:
        boxes: (N, 4) boxes in xyxy format

    Returns:
        (N,) areas
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def remove_small_boxes(
    boxes: torch.Tensor,
    min_size: float
) -> torch.Tensor:
    """
    Remove boxes smaller than min_size.

    Args:
        boxes: (N, 4) boxes in xyxy format
        min_size: Minimum width and height

    Returns:
        Indices of kept boxes
    """
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    return torch.where(keep)[0]


if __name__ == "__main__":
    # Test box utilities
    boxes1 = torch.tensor([
        [0, 0, 10, 10],
        [20, 20, 30, 30],
        [0, 0, 5, 5]
    ], dtype=torch.float32)

    boxes2 = torch.tensor([
        [5, 5, 15, 15],
        [0, 0, 10, 10]
    ], dtype=torch.float32)

    # Test IoU
    iou = box_iou(boxes1, boxes2)
    print(f"IoU matrix:\n{iou}")

    # Test GIoU
    giou = box_giou(boxes1, boxes2)
    print(f"\nGIoU matrix:\n{giou}")

    # Test NMS
    boxes = torch.tensor([
        [0, 0, 10, 10],
        [1, 1, 11, 11],  # overlaps with first
        [20, 20, 30, 30],
        [21, 21, 31, 31]  # overlaps with third
    ], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])

    keep = nms(boxes, scores, iou_threshold=0.5)
    print(f"\nNMS keep indices: {keep}")

    # Test encode/decode
    boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    locations = torch.tensor([[150, 150]], dtype=torch.float32)

    encoded = encode_boxes(boxes, locations)
    print(f"\nEncoded (l,t,r,b): {encoded}")

    decoded = decode_boxes(encoded, locations)
    print(f"Decoded (xyxy): {decoded}")

    # Test clip
    boxes = torch.tensor([[-10, -10, 100, 100]], dtype=torch.float32)
    clipped = clip_boxes(boxes, (50, 50))
    print(f"\nClipped boxes: {clipped}")
