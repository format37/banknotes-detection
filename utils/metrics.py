"""
Metrics computation for object detection: AP and mAP.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from .box_utils import box_iou


def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray
) -> float:
    """
    Compute Average Precision using 11-point interpolation (PASCAL VOC style).

    Args:
        recalls: Array of recall values (sorted)
        precisions: Array of precision values

    Returns:
        Average Precision value
    """
    # Add sentinel values
    recalls = np.concatenate([[0.], recalls, [1.]])
    precisions = np.concatenate([[0.], precisions, [0.]])

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Find points where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0]

    # Compute area under curve
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap


def compute_ap_per_class(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    pred_labels: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict[int, float]:
    """
    Compute AP for each class.

    Args:
        pred_boxes: List of predicted boxes per image
        pred_scores: List of predicted scores per image
        pred_labels: List of predicted labels per image
        gt_boxes: List of ground truth boxes per image
        gt_labels: List of ground truth labels per image
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching

    Returns:
        Dict mapping class index to AP
    """
    aps = {}

    for class_idx in range(num_classes):
        # Collect all predictions and ground truths for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_image_ids = []

        all_gt_boxes = []
        all_gt_image_ids = []

        for img_idx in range(len(pred_boxes)):
            # Predictions
            pred_mask = pred_labels[img_idx] == class_idx
            if pred_mask.sum() > 0:
                all_pred_boxes.append(pred_boxes[img_idx][pred_mask])
                all_pred_scores.append(pred_scores[img_idx][pred_mask])
                all_pred_image_ids.extend([img_idx] * pred_mask.sum().item())

            # Ground truths
            gt_mask = gt_labels[img_idx] == class_idx
            if gt_mask.sum() > 0:
                all_gt_boxes.append(gt_boxes[img_idx][gt_mask])
                all_gt_image_ids.extend([img_idx] * gt_mask.sum().item())

        if len(all_gt_boxes) == 0:
            # No ground truth for this class
            aps[class_idx] = 0.0
            continue

        if len(all_pred_boxes) == 0:
            # No predictions for this class
            aps[class_idx] = 0.0
            continue

        # Concatenate
        all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
        all_pred_scores = torch.cat(all_pred_scores, dim=0)
        all_gt_boxes = torch.cat(all_gt_boxes, dim=0)

        # Sort predictions by score (descending)
        sort_idx = all_pred_scores.argsort(descending=True)
        all_pred_boxes = all_pred_boxes[sort_idx]
        all_pred_scores = all_pred_scores[sort_idx]
        all_pred_image_ids = [all_pred_image_ids[i] for i in sort_idx.tolist()]

        # Track which GT boxes have been matched
        num_gt = len(all_gt_boxes)
        gt_matched = torch.zeros(num_gt, dtype=torch.bool)

        # Build mapping from image_id to GT box indices
        gt_image_to_indices = {}
        for idx, img_id in enumerate(all_gt_image_ids):
            if img_id not in gt_image_to_indices:
                gt_image_to_indices[img_id] = []
            gt_image_to_indices[img_id].append(idx)

        # Compute TP/FP for each prediction
        tp = np.zeros(len(all_pred_boxes))
        fp = np.zeros(len(all_pred_boxes))

        for pred_idx in range(len(all_pred_boxes)):
            pred_box = all_pred_boxes[pred_idx:pred_idx+1]
            pred_img_id = all_pred_image_ids[pred_idx]

            if pred_img_id not in gt_image_to_indices:
                # No GT boxes for this image
                fp[pred_idx] = 1
                continue

            gt_indices = gt_image_to_indices[pred_img_id]
            gt_boxes_for_image = all_gt_boxes[gt_indices]

            # Compute IoU with all GT boxes in this image
            ious = box_iou(pred_box, gt_boxes_for_image)[0]

            # Find best matching GT box
            best_iou, best_idx = ious.max(dim=0)

            if best_iou >= iou_threshold:
                gt_idx = gt_indices[best_idx.item()]
                if not gt_matched[gt_idx]:
                    # True positive
                    tp[pred_idx] = 1
                    gt_matched[gt_idx] = True
                else:
                    # GT already matched, this is a duplicate
                    fp[pred_idx] = 1
            else:
                # IoU too low
                fp[pred_idx] = 1

        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Compute AP
        ap = compute_ap(recalls, precisions)
        aps[class_idx] = ap

    return aps


def compute_map(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    pred_labels: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    num_classes: int,
    iou_thresholds: List[float] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute mean Average Precision.

    Args:
        pred_boxes: List of predicted boxes per image
        pred_scores: List of predicted scores per image
        pred_labels: List of predicted labels per image
        gt_boxes: List of ground truth boxes per image
        gt_labels: List of ground truth labels per image
        num_classes: Number of classes
        iou_thresholds: List of IoU thresholds (default: [0.5] for mAP@0.5)

    Returns:
        Tuple of (mAP, dict with detailed metrics)
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    all_aps = []
    metrics = {}

    for iou_thresh in iou_thresholds:
        aps = compute_ap_per_class(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, num_classes, iou_thresh
        )

        # Store per-class APs
        for class_idx, ap in aps.items():
            metrics[f'AP_{class_idx}_@{iou_thresh}'] = ap
            all_aps.append(ap)

        # mAP at this threshold
        valid_aps = [ap for ap in aps.values() if ap > 0 or any(
            (gt_labels[i] == class_idx).any() for i, class_idx in enumerate(aps.keys()) for i in range(len(gt_labels))
        )]
        if valid_aps:
            map_at_thresh = np.mean(list(aps.values()))
        else:
            map_at_thresh = 0.0
        metrics[f'mAP@{iou_thresh}'] = map_at_thresh

    # Overall mAP (average over all thresholds and classes)
    if all_aps:
        mAP = np.mean(all_aps)
    else:
        mAP = 0.0

    return mAP, metrics


def compute_map_coco(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    pred_labels: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    num_classes: int
) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute COCO-style mAP (mAP@0.5:0.95).

    Args:
        pred_boxes: List of predicted boxes per image
        pred_scores: List of predicted scores per image
        pred_labels: List of predicted labels per image
        gt_boxes: List of ground truth boxes per image
        gt_labels: List of ground truth labels per image
        num_classes: Number of classes

    Returns:
        Tuple of (mAP@0.5:0.95, mAP@0.5, detailed metrics dict)
    """
    iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5, 0.55, ..., 0.95

    all_aps_per_thresh = []
    metrics = {}

    for iou_thresh in iou_thresholds:
        aps = compute_ap_per_class(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, num_classes, iou_thresh
        )

        ap_values = list(aps.values())
        if ap_values:
            map_at_thresh = np.mean(ap_values)
        else:
            map_at_thresh = 0.0

        all_aps_per_thresh.append(map_at_thresh)
        metrics[f'mAP@{iou_thresh:.2f}'] = map_at_thresh

    # mAP@0.5:0.95
    mAP_coco = np.mean(all_aps_per_thresh)
    # mAP@0.5
    mAP_50 = all_aps_per_thresh[0]

    metrics['mAP@0.5:0.95'] = mAP_coco
    metrics['mAP@0.5'] = mAP_50

    return mAP_coco, mAP_50, metrics


if __name__ == "__main__":
    # Test metrics computation
    num_classes = 5

    # Create dummy predictions and ground truths
    pred_boxes = [
        torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=torch.float32),
        torch.tensor([[20, 20, 60, 60]], dtype=torch.float32)
    ]
    pred_scores = [
        torch.tensor([0.9, 0.8]),
        torch.tensor([0.7])
    ]
    pred_labels = [
        torch.tensor([0, 1]),
        torch.tensor([0])
    ]

    gt_boxes = [
        torch.tensor([[10, 10, 50, 50], [65, 65, 105, 105]], dtype=torch.float32),
        torch.tensor([[25, 25, 55, 55]], dtype=torch.float32)
    ]
    gt_labels = [
        torch.tensor([0, 1]),
        torch.tensor([0])
    ]

    # Test mAP@0.5
    mAP, metrics = compute_map(
        pred_boxes, pred_scores, pred_labels,
        gt_boxes, gt_labels, num_classes,
        iou_thresholds=[0.5]
    )
    print(f"mAP@0.5: {mAP:.4f}")
    print(f"Metrics: {metrics}")

    # Test COCO mAP
    mAP_coco, mAP_50, metrics = compute_map_coco(
        pred_boxes, pred_scores, pred_labels,
        gt_boxes, gt_labels, num_classes
    )
    print(f"\nmAP@0.5:0.95: {mAP_coco:.4f}")
    print(f"mAP@0.5: {mAP_50:.4f}")
