"""
Full FCOS (Fully Convolutional One-Stage) object detector.
Combines backbone, FPN, and detection head with inference logic.
"""

import json
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

from .backbone import ResNet18
from .fpn import FPN
from .head import FCOSHead


class FCOS(nn.Module):
    """
    FCOS: Fully Convolutional One-Stage Object Detection.

    A single-stage, anchor-free object detector that predicts bounding boxes
    directly at each spatial location in the feature maps.
    """

    def __init__(
        self,
        num_classes: int = 24,
        fpn_channels: int = 256,
        strides: List[int] = None,
        score_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        max_detections: int = 100
    ):
        """
        Args:
            num_classes: Number of object classes (excluding background)
            fpn_channels: Number of channels in FPN features
            strides: FPN strides for each level
            score_threshold: Minimum score for detection
            nms_threshold: IoU threshold for NMS
            max_detections: Maximum detections per image
        """
        super().__init__()

        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self.strides = strides or [8, 16, 32, 64, 128]
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections

        # Build network components
        self.backbone = ResNet18()
        self.fpn = FPN(
            in_channels_list=self.backbone.out_channels,
            out_channels=fpn_channels
        )
        self.head = FCOSHead(
            in_channels=fpn_channels,
            num_classes=num_classes
        )

    @classmethod
    def from_config(cls, config_path: str) -> 'FCOS':
        """Create FCOS model from config file."""
        with open(config_path, 'r') as f:
            config = json.load(f)

        return cls(
            num_classes=config['num_classes'],
            fpn_channels=config['fpn_channels'],
            strides=config['strides'],
            score_threshold=config.get('score_threshold', 0.05),
            nms_threshold=config.get('nms_threshold', 0.5),
            max_detections=config.get('max_detections', 100)
        )

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.

        Args:
            images: Tensor of shape (B, 3, H, W)
            targets: Optional list of target dicts (for training)

        Returns:
            Tuple of prediction lists (one per FPN level):
                - cls_outputs: List of (B, num_classes, Hi, Wi)
                - reg_outputs: List of (B, 4, Hi, Wi) - l, t, r, b distances
                - ctr_outputs: List of (B, 1, Hi, Wi) - centerness scores
        """
        # Extract backbone features
        backbone_features = self.backbone(images)

        # Build FPN
        fpn_features = self.fpn(backbone_features)

        # Get predictions from head
        cls_outputs, reg_outputs, ctr_outputs = self.head(fpn_features)

        return cls_outputs, reg_outputs, ctr_outputs

    def inference(
        self,
        images: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference and return detections.

        Args:
            images: Tensor of shape (B, 3, H, W)

        Returns:
            List of dicts per image, each containing:
                - boxes: (N, 4) tensor of xyxy boxes
                - scores: (N,) tensor of confidence scores
                - labels: (N,) tensor of class labels
        """
        self.eval()
        batch_size = images.shape[0]
        device = images.device

        with torch.no_grad():
            cls_outputs, reg_outputs, ctr_outputs = self.forward(images)

        results = []

        for batch_idx in range(batch_size):
            all_boxes = []
            all_scores = []
            all_labels = []

            for level, stride in enumerate(self.strides):
                # Get predictions for this level and batch
                cls_pred = cls_outputs[level][batch_idx]  # (C, H, W)
                reg_pred = reg_outputs[level][batch_idx]  # (4, H, W)
                ctr_pred = ctr_outputs[level][batch_idx]  # (1, H, W)

                h, w = cls_pred.shape[1:]

                # Create grid of locations
                shifts_x = torch.arange(0, w, device=device) * stride + stride // 2
                shifts_y = torch.arange(0, h, device=device) * stride + stride // 2
                shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
                locations = torch.stack([shift_x, shift_y], dim=0).float()  # (2, H, W)

                # Decode boxes from (l, t, r, b) predictions
                l, t, r, b = reg_pred[0], reg_pred[1], reg_pred[2], reg_pred[3]
                x1 = locations[0] - l
                y1 = locations[1] - t
                x2 = locations[0] + r
                y2 = locations[1] + b

                boxes = torch.stack([x1, y1, x2, y2], dim=0)  # (4, H, W)

                # Apply sigmoid to get probabilities
                cls_scores = cls_pred.sigmoid()  # (C, H, W)
                ctr_scores = ctr_pred.sigmoid()  # (1, H, W)

                # Combine classification and centerness scores
                scores = cls_scores * ctr_scores  # (C, H, W)

                # Flatten predictions
                boxes = boxes.permute(1, 2, 0).reshape(-1, 4)  # (H*W, 4)
                scores = scores.permute(1, 2, 0).reshape(-1, self.num_classes)  # (H*W, C)

                # Get best class per location
                max_scores, max_labels = scores.max(dim=1)

                # Filter by score threshold
                keep = max_scores > self.score_threshold
                boxes = boxes[keep]
                max_scores = max_scores[keep]
                max_labels = max_labels[keep]

                all_boxes.append(boxes)
                all_scores.append(max_scores)
                all_labels.append(max_labels)

            # Concatenate predictions from all levels
            if len(all_boxes) > 0:
                all_boxes = torch.cat(all_boxes, dim=0)
                all_scores = torch.cat(all_scores, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                # Apply NMS per class
                keep_indices = self._batched_nms(
                    all_boxes, all_scores, all_labels,
                    self.nms_threshold
                )

                # Limit detections
                if len(keep_indices) > self.max_detections:
                    _, top_indices = all_scores[keep_indices].topk(self.max_detections)
                    keep_indices = keep_indices[top_indices]

                boxes = all_boxes[keep_indices]
                scores = all_scores[keep_indices]
                labels = all_labels[keep_indices]
            else:
                boxes = torch.zeros((0, 4), device=device)
                scores = torch.zeros((0,), device=device)
                labels = torch.zeros((0,), dtype=torch.long, device=device)

            results.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })

        return results

    def _batched_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        iou_threshold: float
    ) -> torch.Tensor:
        """
        Perform batched NMS (NMS per class).

        Args:
            boxes: (N, 4) boxes in xyxy format
            scores: (N,) confidence scores
            labels: (N,) class labels
            iou_threshold: IoU threshold for NMS

        Returns:
            Indices of kept boxes
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)

        # Offset boxes by class to avoid cross-class suppression
        max_coordinate = boxes.max()
        offsets = labels.float() * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

        # Apply NMS
        keep = self._nms(boxes_for_nms, scores, iou_threshold)

        return keep

    def _nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float
    ) -> torch.Tensor:
        """
        Non-maximum suppression implementation.

        Args:
            boxes: (N, 4) boxes in xyxy format
            scores: (N,) confidence scores
            iou_threshold: IoU threshold

        Returns:
            Indices of kept boxes
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)

        # Sort by scores
        _, order = scores.sort(descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order[0].item())
                break

            i = order[0].item()
            keep.append(i)

            # Compute IoU with remaining boxes
            iou = self._box_iou(boxes[i:i+1], boxes[order[1:]])

            # Keep boxes with IoU below threshold
            mask = iou.squeeze(0) <= iou_threshold
            order = order[1:][mask]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def _box_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
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

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2 - inter

        return inter / (union + 1e-6)

    def get_locations(
        self,
        features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Get grid locations for each FPN level.

        Args:
            features: List of feature maps

        Returns:
            List of location tensors, each (H*W, 2) with (x, y) coordinates
        """
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.shape[2:]
            stride = self.strides[level]

            shifts_x = torch.arange(0, w, device=feature.device) * stride + stride // 2
            shifts_y = torch.arange(0, h, device=feature.device) * stride + stride // 2

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            locs = torch.stack([shift_x.flatten(), shift_y.flatten()], dim=1).float()
            locations.append(locs)

        return locations


if __name__ == "__main__":
    # Test FCOS model
    model = FCOS(num_classes=24)

    print(f"FCOS model created")
    print(f"  Num classes: {model.num_classes}")
    print(f"  FPN channels: {model.fpn_channels}")
    print(f"  Strides: {model.strides}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    x = torch.randn(2, 3, 1080, 1920)
    print(f"\nInput shape: {x.shape}")

    model.eval()
    cls_outputs, reg_outputs, ctr_outputs = model(x)

    print("\nOutput shapes per level:")
    for i, (cls, reg, ctr) in enumerate(zip(cls_outputs, reg_outputs, ctr_outputs)):
        print(f"  P{i+3}: cls={cls.shape}, reg={reg.shape}, ctr={ctr.shape}")

    # Test inference
    print("\nTesting inference...")
    results = model.inference(x)
    for i, res in enumerate(results):
        print(f"  Image {i}: {len(res['boxes'])} detections")
