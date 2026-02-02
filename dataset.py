"""
Custom Dataset for Russian Rubles Banknotes detection.
Loads from HuggingFace Hub and prepares FCOS-style targets.
"""

import json
import os
from typing import Dict, List, Tuple, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Class ID to name mapping (from original dataset)
ID_TO_CLASS_NAME = {
    0: "check",
    1: "napkins",
    2: "glue",
    3: "banknotes",
    4: "banknotes_batch",
    5: "calculator",
    6: "bcard",
    7: "partpack",
    8: "photo_paper",
    9: "iceberg_card",
    10: "document",
    11: "package",
    12: "smartphone",
    13: "svetocopy",
    14: "banknotes10",
    15: "banknotes50",
    16: "banknotes100",
    17: "banknotes200",
    18: "banknotes500",
    19: "banknotes1000",
    20: "banknotes2000",
    21: "banknotes5000",
    22: "mouse",
    23: "other",
}


class BanknoteDataset(Dataset):
    """Dataset for banknote detection with FCOS-style target encoding."""

    def __init__(
        self,
        split: str = "train",
        config_path: str = "config.json",
        transform: Optional[A.Compose] = None
    ):
        """
        Args:
            split: 'train' or 'test'
            config_path: Path to config.json
            transform: Albumentations transform pipeline
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.split = split
        self.image_size = tuple(self.config['image_size'])  # (W, H)
        self.num_classes = self.config['num_classes']
        self.strides = self.config['strides']
        self.fpn_channels = self.config['fpn_channels']

        # Size ranges for assigning objects to FPN levels
        # Objects are assigned based on their max(l, t, r, b) distance
        self.size_ranges = [
            (0, 64),      # P3: stride 8
            (64, 128),    # P4: stride 16
            (128, 256),   # P5: stride 32
            (256, 512),   # P6: stride 64
            (512, float('inf'))  # P7: stride 128
        ]

        # Load dataset from HuggingFace
        hf_token = os.getenv('HF_TOKEN')
        self.dataset = load_dataset(
            self.config['hf_dataset'],
            split=split,
            token=hf_token
        )

        # Build class mapping
        self._build_class_mapping()

        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform(split == 'train')

    def _build_class_mapping(self):
        """Build mapping from class IDs to names and indices."""
        # Collect all unique category IDs from dataset
        all_category_ids = set()
        for item in self.dataset:
            for obj in item['objects']:
                all_category_ids.add(obj['category'])

        # Sort category IDs and create mappings
        sorted_ids = sorted(list(all_category_ids))

        # Map original category IDs to sequential indices (0, 1, 2, ...)
        # and get proper class names from ID_TO_CLASS_NAME
        self.category_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted_ids)}
        self.idx_to_category_id = {idx: cat_id for cat_id, idx in self.category_id_to_idx.items()}

        # Get class names using the ID_TO_CLASS_NAME mapping
        self.class_names = [ID_TO_CLASS_NAME.get(cat_id, f"class_{cat_id}") for cat_id in sorted_ids]
        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Update num_classes based on actual data
        self.num_classes = len(self.class_names)

    def _get_default_transform(self, is_train: bool) -> A.Compose:
        """Get default augmentation pipeline."""
        if is_train:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.5),  # Scale 0.8-1.2x
                A.RandomRotate90(p=0.5),  # Rotate by 90° increments
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Add slight noise
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3
            ))
        else:
            return A.Compose([], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels']
            ))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            image: Tensor of shape (3, H, W), normalized to [0, 1]
            targets: Dict containing:
                - boxes: Tensor of shape (N, 4) in xyxy format
                - labels: Tensor of shape (N,) with class indices
                - fcos_targets: List of dicts per FPN level
        """
        item = self.dataset[idx]

        # Load image
        image = np.array(item['image'])
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        orig_h, orig_w = image.shape[:2]

        # Parse bounding boxes (COCO format: x, y, w, h -> pascal_voc: x1, y1, x2, y2)
        boxes = []
        class_labels = []
        for obj in item['objects']:
            bbox = obj['bbox']
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Clip to image bounds
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            class_labels.append(self.category_id_to_idx[obj['category']])

        # Apply augmentations
        if len(boxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=class_labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']

        # Resize image if needed
        target_w, target_h = self.image_size
        if image.shape[:2] != (target_h, target_w):
            scale_x = target_w / image.shape[1]
            scale_y = target_h / image.shape[0]
            image = cv2.resize(image, (target_w, target_h))

            # Scale boxes
            boxes = [
                [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
                for b in boxes
            ]

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Prepare targets
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(class_labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        targets = {
            'boxes': boxes,
            'labels': labels,
            'image_id': idx
        }

        return image, targets

    def compute_fcos_targets(
        self,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device = torch.device('cpu')
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Compute FCOS targets for all FPN levels.

        Args:
            boxes: Tensor of shape (N, 4) in xyxy format
            labels: Tensor of shape (N,) with class indices
            device: Device to place tensors on

        Returns:
            List of dicts per FPN level, each containing:
                - cls_targets: (H, W) class labels (-1 for ignore, 0 for background)
                - reg_targets: (H, W, 4) regression targets (l, t, r, b)
                - ctr_targets: (H, W) centerness targets
        """
        target_w, target_h = self.image_size
        fcos_targets = []

        for level_idx, stride in enumerate(self.strides):
            feat_h = target_h // stride
            feat_w = target_w // stride
            min_size, max_size = self.size_ranges[level_idx]

            # Initialize targets
            cls_targets = torch.zeros((feat_h, feat_w), dtype=torch.long, device=device)
            reg_targets = torch.zeros((feat_h, feat_w, 4), dtype=torch.float32, device=device)
            ctr_targets = torch.zeros((feat_h, feat_w), dtype=torch.float32, device=device)

            if len(boxes) == 0:
                fcos_targets.append({
                    'cls_targets': cls_targets,
                    'reg_targets': reg_targets,
                    'ctr_targets': ctr_targets
                })
                continue

            # Create grid of locations
            shifts_x = torch.arange(0, feat_w, device=device) * stride + stride // 2
            shifts_y = torch.arange(0, feat_h, device=device) * stride + stride // 2

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            locations = torch.stack([shift_x, shift_y], dim=-1).float()  # (H, W, 2)

            # For each location, find the best matching box
            for box_idx in range(len(boxes)):
                box = boxes[box_idx].to(device)
                label = labels[box_idx].item()

                x1, y1, x2, y2 = box

                # Compute l, t, r, b for all locations
                l = locations[..., 0] - x1  # distance to left edge
                t = locations[..., 1] - y1  # distance to top edge
                r = x2 - locations[..., 0]  # distance to right edge
                b = y2 - locations[..., 1]  # distance to bottom edge

                reg = torch.stack([l, t, r, b], dim=-1)  # (H, W, 4)

                # Check if location is inside the box
                inside_mask = (l > 0) & (t > 0) & (r > 0) & (b > 0)

                # Check if object should be assigned to this level based on size
                max_reg = reg.max(dim=-1)[0]  # (H, W)
                size_mask = (max_reg >= min_size) & (max_reg < max_size)

                # Combined mask
                valid_mask = inside_mask & size_mask

                # Compute centerness
                lr_min = torch.min(l, r)
                lr_max = torch.max(l, r)
                tb_min = torch.min(t, b)
                tb_max = torch.max(t, b)

                centerness = torch.sqrt(
                    (lr_min / (lr_max + 1e-6)) * (tb_min / (tb_max + 1e-6))
                )

                # Only update locations that are valid and have higher centerness
                # (prefer center of objects for ambiguous locations)
                update_mask = valid_mask & (centerness > ctr_targets)

                cls_targets[update_mask] = label + 1  # 0 is background, classes start at 1
                reg_targets[update_mask] = reg[update_mask]
                ctr_targets[update_mask] = centerness[update_mask]

            fcos_targets.append({
                'cls_targets': cls_targets,
                'reg_targets': reg_targets,
                'ctr_targets': ctr_targets
            })

        return fcos_targets


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """Custom collate function for variable number of boxes per image."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def get_dataloaders(
    config_path: str = "config.json",
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_dataset = BanknoteDataset(split='train', config_path=config_path)
    test_dataset = BanknoteDataset(split='test', config_path=config_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Loading dataset...")
    dataset = BanknoteDataset(split='train')
    print(f"Number of training samples: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Class names: {dataset.class_names}")

    # Test single item
    image, targets = dataset[0]
    print(f"\nImage shape: {image.shape}")
    print(f"Number of boxes: {len(targets['boxes'])}")
    print(f"Boxes shape: {targets['boxes'].shape}")
    print(f"Labels shape: {targets['labels'].shape}")

    # Test FCOS target computation
    fcos_targets = dataset.compute_fcos_targets(targets['boxes'], targets['labels'])
    print(f"\nFCOS targets per level:")
    for i, level_targets in enumerate(fcos_targets):
        cls_shape = level_targets['cls_targets'].shape
        reg_shape = level_targets['reg_targets'].shape
        pos_count = (level_targets['cls_targets'] > 0).sum().item()
        print(f"  Level {i}: cls={cls_shape}, reg={reg_shape}, positive={pos_count}")

    # Test dataloader
    print("\nTesting dataloader...")
    train_loader, test_loader = get_dataloaders()
    images, targets = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch size: {len(targets)}")
