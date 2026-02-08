"""
Custom Dataset for Russian Rubles Banknotes detection.
Loads from HuggingFace Hub and prepares FCOS-style targets.
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

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

# Mapping for merging banknote classes
# All specific denominations map to generic "banknotes" class
BANKNOTE_MERGE_MAP = {
    3: "banknotes",      # banknotes (generic)
    4: "banknotes",      # banknotes_batch
    14: "banknotes",     # banknotes10
    15: "banknotes",     # banknotes50
    16: "banknotes",     # banknotes100
    17: "banknotes",     # banknotes200
    18: "banknotes",     # banknotes500
    19: "banknotes",     # banknotes1000
    20: "banknotes",     # banknotes2000
    21: "banknotes",     # banknotes5000
}


class BanknoteDataset(Dataset):
    """Dataset for banknote detection with FCOS-style target encoding."""

    def __init__(
        self,
        split: str = "train",
        config_path: str = "config.json",
        transform: Optional[A.Compose] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        merge_banknotes: bool = True
    ):
        """
        Args:
            split: 'train', 'val', or 'test'
            config_path: Path to config.json
            transform: Albumentations transform pipeline
            train_ratio: Ratio for training set (default: 0.7)
            val_ratio: Ratio for validation set (default: 0.15)
            test_ratio: Ratio for test set (default: 0.15)
            random_seed: Random seed for reproducible splits
            merge_banknotes: If True, merge all banknote classes into one (default: True)
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.split = split
        self.image_size = tuple(self.config['image_size'])  # (W, H)
        self.num_classes = self.config['num_classes']
        self.strides = self.config['strides']
        self.fpn_channels = self.config['fpn_channels']
        self.merge_banknotes = merge_banknotes

        # Size ranges for assigning objects to FPN levels
        # Objects are assigned based on their max(l, t, r, b) distance
        self.size_ranges = [
            (0, 64),      # P3: stride 8
            (64, 128),    # P4: stride 16
            (128, 256),   # P5: stride 32
            (256, 512),   # P6: stride 64
            (512, float('inf'))  # P7: stride 128
        ]

        # Load and concatenate both train and test from HuggingFace
        hf_token = os.getenv('HF_TOKEN')
        print(f"Loading complete dataset from HuggingFace...")

        train_dataset = load_dataset(
            self.config['hf_dataset'],
            split='train',
            token=hf_token
        )
        test_dataset = load_dataset(
            self.config['hf_dataset'],
            split='test',
            token=hf_token
        )

        # Concatenate train and test
        from datasets import concatenate_datasets
        full_dataset = concatenate_datasets([train_dataset, test_dataset])
        print(f"  Total images: {len(full_dataset)} (train: {len(train_dataset)}, test: {len(test_dataset)})")

        # Create custom train/val/test split
        train_idx, val_idx, test_idx = self._create_custom_split(
            full_dataset, train_ratio, val_ratio, test_ratio, random_seed
        )

        # Select subset based on split
        if split == 'train':
            self.dataset = full_dataset.select(train_idx)
        elif split == 'val':
            self.dataset = full_dataset.select(val_idx)
        elif split == 'test':
            self.dataset = full_dataset.select(test_idx)
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"  Split '{split}': {len(self.dataset)} images")

        # Build class mapping (with optional merging)
        self._build_class_mapping()

        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform(split == 'train')

    def _create_custom_split(
        self,
        dataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ):
        """
        Create custom train/val/test split from complete dataset.

        Args:
            dataset: Full HuggingFace dataset
            train_ratio: Proportion for training (default: 0.7)
            val_ratio: Proportion for validation (default: 0.15)
            test_ratio: Proportion for test (default: 0.15)
            random_seed: Random seed for reproducibility

        Returns:
            train_idx, val_idx, test_idx: Lists of indices for each split
        """
        from sklearn.model_selection import train_test_split

        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

        # Get labels for stratification (use first object's category, merged if applicable)
        indices = list(range(len(dataset)))
        labels = []
        for item in dataset:
            if len(item['objects']) > 0:
                cat_id = item['objects'][0]['category']
                # Apply merging if enabled
                if self.merge_banknotes and cat_id in BANKNOTE_MERGE_MAP:
                    labels.append('banknotes')  # Use string label for merged class
                else:
                    labels.append(cat_id)
            else:
                labels.append(-1)

        # First split: separate out test set
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            stratify=labels,
            random_state=random_seed
        )

        # Second split: separate train and val from remaining data
        # Adjust val_ratio to account for already removed test set
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)

        train_val_labels = [labels[i] for i in train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=adjusted_val_ratio,
            stratify=train_val_labels,
            random_state=random_seed
        )

        print(f"  Custom split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        print(f"  Ratios: {len(train_idx)/len(indices):.1%} / {len(val_idx)/len(indices):.1%} / {len(test_idx)/len(indices):.1%}")

        return train_idx, val_idx, test_idx

    def _create_train_val_split(self, train_dataset, val_ratio=0.2, random_seed=42):
        """
        Split HF train dataset into train/val using stratified sampling when possible.
        Falls back to random split for classes with too few samples.

        Args:
            train_dataset: HuggingFace dataset to split
            val_ratio: Ratio of data to use for validation
            random_seed: Random seed for reproducibility

        Returns:
            train_idx, val_idx: Lists of indices for train and validation sets
        """
        from sklearn.model_selection import train_test_split
        from collections import Counter

        # Get all indices and their labels (use first object's category)
        indices = list(range(len(train_dataset)))
        labels = []
        for item in train_dataset:
            if len(item['objects']) > 0:
                labels.append(item['objects'][0]['category'])
            else:
                labels.append(-1)  # Images with no objects

        # Check class distribution
        class_counts = Counter(labels)

        # Try stratified split, but if some classes have < 2 samples, use random split
        try:
            # Stratified split to maintain class distribution
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_ratio,
                stratify=labels,
                random_state=random_seed
            )
        except ValueError as e:
            # Some classes have too few samples for stratification
            # Fall back to random split
            print(f"Warning: Cannot use stratified split (some classes have < 2 samples)")
            print(f"  Falling back to random split")
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_ratio,
                random_state=random_seed,
                shuffle=True
            )

        return train_idx, val_idx

    def _build_class_mapping(self):
        """Build mapping from class IDs to names and indices, with optional banknote merging."""
        # Collect all unique category IDs from dataset
        all_category_ids = set()
        for item in self.dataset:
            for obj in item['objects']:
                cat_id = obj['category']
                all_category_ids.add(cat_id)

        # Apply merging if enabled
        if self.merge_banknotes:
            # Create merged class names
            merged_class_names = set()
            original_to_merged = {}

            for cat_id in all_category_ids:
                if cat_id in BANKNOTE_MERGE_MAP:
                    merged_name = BANKNOTE_MERGE_MAP[cat_id]
                    merged_class_names.add(merged_name)
                    original_to_merged[cat_id] = merged_name
                else:
                    class_name = ID_TO_CLASS_NAME.get(cat_id, f"class_{cat_id}")
                    merged_class_names.add(class_name)
                    original_to_merged[cat_id] = class_name

            # Create sorted list of unique class names
            self.class_names = sorted(list(merged_class_names))
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

            # Map original category IDs to merged class indices
            self.category_id_to_idx = {
                cat_id: self.class_to_idx[original_to_merged[cat_id]]
                for cat_id in all_category_ids
            }
            self.idx_to_category_id = None  # Not meaningful with merging

            print(f"  Classes after merging: {self.class_names}")
            print(f"  Total classes: {len(self.class_names)}")

        else:
            # Original behavior without merging
            sorted_ids = sorted(list(all_category_ids))

            self.category_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted_ids)}
            self.idx_to_category_id = {idx: cat_id for cat_id, idx in self.category_id_to_idx.items()}

            self.class_names = [ID_TO_CLASS_NAME.get(cat_id, f"class_{cat_id}") for cat_id in sorted_ids]
            self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Update num_classes based on actual data
        self.num_classes = len(self.class_names)

    def compute_class_weights(self, method='inverse_freq'):
        """
        Compute per-class weights based on class frequency to handle imbalance.

        Args:
            method: 'inverse_freq' for inverse frequency weighting

        Returns:
            Tensor of shape (num_classes,) with per-class weights
        """
        # Count occurrences of each class
        class_counts = defaultdict(int)
        for item in self.dataset:
            for obj in item['objects']:
                class_idx = self.category_id_to_idx[obj['category']]
                class_counts[class_idx] += 1

        # Compute inverse frequency weights
        total = sum(class_counts.values())
        weights = torch.zeros(self.num_classes)
        for cls_idx in range(self.num_classes):
            count = class_counts.get(cls_idx, 1)  # Avoid division by zero
            weights[cls_idx] = total / (self.num_classes * count)

        # Normalize weights to have mean 1.0, then clamp to [0.5, 2.0]
        # This prevents extreme weights from destabilizing training
        weights = weights / weights.mean()
        weights = torch.clamp(weights, 0.5, 2.0)

        return weights

    def _get_default_transform(self, is_train: bool) -> A.Compose:
        """Get default augmentation pipeline with aggressive realistic transforms."""
        if is_train:
            return A.Compose([
                # Geometric transforms - realistic for banknote detection
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=30, p=0.7),  # Realistic rotation ±30°
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.3,  # More aggressive: 0.7-1.3x
                    rotate_limit=15,
                    p=0.6
                ),
                A.Perspective(scale=(0.05, 0.15), p=0.4),  # Camera perspective
                A.ElasticTransform(alpha=1, sigma=50, p=0.2),  # Deformation

                # Photometric transforms - simulate different lighting conditions
                A.RandomBrightnessContrast(
                    brightness_limit=0.4,  # More aggressive
                    contrast_limit=0.4,
                    p=0.7
                ),
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(70, 130), p=0.3),
                A.CLAHE(clip_limit=4.0, p=0.3),  # Adaptive histogram equalization

                # Noise and blur - simulate camera artifacts
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 80.0), mean=0, per_channel=True, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.3),

                # Cutout/Coarse Dropout - simulate occlusion
                A.CoarseDropout(
                    num_holes_range=(1, 3),
                    hole_height_range=(30, 60),
                    hole_width_range=(30, 60),
                    fill_value=0,
                    p=0.3
                ),
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3,
                min_area=100  # Filter tiny boxes after augmentation
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
    num_workers: int = 4,
    merge_banknotes: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_dataset = BanknoteDataset(
        split='train',
        config_path=config_path,
        merge_banknotes=merge_banknotes
    )
    val_dataset = BanknoteDataset(
        split='val',
        config_path=config_path,
        merge_banknotes=merge_banknotes
    )
    test_dataset = BanknoteDataset(
        split='test',
        config_path=config_path,
        merge_banknotes=merge_banknotes
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
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

    return train_loader, val_loader, test_loader


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
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    images, targets = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch size: {len(targets)}")
