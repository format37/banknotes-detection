"""
FCOS detection head implementation.
Contains classification, regression, and centerness prediction heads.
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple


class FCOSHead(nn.Module):
    """
    FCOS detection head with classification, regression, and centerness branches.

    Uses shared convolutional layers followed by separate prediction layers.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 24,
        num_convs: int = 4,
        prior_prob: float = 0.01
    ):
        """
        Args:
            in_channels: Number of input channels from FPN
            num_classes: Number of object classes (excluding background)
            num_convs: Number of shared conv layers per branch
            prior_prob: Prior probability for classification (for bias init)
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_convs = num_convs

        # Shared classification tower
        cls_tower = []
        for i in range(num_convs):
            cls_tower.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)

        # Shared regression tower
        reg_tower = []
        for i in range(num_convs):
            reg_tower.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
            )
            reg_tower.append(nn.GroupNorm(32, in_channels))
            reg_tower.append(nn.ReLU(inplace=True))
        self.reg_tower = nn.Sequential(*reg_tower)

        # Classification prediction (num_classes outputs per location)
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)

        # Regression prediction (4 outputs: l, t, r, b)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)

        # Centerness prediction (1 output)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)

        # Learnable scale parameters for each FPN level (for regression)
        self.scales = nn.ModuleList([Scale() for _ in range(5)])

        # Initialize weights
        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob: float):
        """Initialize weights with proper initialization."""
        # Initialize conv layers
        for modules in [self.cls_tower, self.reg_tower]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # Initialize prediction layers
        for m in [self.cls_logits, self.bbox_pred, self.centerness]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

        # Initialize classification bias for focal loss
        # This helps with training stability at the start
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(
        self,
        features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the detection head.

        Args:
            features: List of FPN features [P3, P4, P5, P6, P7]

        Returns:
            Tuple of lists:
                - cls_outputs: List of (B, num_classes, H, W) tensors
                - reg_outputs: List of (B, 4, H, W) tensors
                - ctr_outputs: List of (B, 1, H, W) tensors
        """
        cls_outputs = []
        reg_outputs = []
        ctr_outputs = []

        for level, feature in enumerate(features):
            # Classification branch
            cls_feat = self.cls_tower(feature)
            cls_logits = self.cls_logits(cls_feat)

            # Regression branch
            reg_feat = self.reg_tower(feature)
            bbox_pred = self.bbox_pred(reg_feat)
            # Apply scale and exp to ensure positive values
            bbox_pred = self.scales[level](bbox_pred).exp()

            # Centerness (from regression features, as in original FCOS)
            centerness = self.centerness(reg_feat)

            cls_outputs.append(cls_logits)
            reg_outputs.append(bbox_pred)
            ctr_outputs.append(centerness)

        return cls_outputs, reg_outputs, ctr_outputs


class Scale(nn.Module):
    """Learnable scale parameter for FCOS regression."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


if __name__ == "__main__":
    # Test head
    head = FCOSHead(in_channels=256, num_classes=24)

    print(f"FCOS Head created for {head.num_classes} classes")

    # Create dummy FPN features
    batch_size = 2
    features = [
        torch.randn(batch_size, 256, 135, 240),  # P3: 1080/8, 1920/8
        torch.randn(batch_size, 256, 68, 120),   # P4: 1080/16, 1920/16
        torch.randn(batch_size, 256, 34, 60),    # P5: 1080/32, 1920/32
        torch.randn(batch_size, 256, 17, 30),    # P6: 1080/64, 1920/64
        torch.randn(batch_size, 256, 9, 15),     # P7: 1080/128, 1920/128
    ]

    cls_outputs, reg_outputs, ctr_outputs = head(features)

    print("\nOutput shapes:")
    strides = [8, 16, 32, 64, 128]
    for i, (cls, reg, ctr) in enumerate(zip(cls_outputs, reg_outputs, ctr_outputs)):
        print(f"  P{i+3} (stride {strides[i]}):")
        print(f"    Classification: {cls.shape}")
        print(f"    Regression: {reg.shape}")
        print(f"    Centerness: {ctr.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in head.parameters())
    print(f"\nTotal parameters: {total_params:,}")
