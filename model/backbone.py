"""
ResNet-18 backbone implementation from scratch.
Extracts multi-scale features for use with FPN.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """
    ResNet-18 backbone implementation from scratch.

    Returns features at 4 scales (C2, C3, C4, C5) with strides [4, 8, 16, 32].
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)   # C2: stride 4
        self.layer2 = self._make_layer(128, 2, stride=2)  # C3: stride 8
        self.layer3 = self._make_layer(256, 2, stride=2)  # C4: stride 16
        self.layer4 = self._make_layer(512, 2, stride=2)  # C5: stride 32

        # Output channel dimensions
        self.out_channels = [64, 128, 256, 512]

        # Initialize weights
        self._init_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a residual layer with specified number of blocks."""
        downsample = None

        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * BasicBlock.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            List of feature maps [C2, C3, C4, C5] with strides [4, 8, 16, 32]
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        c2 = self.layer1(x)   # stride 4
        c3 = self.layer2(c2)  # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32

        return [c2, c3, c4, c5]


if __name__ == "__main__":
    # Test backbone
    model = ResNet18()
    print(f"ResNet-18 backbone created")
    print(f"Output channels: {model.out_channels}")

    # Test forward pass
    x = torch.randn(2, 3, 1080, 1920)
    features = model(x)

    print("\nFeature map shapes:")
    for i, feat in enumerate(features):
        print(f"  C{i+2}: {feat.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
