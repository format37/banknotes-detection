"""
ResNet backbone implementations for FCOS.
Supports ResNet-18 (from scratch) and ResNet-50 (with pretrained weights).
Extracts multi-scale features for use with FPN.
"""

import torch
import torch.nn as nn
import torchvision.models as models
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


class ResNet50(nn.Module):
    """
    ResNet-50 backbone wrapper for torchvision pretrained models.

    Returns features at 4 scales (C2, C3, C4, C5) with strides [4, 8, 16, 32].
    Channel dimensions: [256, 512, 1024, 2048]
    """

    def __init__(self, pretrained: bool = True):
        """
        Args:
            pretrained: If True, load ImageNet pretrained weights
        """
        super().__init__()

        # Load pretrained or random ResNet-50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(weights=None)

        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # C2: 256 channels, stride 4
        self.layer2 = resnet.layer2  # C3: 512 channels, stride 8
        self.layer3 = resnet.layer3  # C4: 1024 channels, stride 16
        self.layer4 = resnet.layer4  # C5: 2048 channels, stride 32

        # Output channel dimensions
        self.out_channels = [256, 512, 1024, 2048]

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
        c2 = self.layer1(x)   # stride 4, 256 channels
        c3 = self.layer2(c2)  # stride 8, 512 channels
        c4 = self.layer3(c3)  # stride 16, 1024 channels
        c5 = self.layer4(c4)  # stride 32, 2048 channels

        return [c2, c3, c4, c5]


class BackboneFactory:
    """Factory for creating backbone networks."""

    @staticmethod
    def create_backbone(backbone_type: str = 'resnet50', pretrained: bool = True):
        """
        Create a backbone network.

        Args:
            backbone_type: 'resnet18' or 'resnet50'
            pretrained: If True and using resnet50, load ImageNet pretrained weights

        Returns:
            Backbone module with .out_channels attribute
        """
        if backbone_type == 'resnet50':
            return ResNet50(pretrained=pretrained)
        elif backbone_type == 'resnet18':
            return ResNet18()
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}. "
                           f"Supported: 'resnet18', 'resnet50'")


if __name__ == "__main__":
    # Test ResNet-18
    print("Testing ResNet-18:")
    model = ResNet18()
    print(f"  Output channels: {model.out_channels}")
    x = torch.randn(2, 3, 1080, 1920)
    features = model(x)
    print("  Feature map shapes:")
    for i, feat in enumerate(features):
        print(f"    C{i+2}: {feat.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Test ResNet-50
    print("\nTesting ResNet-50 (pretrained):")
    model = ResNet50(pretrained=True)
    print(f"  Output channels: {model.out_channels}")
    features = model(x)
    print("  Feature map shapes:")
    for i, feat in enumerate(features):
        print(f"    C{i+2}: {feat.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Test factory
    print("\nTesting BackboneFactory:")
    backbone = BackboneFactory.create_backbone('resnet50', pretrained=True)
    print(f"  Created: {type(backbone).__name__}")
    print(f"  Output channels: {backbone.out_channels}")
