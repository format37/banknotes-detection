"""
Feature Pyramid Network (FPN) implementation.
Creates multi-scale feature maps with top-down pathway and lateral connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale object detection.

    Takes features from backbone (C2, C3, C4, C5) and produces
    pyramid features (P3, P4, P5, P6, P7) with strides [8, 16, 32, 64, 128].
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256
    ):
        """
        Args:
            in_channels_list: List of input channel dimensions [C2, C3, C4, C5]
            out_channels: Output channel dimension for all pyramid levels
        """
        super().__init__()

        self.out_channels = out_channels

        # Lateral connections (1x1 convs to reduce channel dimension)
        self.lateral_c2 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral_c3 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels_list[3], out_channels, 1)

        # Output convolutions (3x3 convs to reduce aliasing from upsampling)
        self.output_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.output_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.output_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Extra layers for P6 and P7 (from P5)
        self.p6_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Upsample x and add to y."""
        _, _, h, w = y.shape
        return F.interpolate(x, size=(h, w), mode='nearest') + y

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass building the feature pyramid.

        Args:
            features: List of backbone features [C2, C3, C4, C5]

        Returns:
            List of pyramid features [P3, P4, P5, P6, P7] with strides [8, 16, 32, 64, 128]
        """
        c2, c3, c4, c5 = features

        # Lateral connections
        l_c5 = self.lateral_c5(c5)
        l_c4 = self.lateral_c4(c4)
        l_c3 = self.lateral_c3(c3)
        l_c2 = self.lateral_c2(c2)

        # Top-down pathway with lateral connections
        p5 = l_c5
        p4 = self._upsample_add(p5, l_c4)
        p3 = self._upsample_add(p4, l_c3)

        # Note: We skip P2 for FCOS as small objects at stride 4 are rare
        # and it saves computation

        # Apply output convolutions
        p3 = self.output_p3(p3)
        p4 = self.output_p4(p4)
        p5 = self.output_p5(p5)

        # Generate P6 and P7 from P5
        p6 = self.p6_conv(p5)
        p6 = self.relu(p6)

        p7 = self.p7_conv(p6)
        p7 = self.relu(p7)

        return [p3, p4, p5, p6, p7]


if __name__ == "__main__":
    # Test FPN
    from backbone import ResNet18

    backbone = ResNet18()
    fpn = FPN(in_channels_list=backbone.out_channels, out_channels=256)

    print(f"FPN created with output channels: {fpn.out_channels}")

    # Test forward pass
    x = torch.randn(2, 3, 1080, 1920)
    backbone_features = backbone(x)
    pyramid_features = fpn(backbone_features)

    print("\nPyramid feature shapes:")
    strides = [8, 16, 32, 64, 128]
    for i, (feat, stride) in enumerate(zip(pyramid_features, strides)):
        print(f"  P{i+3} (stride {stride}): {feat.shape}")

    # Verify strides
    print("\nVerifying strides:")
    _, _, h, w = x.shape
    for i, (feat, stride) in enumerate(zip(pyramid_features, strides)):
        expected_h = h // stride
        expected_w = w // stride
        actual_h, actual_w = feat.shape[2], feat.shape[3]
        print(f"  P{i+3}: expected ({expected_h}, {expected_w}), got ({actual_h}, {actual_w})")
