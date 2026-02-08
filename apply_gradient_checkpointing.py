"""
Utility to apply gradient checkpointing to FCOS model.
Reduces memory usage by ~30-40% with ~15-20% slower training.
"""

import torch
from torch.utils.checkpoint import checkpoint


def apply_gradient_checkpointing(model):
    """
    Apply gradient checkpointing to backbone layers.

    Gradient checkpointing trades compute for memory by:
    - Not storing intermediate activations during forward pass
    - Recomputing them during backward pass when needed

    This saves significant memory with minimal speed impact.
    """

    # Check if backbone has layers to checkpoint
    if not hasattr(model.backbone, 'layer1'):
        print("  Warning: Backbone doesn't have layer1, skipping checkpointing")
        return model

    # Wrapper class to apply checkpointing
    class CheckpointedLayer(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            # Use checkpoint during training, normal forward during eval
            if self.training:
                return checkpoint(self.layer, x, use_reentrant=False)
            else:
                return self.layer(x)

    # Apply to each backbone layer
    layers_checkpointed = []
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if hasattr(model.backbone, layer_name):
            original_layer = getattr(model.backbone, layer_name)
            checkpointed_layer = CheckpointedLayer(original_layer)
            setattr(model.backbone, layer_name, checkpointed_layer)
            layers_checkpointed.append(layer_name)

    print(f"  ✓ Gradient checkpointing applied to: {', '.join(layers_checkpointed)}")
    print(f"  ✓ Expected memory savings: ~30-40%")
    print(f"  ✓ Expected slowdown: ~15-20%")

    return model


if __name__ == "__main__":
    # Test
    from model import FCOS

    model = FCOS(
        num_classes=11,
        backbone_type='resnet50',
        pretrained=False
    )

    print("Before checkpointing:")
    print(f"  Backbone layers: {hasattr(model.backbone, 'layer1')}")

    model = apply_gradient_checkpointing(model)

    print("\nAfter checkpointing:")
    print(f"  Layer1 type: {type(model.backbone.layer1)}")

    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print("\n✓ Forward pass works!")
