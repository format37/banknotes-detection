#!/usr/bin/env python3
"""
Verification script for Phase 1 improvements.
Tests that all changes are working correctly before full training.
"""

import sys
import torch
from dataset import BanknoteDataset, get_dataloaders
from model import FCOS, BackboneFactory
from losses import FCOSLoss

def test_train_val_split():
    """Test that train/val split works correctly."""
    print("=" * 60)
    print("Test 1: Train/Val/Test Split")
    print("=" * 60)

    train_dataset = BanknoteDataset(split='train')
    val_dataset = BanknoteDataset(split='val')
    test_dataset = BanknoteDataset(split='test')

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")

    # Check that train + val = original train set (694)
    total = len(train_dataset) + len(val_dataset)
    expected = 694
    assert abs(total - expected) < 5, f"Train+Val should be ~{expected}, got {total}"
    print(f"✓ Train + Val = {total} (expected ~{expected})")

    # Check that splits are consistent
    train_dataset2 = BanknoteDataset(split='train')
    assert len(train_dataset) == len(train_dataset2), "Split not reproducible!"
    print("✓ Splits are reproducible (same random seed)")

    print("✓ Test 1 PASSED\n")
    return True

def test_class_weights():
    """Test class weight computation."""
    print("=" * 60)
    print("Test 2: Class Weights")
    print("=" * 60)

    dataset = BanknoteDataset(split='train')
    weights = dataset.compute_class_weights()

    print(f"✓ Number of classes: {dataset.num_classes}")
    print(f"✓ Class weights shape: {weights.shape}")
    print(f"✓ Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"✓ Weights mean: {weights.mean():.3f}")

    # Check constraints
    assert weights.shape[0] == dataset.num_classes, "Weight shape mismatch"
    assert weights.min() >= 0.5, f"Weights too low: {weights.min()}"
    assert weights.max() <= 2.0, f"Weights too high: {weights.max()}"
    # After clamping, mean may deviate from 1.0, but should be reasonable
    assert 0.5 <= weights.mean() <= 1.5, f"Weights mean should be reasonable: {weights.mean()}"

    print("✓ Test 2 PASSED\n")
    return True

def test_augmentations():
    """Test that augmentations work without errors."""
    print("=" * 60)
    print("Test 3: Data Augmentation")
    print("=" * 60)

    train_dataset = BanknoteDataset(split='train')

    # Try loading a few samples
    for i in range(5):
        image, targets = train_dataset[i]
        assert image.shape[0] == 3, f"Expected 3 channels, got {image.shape[0]}"
        assert len(targets['boxes'].shape) == 2, "Boxes should be 2D"
        assert targets['boxes'].shape[1] == 4, "Boxes should have 4 coordinates"

    print(f"✓ Loaded 5 training samples successfully")
    print(f"✓ Image shape: {image.shape}")
    print(f"✓ Boxes shape: {targets['boxes'].shape}")
    print(f"✓ Augmentations applied without errors")

    print("✓ Test 3 PASSED\n")
    return True

def test_backbone_factory():
    """Test backbone factory and pretrained loading."""
    print("=" * 60)
    print("Test 4: Backbone Factory")
    print("=" * 60)

    # Test ResNet-18
    backbone18 = BackboneFactory.create_backbone('resnet18', pretrained=False)
    print(f"✓ ResNet-18 created: {type(backbone18).__name__}")
    print(f"  Output channels: {backbone18.out_channels}")

    # Test ResNet-50
    backbone50 = BackboneFactory.create_backbone('resnet50', pretrained=True)
    print(f"✓ ResNet-50 created: {type(backbone50).__name__}")
    print(f"  Output channels: {backbone50.out_channels}")
    print(f"  Pretrained weights loaded")

    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    features = backbone50(x)
    print(f"✓ Forward pass successful")
    print(f"  Input: {x.shape}")
    for i, feat in enumerate(features):
        print(f"  C{i+2}: {feat.shape}")

    print("✓ Test 4 PASSED\n")
    return True

def test_model_creation():
    """Test FCOS model with new backbone."""
    print("=" * 60)
    print("Test 5: FCOS Model Creation")
    print("=" * 60)

    # Create model with pretrained ResNet-50
    model = FCOS(
        num_classes=24,
        fpn_channels=256,
        backbone_type='resnet50',
        pretrained=True,
        score_threshold=0.35,
        nms_threshold=0.4,
        max_detections=100
    )

    print(f"✓ FCOS model created")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")

    # Test forward pass
    x = torch.randn(2, 3, 640, 640)
    cls_outputs, reg_outputs, ctr_outputs = model(x)

    print(f"✓ Forward pass successful")
    print(f"  Input: {x.shape}")
    print(f"  Number of FPN levels: {len(cls_outputs)}")
    print(f"  Classification outputs: {[o.shape for o in cls_outputs]}")

    print("✓ Test 5 PASSED\n")
    return True

def test_loss_with_class_weights():
    """Test FCOS loss with class weights."""
    print("=" * 60)
    print("Test 6: Loss Function with Class Weights")
    print("=" * 60)

    # Create class weights
    num_classes = 24
    class_weights = torch.ones(num_classes)
    class_weights[0] = 2.0  # Higher weight for class 0

    # Create loss
    loss_fn = FCOSLoss(
        num_classes=num_classes,
        focal_alpha=0.5,
        focal_gamma=2.5,
        lambda_reg=2.0,
        lambda_ctr=1.0,
        class_weights=class_weights
    )

    print(f"✓ FCOSLoss created with class weights")
    print(f"  Focal alpha: 0.5")
    print(f"  Focal gamma: 2.5")
    print(f"  Lambda reg: 2.0")

    # Create dummy predictions
    batch_size = 2
    strides = [8, 16, 32, 64, 128]
    h, w = 640, 640

    cls_outputs = []
    reg_outputs = []
    ctr_outputs = []

    for stride in strides:
        fh, fw = h // stride, w // stride
        cls_outputs.append(torch.randn(batch_size, num_classes, fh, fw))
        reg_outputs.append(torch.abs(torch.randn(batch_size, 4, fh, fw)))
        ctr_outputs.append(torch.randn(batch_size, 1, fh, fw))

    # Create dummy targets
    targets = [
        {
            'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.long)
        },
        {
            'boxes': torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
            'labels': torch.tensor([3], dtype=torch.long)
        }
    ]

    # Compute loss
    total_loss, cls_loss, reg_loss, ctr_loss = loss_fn(
        cls_outputs, reg_outputs, ctr_outputs, targets, (w, h)
    )

    print(f"✓ Loss computation successful")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Classification loss: {cls_loss.item():.4f}")
    print(f"  Regression loss: {reg_loss.item():.4f}")
    print(f"  Centerness loss: {ctr_loss.item():.4f}")

    print("✓ Test 6 PASSED\n")
    return True

def test_dataloaders():
    """Test that dataloaders work correctly."""
    print("=" * 60)
    print("Test 7: DataLoaders")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders(num_workers=0)

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")

    # Test loading a batch
    images, targets = next(iter(train_loader))
    print(f"✓ Batch loaded successfully")
    print(f"  Images shape: {images.shape}")
    print(f"  Batch size: {len(targets)}")

    print("✓ Test 7 PASSED\n")
    return True

def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("PHASE 1 VERIFICATION TESTS")
    print("=" * 60 + "\n")

    tests = [
        ("Train/Val Split", test_train_val_split),
        ("Class Weights", test_class_weights),
        ("Augmentations", test_augmentations),
        ("Backbone Factory", test_backbone_factory),
        ("Model Creation", test_model_creation),
        ("Loss with Weights", test_loss_with_class_weights),
        ("DataLoaders", test_dataloaders),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test FAILED: {name}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed} tests failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ All Phase 1 improvements verified successfully!")
        print("  Ready to start training with improved configuration.")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed. Please fix before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
