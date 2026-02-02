# FCOS Banknote Detector

A PyTorch implementation of FCOS (Fully Convolutional One-Stage Object Detection) from scratch for detecting Russian ruble banknotes.

## Features

- **Custom Implementation**: All components built from scratch without pretrained weights
  - ResNet-18 backbone
  - Feature Pyramid Network (FPN)
  - FCOS detection head with centerness prediction

- **Custom Loss Functions**:
  - Focal Loss for classification (handles class imbalance)
  - GIoU Loss for bounding box regression (better gradient for non-overlapping boxes)
  - Centerness Loss for improving detection quality

- **Training Pipeline**:
  - Mixed precision training (AMP) for RTX 4090 efficiency
  - Gradient clipping
  - Learning rate warmup + cosine annealing
  - Checkpointing and early stopping
  - TensorBoard logging

- **Evaluation**:
  - mAP@0.5 (PASCAL VOC standard)
  - mAP@0.5:0.95 (COCO standard)
  - Per-class AP
  - Inference speed (FPS)

## Project Structure

```
banknotes/
├── config.json              # Hyperparameters
├── dataset.py               # Dataset loading and FCOS target encoding
├── model/
│   ├── backbone.py          # ResNet-18 from scratch
│   ├── fpn.py               # Feature Pyramid Network
│   ├── head.py              # Classification, regression, centerness heads
│   └── fcos.py              # Full FCOS detector
├── losses/
│   ├── focal_loss.py        # Focal Loss implementation
│   ├── giou_loss.py         # GIoU Loss implementation
│   └── fcos_loss.py         # Combined FCOS loss
├── utils/
│   ├── box_utils.py         # IoU, NMS, box encoding/decoding
│   └── metrics.py           # mAP calculation
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── inference.py             # Inference and visualization
└── benchmark.py             # Comprehensive benchmarking
```

## Installation

```bash
# Clone the repository
cd ~/projects/banknotes

# Install dependencies
pip install -r requirements.txt

# Set up HuggingFace token for private dataset access
echo "HF_TOKEN=your_token_here" > .env
```

## Dataset

This project uses the `format37/russian-rubles-banknotes` private HuggingFace dataset:
- **Images**: 1,001 (694 train / 307 test)
- **Resolution**: 1920×1080
- **Classes**: 24 banknote categories
- **Format**: COCO-style bounding boxes

### Setup

1. Get access to the dataset on [HuggingFace](https://huggingface.co/datasets/format37/russian-rubles-banknotes)
2. Create a HuggingFace token at https://huggingface.co/settings/tokens
3. Add token to `.env`:
   ```bash
   echo "HF_TOKEN=hf_your_token_here" > .env
   ```

The dataset is downloaded automatically on first run. To verify access:
```bash
python -c "from dataset import BanknoteDataset; ds = BanknoteDataset('train'); print(f'Loaded {len(ds)} samples, {ds.num_classes} classes')"
```

## Configuration

Edit `config.json` to modify hyperparameters:

```json
{
  "batch_size": 4,
  "epochs": 100,
  "lr": 1e-4,
  "image_size": [1920, 1080],
  "num_classes": 24,
  "fpn_channels": 256,
  "strides": [8, 16, 32, 64, 128]
}
```

For faster training, reduce image size:
```json
{
  "image_size": [800, 800],
  "batch_size": 8
}
```

## Training

```bash
# Train with default config
python train.py

# Train with custom settings
python train.py --epochs 50 --output-dir my_experiment

# Resume from checkpoint
python train.py --resume outputs/checkpoint_latest.pt
```

Training outputs:
- `outputs/<timestamp>/checkpoint_best.pt` - Best model
- `outputs/<timestamp>/checkpoint_latest.pt` - Latest checkpoint
- `outputs/<timestamp>/logs/` - TensorBoard logs

View training progress:
```bash
tensorboard --logdir outputs/
```

## Evaluation

```bash
# Evaluate on test set
python evaluate.py --checkpoint outputs/<timestamp>/checkpoint_best.pt

# Save results to JSON
python evaluate.py --checkpoint checkpoint_best.pt --output results.json
```

## Inference

```bash
# Run on test dataset samples
python inference.py --checkpoint checkpoint_best.pt --num-samples 10

# Run on custom image
python inference.py --checkpoint checkpoint_best.pt --image path/to/image.jpg
```

## Benchmarking

```bash
# Generate comprehensive benchmark report
python benchmark.py --checkpoint checkpoint_best.pt --output-dir benchmark_results
```

This creates:
- `benchmark_report.md` - Markdown report with all metrics
- `per_class_ap.png` - Per-class AP visualization
- `detection_distribution.png` - GT vs predictions per class
- `benchmark_results.json` - Raw results data

## Model Architecture

### FCOS Overview

FCOS is an anchor-free, fully convolutional object detector that:
1. Extracts multi-scale features using ResNet-18 + FPN
2. Predicts at each spatial location:
   - Class probabilities (C channels)
   - Box regression (l, t, r, b distances to edges)
   - Centerness score (1 channel)
3. Uses centerness to down-weight low-quality predictions

### Network Details

- **Backbone**: ResNet-18 (11.7M parameters)
  - 4 stages with strides [4, 8, 16, 32]
  - Output channels: [64, 128, 256, 512]

- **FPN**: 5 levels (P3-P7)
  - Strides: [8, 16, 32, 64, 128]
  - 256 channels per level

- **Detection Head**:
  - 4 shared conv layers per branch (cls/reg)
  - GroupNorm + ReLU
  - Learnable scale parameters for regression

### Loss Functions

1. **Focal Loss** (Classification):
   ```
   FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
   ```
   - α = 0.25, γ = 2.0
   - Down-weights easy negatives

2. **GIoU Loss** (Regression):
   ```
   GIoU = IoU - |C \ (A ∪ B)| / |C|
   Loss = 1 - GIoU
   ```
   - Works with non-overlapping boxes

3. **Centerness Loss** (BCE):
   ```
   centerness = sqrt((min(l,r)/max(l,r)) * (min(t,b)/max(t,b)))
   ```

## Expected Results

- **Training time**: ~3-6 hours on RTX 4090 (native resolution)
- **mAP@0.5**: 0.5-0.7 (with 694 training images)
- **mAP@0.5:0.95**: 0.3-0.5
- **Inference speed**: 15-25 FPS at 1920×1080, 40+ FPS at 800×800

## HuggingFace Upload

After training, upload to HuggingFace:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="outputs/<timestamp>",
    repo_id="format37/fcos-banknotes-detector",
    repo_type="model"
)
```

## License

MIT License

## References

- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355)
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
