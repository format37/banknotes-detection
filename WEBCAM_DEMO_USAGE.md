# Webcam Demo Usage Guide

Real-time banknote detection demo supporting webcam and video file input.

## Quick Start

```bash
# Headless mode (recommended for Linux/SSH environments)
python webcam_demo.py --no-display --output demo.mp4

# With display (if GUI support available)
python webcam_demo.py

# Press 'q' to stop (display mode) or Ctrl+C (headless mode)
```

**Note for Linux users**: If you see "Display not available" warnings, use `--no-display` flag. The script will auto-detect and switch to headless mode, but this suppresses the warning.

## Command-Line Options

```
--checkpoint CHECKPOINT    Path to model checkpoint (default: auto-detect latest)
--config CONFIG           Path to config.json (default: config.json)
--input INPUT             Input video file (if not specified, uses webcam)
--camera CAMERA           Camera device ID (default: 0)
--output OUTPUT           Output video file (default: webcam_demo.mp4)
--no-display              Run headless without preview window
--score-threshold FLOAT   Detection confidence threshold (default: 0.3)
--max-detections INT      Maximum detections to show (default: 10)
--fps INT                 Target output video FPS (default: 30)
```

## Usage Examples

### 1. Basic Webcam Mode
```bash
python webcam_demo.py
```
- Opens default camera (device 0)
- Shows real-time preview window
- Records to `webcam_demo.mp4`
- Press 'q' to stop

### 2. Specific Checkpoint and Camera
```bash
python webcam_demo.py \
  --checkpoint outputs/20260202_154052/checkpoint_best.pt \
  --camera 1
```

### 3. Process Existing Video File
```bash
python webcam_demo.py \
  --input test_video.mp4 \
  --output annotated_output.mp4
```

### 4. Headless Background Recording
```bash
python webcam_demo.py \
  --no-display \
  --output background_recording.mp4
```
- No display window
- Prints FPS to console every 5 seconds
- Press Ctrl+C to stop

### 5. High-Sensitivity Detection
```bash
python webcam_demo.py \
  --score-threshold 0.2 \
  --output high_sensitivity_demo.mp4
```

### 6. Process Video File Headless
```bash
python webcam_demo.py \
  --input my_video.mp4 \
  --output processed.mp4 \
  --no-display
```

## Features

- **Auto-checkpoint detection**: Automatically finds latest `checkpoint_best.pt`
- **Real-time FPS counter**: Color-coded performance indicator
  - Green: ≥20 FPS
  - Yellow: 10-20 FPS
  - Red: <10 FPS
- **YouTube-compatible output**: MP4 format (1920×1080 @ 30fps)
- **Dual mode**: Webcam or video file input
- **Headless operation**: Run without display for batch processing
- **Error recovery**: Skips corrupted frames, falls back to CPU if CUDA OOM

## Output Format

- **Resolution**: 1920×1080 (matches model input size)
- **Format**: MP4 (H.264 codec preferred)
- **Frame rate**: 30 FPS (configurable)
- **Annotations**: Bounding boxes with class names and confidence scores

## Keyboard Controls

- **q**: Quit and save recording (in display mode)
- **Ctrl+C**: Stop recording (in headless mode)

## Troubleshooting

### Display not available (Linux/Headless environments)
```
Warning: Display not available (OpenCV GUI support missing)
Running in headless mode. Use --no-display to suppress this warning.
```
**Explanation**: Your OpenCV installation lacks GUI support (GTK+). This is common in:
- SSH sessions without X11 forwarding
- Conda environments without `opencv` GUI dependencies
- Docker containers without display access

**Solution**: Use `--no-display` flag to run in headless mode:
```bash
python webcam_demo.py --no-display --output demo.mp4
```

The script will automatically detect this and switch to headless mode, but using `--no-display` explicitly suppresses the warning.

### Camera not accessible
```
Error: Cannot access camera 0
Try alternative camera IDs: --camera 1, --camera 2, etc.
```
**Solution**: Try `--camera 1` or `--camera 2`

### No checkpoint found
```
Error: No checkpoint found
```
**Solution**: Specify checkpoint manually:
```bash
python webcam_demo.py --checkpoint outputs/20260202_172545/checkpoint_best.pt
```

### CUDA out of memory
```
CUDA out of memory, falling back to CPU...
```
**Solution**: Script automatically falls back to CPU. For better performance, reduce batch processing or use a smaller model.

### Video codec not available
```
Warning: Video codec mp4v not available, trying avc1...
```
**Solution**: Script automatically tries fallback codecs (mp4v → avc1 → XVID)

## Performance Tips

- **GPU recommended**: ~20-30 FPS on GPU, ~5-10 FPS on CPU
- **Lower threshold for more detections**: `--score-threshold 0.2`
- **Higher threshold for cleaner output**: `--score-threshold 0.4`
- **Reduce max detections**: `--max-detections 5` for faster processing

## Available Checkpoints

```bash
# Latest (recommended)
outputs/20260202_172545/checkpoint_best.pt

# Previous run
outputs/20260202_154052/checkpoint_best.pt
```

## Model Details

- **Classes**: 18 object categories (Russian ruble denominations + related objects)
- **Input size**: 1920×1080
- **Architecture**: FCOS (Fully Convolutional One-Stage Object Detector)
- **Backbone**: ResNet-50 + FPN
