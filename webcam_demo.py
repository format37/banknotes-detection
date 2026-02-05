"""
Webcam inference demo for FCOS banknote detector.
Supports real-time webcam capture and video file input with optional display and recording.
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import BanknoteDataset
from model import FCOS


# Color palette for visualization (from inference.py)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128),
    (128, 128, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255)
]


def find_best_checkpoint():
    """Find the most recent checkpoint_best.pt file."""
    outputs_dir = Path('outputs')
    if not outputs_dir.exists():
        return None

    checkpoints = list(outputs_dir.glob('*/checkpoint_best.pt'))
    if not checkpoints:
        return None

    # Sort by modification time, most recent first
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(checkpoints[0])


def setup_model(checkpoint_path, config_path, device):
    """Load model following inference.py pattern."""
    print(f'Loading checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint or load from file
    config = checkpoint.get('config', None)
    if config is None:
        with open(config_path, 'r') as f:
            config = json.load(f)

    # Load dataset for class names
    dataset = BanknoteDataset(split='test', config_path=config_path)
    class_names = dataset.idx_to_class

    print(f'Number of classes: {dataset.num_classes}')
    print(f'Class names: {list(class_names.values())}')

    # Create model
    model = FCOS(
        num_classes=dataset.num_classes,
        fpn_channels=config['fpn_channels'],
        strides=config['strides'],
        score_threshold=config.get('score_threshold', 0.05),
        nms_threshold=config.get('nms_threshold', 0.5),
        max_detections=config.get('max_detections', 100)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, class_names, config


def draw_detections(
    image: np.ndarray,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_names: dict,
    score_threshold: float = 0.3
) -> np.ndarray:
    """Draw detection boxes on image (from inference.py)."""
    image = image.copy()

    for i in range(len(boxes)):
        if scores[i] < score_threshold:
            continue

        box = boxes[i].cpu().numpy().astype(int)
        score = scores[i].cpu().item()
        label = labels[i].cpu().item()

        x1, y1, x2, y2 = box
        color = COLORS[label % len(COLORS)]
        class_name = class_names.get(label, f'class_{label}')

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_text = f'{class_name}: {score:.2f}'
        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_h - 10),
            (x1 + text_w + 4, y1),
            color, -1
        )

        # Draw label text
        cv2.putText(
            image, label_text,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return image


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Add FPS counter overlay to frame."""
    # Color-code FPS: green (>=20), yellow (10-20), red (<10)
    if fps >= 20:
        color = (0, 255, 0)
    elif fps >= 10:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.putText(
        frame, f'FPS: {fps:.1f}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
    )
    return frame


def parse_args():
    parser = argparse.ArgumentParser(
        description='Real-time banknote detection demo'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint (default: auto-detect latest checkpoint_best.pt)'
    )
    parser.add_argument(
        '--config', type=str, default='config.json',
        help='Path to config file'
    )
    parser.add_argument(
        '--input', type=str, default=None,
        help='Input video file path (if not specified, use webcam)'
    )
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera device ID (default: 0, only for webcam mode)'
    )
    parser.add_argument(
        '--output', type=str, default='webcam_demo.mp4',
        help='Output video file path'
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help='Run headless without preview window'
    )
    parser.add_argument(
        '--score-threshold', type=float, default=0.3,
        help='Detection confidence threshold (default: 0.3)'
    )
    parser.add_argument(
        '--max-detections', type=int, default=10,
        help='Maximum detections to show (default: 10)'
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Target output video FPS (default: 30)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Find or validate checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint()
        if checkpoint_path is None:
            print('Error: No checkpoint found. Available checkpoints:')
            outputs_dir = Path('outputs')
            if outputs_dir.exists():
                for ckpt in outputs_dir.glob('*/checkpoint*.pt'):
                    print(f'  {ckpt}')
            else:
                print('  outputs/ directory not found')
            return
        print(f'Auto-detected checkpoint: {checkpoint_path}')
    elif not Path(checkpoint_path).exists():
        print(f'Error: Checkpoint not found: {checkpoint_path}')
        print('Available checkpoints:')
        outputs_dir = Path('outputs')
        if outputs_dir.exists():
            for ckpt in outputs_dir.glob('*/checkpoint*.pt'):
                print(f'  {ckpt}')
        return

    # Setup model
    try:
        model, class_names, config = setup_model(checkpoint_path, args.config, device)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print('CUDA out of memory, falling back to CPU...')
            device = torch.device('cpu')
            model, class_names, config = setup_model(checkpoint_path, args.config, device)
        else:
            raise

    # Get target resolution from config
    target_w, target_h = config['image_size']
    print(f'Target resolution: {target_w}x{target_h}')

    # Open video source
    if args.input:
        print(f'Opening video file: {args.input}')
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f'Error: Cannot open video file: {args.input}')
            return
    else:
        print(f'Opening webcam (device {args.camera})...')
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f'Error: Cannot access camera {args.camera}')
            print('Try alternative camera IDs: --camera 1, --camera 2, etc.')
            return

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (target_w, target_h))

    if not out.isOpened():
        print('Warning: Video codec mp4v not available, trying avc1...')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(args.output, fourcc, args.fps, (target_w, target_h))

        if not out.isOpened():
            print('Warning: Video codec avc1 not available, trying XVID...')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(args.output, fourcc, args.fps, (target_w, target_h))

            if not out.isOpened():
                print('Error: Cannot initialize video writer')
                cap.release()
                return

    print(f'Recording to: {args.output}')

    # FPS tracking
    frame_times = deque(maxlen=30)
    last_time = time.time()
    frame_count = 0
    last_console_update = time.time()
    corrupted_frames = 0
    display_available = not args.no_display

    # Test if display is available
    if not args.no_display:
        try:
            # Try to create a test window to check if display works
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('Display Test', test_frame)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        except cv2.error:
            print('Warning: Display not available (OpenCV GUI support missing)')
            print('Running in headless mode. Use --no-display to suppress this warning.')
            display_available = False

    if display_available:
        print('Display mode: Press "q" to stop')
    else:
        print('Headless mode: Press Ctrl+C to stop')

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                if args.input:
                    print('End of video file')
                else:
                    print('Error: Cannot read from camera')
                break

            try:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize to target resolution
                resized = cv2.resize(rgb_frame, (target_w, target_h))

                # Convert to tensor
                image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                image_tensor = image_tensor.unsqueeze(0).to(device)

                # Run inference
                with torch.no_grad():
                    results = model.inference(image_tensor)

                # Extract results
                boxes = results[0]['boxes']
                scores = results[0]['scores']
                labels = results[0]['labels']

                # Convert back to BGR for OpenCV
                bgr_frame = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

                # Draw detections
                annotated_frame = draw_detections(
                    bgr_frame, boxes, scores, labels, class_names, args.score_threshold
                )

                # Calculate FPS
                current_time = time.time()
                frame_times.append(current_time - last_time)
                last_time = current_time

                fps = len(frame_times) / sum(frame_times) if frame_times else 0

                # Draw FPS
                annotated_frame = draw_fps(annotated_frame, fps)

                # Display if available
                if display_available:
                    try:
                        cv2.imshow('Banknote Detection Demo', annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print('Stopped by user')
                            break
                    except cv2.error:
                        # Display failed, switch to headless
                        display_available = False
                        print('Display failed, continuing in headless mode...')

                # Write to output video
                out.write(annotated_frame)

                frame_count += 1

                # Console updates in headless mode
                if not display_available and (current_time - last_console_update) >= 5.0:
                    print(f'Frames: {frame_count}, FPS: {fps:.1f}')
                    last_console_update = current_time

            except cv2.error as e:
                if 'cvShowImage' in str(e) or 'cvDestroyAllWindows' in str(e):
                    # OpenCV GUI error, switch to headless
                    display_available = False
                    print('Display error detected, continuing in headless mode...')
                    continue
                else:
                    raise
            except Exception as e:
                # Only treat as corrupted frame if not a display error
                if 'cvShowImage' not in str(e):
                    print(f'Error processing frame {frame_count}: {e}')
                    corrupted_frames += 1
                    if corrupted_frames > 10:
                        print('Too many corrupted frames, stopping...')
                        break
                continue

    except KeyboardInterrupt:
        print('\nStopped by user (Ctrl+C)')

    finally:
        # Cleanup
        cap.release()
        out.release()
        if display_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # Ignore cleanup errors

        print(f'\nProcessed {frame_count} frames')
        if corrupted_frames > 0:
            print(f'Skipped {corrupted_frames} corrupted frames')
        print(f'Video saved to: {args.output}')


if __name__ == '__main__':
    main()
