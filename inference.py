"""
Inference script for FCOS banknote detector.
Runs detection on images and visualizes results.
"""

import argparse
import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image

from dataset import BanknoteDataset
from model import FCOS


# Color palette for visualization
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128),
    (128, 128, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255)
]


def parse_args():
    parser = argparse.ArgumentParser(description='Run FCOS inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image (if not specified, uses test dataset)')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                        help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize from test set')
    parser.add_argument('--score-threshold', type=float, default=0.3,
                        help='Score threshold for visualization')
    return parser.parse_args()


def draw_detections(
    image: np.ndarray,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_names: dict,
    score_threshold: float = 0.3
) -> np.ndarray:
    """Draw detection boxes on image."""
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


def draw_ground_truth(
    image: np.ndarray,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    class_names: dict
) -> np.ndarray:
    """Draw ground truth boxes on image."""
    image = image.copy()

    for i in range(len(boxes)):
        box = boxes[i].cpu().numpy().astype(int)
        label = labels[i].cpu().item()

        x1, y1, x2, y2 = box
        class_name = class_names.get(label, f'class_{label}')

        # Draw dashed box (using dotted line approximation)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Draw label
        cv2.putText(
            image, f'GT: {class_name}',
            (x1, y2 + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )

    return image


@torch.no_grad()
def run_inference(
    model: FCOS,
    image: torch.Tensor,
    device: torch.device
) -> dict:
    """Run inference on a single image."""
    model.eval()

    image = image.unsqueeze(0).to(device)
    results = model.inference(image)

    return results[0]


def main():
    args = parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    print(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config = checkpoint.get('config', None)
    if config is None:
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Load dataset for class names
    dataset = BanknoteDataset(split='test', config_path=args.config)
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

    if args.image:
        # Run on single image
        print(f'Running inference on: {args.image}')

        # Load and preprocess image
        image = cv2.imread(args.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        target_w, target_h = config['image_size']
        image_resized = cv2.resize(image, (target_w, target_h))

        # Convert to tensor
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        # Run inference
        result = run_inference(model, image_tensor, device)

        print(f'Found {len(result["boxes"])} detections')

        # Draw results
        vis_image = draw_detections(
            image_resized, result['boxes'], result['scores'],
            result['labels'], class_names, args.score_threshold
        )

        # Save result
        output_path = os.path.join(
            args.output_dir,
            os.path.basename(args.image).replace('.', '_detected.')
        )
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f'Saved result to: {output_path}')

    else:
        # Run on test dataset samples
        print(f'Running inference on {args.num_samples} test samples...')

        # Select random samples
        indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

        for idx in indices:
            image, target = dataset[idx]

            # Run inference
            result = run_inference(model, image, device)

            # Convert image tensor to numpy for visualization
            image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Create side-by-side visualization
            vis_pred = draw_detections(
                image_np.copy(), result['boxes'], result['scores'],
                result['labels'], class_names, args.score_threshold
            )
            vis_gt = draw_ground_truth(
                image_np.copy(), target['boxes'], target['labels'], class_names
            )

            # Combine images
            combined = np.hstack([vis_gt, vis_pred])

            # Add labels
            h, w = combined.shape[:2]
            cv2.putText(combined, 'Ground Truth', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, 'Predictions', (w // 2 + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Save result
            output_path = os.path.join(args.output_dir, f'sample_{idx}.jpg')
            cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            print(f'  Sample {idx}: {len(result["boxes"])} detections, '
                  f'{len(target["boxes"])} ground truth')

        print(f'\nResults saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
