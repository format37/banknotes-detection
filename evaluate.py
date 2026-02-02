"""
Evaluation script for FCOS banknote detector.
Computes mAP and per-class metrics on the test set.
"""

import argparse
import json
import os

import torch
from tqdm import tqdm

from dataset import BanknoteDataset, get_dataloaders
from model import FCOS
from utils.metrics import compute_map_coco, compute_ap_per_class


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate FCOS banknote detector')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: FCOS,
    val_loader,
    dataset: BanknoteDataset,
    device: torch.device
) -> dict:
    """Run full evaluation."""
    model.eval()

    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []

    print('Running inference...')
    for images, targets in tqdm(val_loader, desc='Evaluating'):
        images = images.to(device)

        results = model.inference(images)

        for result, target in zip(results, targets):
            all_pred_boxes.append(result['boxes'].cpu())
            all_pred_scores.append(result['scores'].cpu())
            all_pred_labels.append(result['labels'].cpu())
            all_gt_boxes.append(target['boxes'])
            all_gt_labels.append(target['labels'])

    # Compute overall mAP
    print('\nComputing metrics...')
    mAP_coco, mAP_50, metrics = compute_map_coco(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_gt_boxes, all_gt_labels, dataset.num_classes
    )

    # Compute per-class AP at IoU=0.5
    per_class_ap = compute_ap_per_class(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_gt_boxes, all_gt_labels, dataset.num_classes,
        iou_threshold=0.5
    )

    # Count detections
    total_preds = sum(len(boxes) for boxes in all_pred_boxes)
    total_gt = sum(len(boxes) for boxes in all_gt_boxes)

    results = {
        'mAP@0.5': mAP_50,
        'mAP@0.5:0.95': mAP_coco,
        'total_predictions': total_preds,
        'total_ground_truth': total_gt,
        'num_images': len(all_pred_boxes),
        'per_class_ap': {}
    }

    # Add per-class results
    for class_idx, ap in per_class_ap.items():
        class_name = dataset.idx_to_class.get(class_idx, f'class_{class_idx}')
        results['per_class_ap'][class_name] = ap

    return results


def main():
    args = parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load checkpoint
    print(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config = checkpoint.get('config', None)
    if config is None:
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Create dataloader
    _, val_loader = get_dataloaders(
        config_path=args.config,
        num_workers=args.num_workers
    )
    dataset = val_loader.dataset

    print(f'Test samples: {len(dataset)}')
    print(f'Number of classes: {dataset.num_classes}')

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

    # Evaluate
    results = evaluate(model, val_loader, dataset, device)

    # Print results
    print('\n' + '=' * 60)
    print('EVALUATION RESULTS')
    print('=' * 60)
    print(f'mAP@0.5:        {results["mAP@0.5"]:.4f}')
    print(f'mAP@0.5:0.95:   {results["mAP@0.5:0.95"]:.4f}')
    print(f'Total predictions:    {results["total_predictions"]}')
    print(f'Total ground truth:   {results["total_ground_truth"]}')
    print(f'Number of images:     {results["num_images"]}')

    print('\nPer-class AP@0.5:')
    print('-' * 40)
    for class_name, ap in sorted(results['per_class_ap'].items(), key=lambda x: -x[1]):
        print(f'  {class_name:25s}: {ap:.4f}')

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to: {args.output}')


if __name__ == '__main__':
    main()
