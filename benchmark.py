"""
Benchmark script for FCOS banknote detector.
Generates comprehensive evaluation report with metrics, visualizations, and timing.
"""

import argparse
import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset import BanknoteDataset, get_dataloaders
from model import FCOS
from utils.metrics import compute_map_coco, compute_ap_per_class


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark FCOS detector')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Output directory for benchmark results')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--warmup-iterations', type=int, default=10,
                        help='Number of warmup iterations for timing')
    parser.add_argument('--timing-iterations', type=int, default=100,
                        help='Number of iterations for timing')
    return parser.parse_args()


def measure_inference_speed(
    model: FCOS,
    device: torch.device,
    image_size: tuple,
    warmup_iterations: int = 10,
    timing_iterations: int = 100
) -> dict:
    """Measure inference speed in FPS."""
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size[1], image_size[0]).to(device)

    # Warmup
    print('Warming up...')
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model.inference(dummy_input)

    # Synchronize CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timing
    print('Measuring inference speed...')
    times = []
    with torch.no_grad():
        for _ in tqdm(range(timing_iterations)):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model.inference(dummy_input)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times)

    return {
        'mean_latency_ms': np.mean(times) * 1000,
        'std_latency_ms': np.std(times) * 1000,
        'min_latency_ms': np.min(times) * 1000,
        'max_latency_ms': np.max(times) * 1000,
        'fps': 1.0 / np.mean(times)
    }


@torch.no_grad()
def evaluate_full(
    model: FCOS,
    val_loader,
    dataset: BanknoteDataset,
    device: torch.device
) -> dict:
    """Run full evaluation with detailed metrics."""
    model.eval()

    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []

    inference_times = []

    print('Running evaluation...')
    for images, targets in tqdm(val_loader, desc='Evaluating'):
        images = images.to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()

        results = model.inference(images)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()

        inference_times.append((end - start) / len(images))

        for result, target in zip(results, targets):
            all_pred_boxes.append(result['boxes'].cpu())
            all_pred_scores.append(result['scores'].cpu())
            all_pred_labels.append(result['labels'].cpu())
            all_gt_boxes.append(target['boxes'])
            all_gt_labels.append(target['labels'])

    # Compute mAP at multiple thresholds
    print('\nComputing metrics...')
    mAP_coco, mAP_50, metrics = compute_map_coco(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_gt_boxes, all_gt_labels, dataset.num_classes
    )

    # Per-class AP
    per_class_ap = compute_ap_per_class(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_gt_boxes, all_gt_labels, dataset.num_classes, 0.5
    )

    # Detection statistics
    total_preds = sum(len(boxes) for boxes in all_pred_boxes)
    total_gt = sum(len(boxes) for boxes in all_gt_boxes)

    # Per-class statistics
    class_stats = {}
    for class_idx in range(dataset.num_classes):
        class_name = dataset.idx_to_class.get(class_idx, f'class_{class_idx}')
        num_gt = sum((labels == class_idx).sum().item() for labels in all_gt_labels)
        num_pred = sum((labels == class_idx).sum().item() for labels in all_pred_labels)
        class_stats[class_name] = {
            'num_ground_truth': num_gt,
            'num_predictions': num_pred,
            'ap': per_class_ap.get(class_idx, 0.0)
        }

    return {
        'mAP@0.5': mAP_50,
        'mAP@0.5:0.95': mAP_coco,
        'detailed_metrics': metrics,
        'per_class_stats': class_stats,
        'total_predictions': total_preds,
        'total_ground_truth': total_gt,
        'num_images': len(all_pred_boxes),
        'mean_inference_time_ms': np.mean(inference_times) * 1000,
        'per_class_ap': {
            dataset.idx_to_class.get(k, f'class_{k}'): v
            for k, v in per_class_ap.items()
        }
    }


def plot_per_class_ap(results: dict, output_path: str):
    """Create bar chart of per-class AP."""
    class_names = list(results['per_class_ap'].keys())
    ap_values = list(results['per_class_ap'].values())

    # Sort by AP value
    sorted_pairs = sorted(zip(ap_values, class_names), reverse=True)
    ap_values, class_names = zip(*sorted_pairs)

    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(class_names)), ap_values, color='steelblue')
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel('AP@0.5')
    plt.title('Per-Class Average Precision')
    plt.xlim(0, 1)

    # Add value labels
    for bar, val in zip(bars, ap_values):
        plt.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_detection_distribution(results: dict, output_path: str):
    """Plot distribution of predictions vs ground truth per class."""
    stats = results['per_class_stats']
    class_names = list(stats.keys())
    gt_counts = [stats[c]['num_ground_truth'] for c in class_names]
    pred_counts = [stats[c]['num_predictions'] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, gt_counts, width, label='Ground Truth', color='green', alpha=0.7)
    plt.bar(x + width/2, pred_counts, width, label='Predictions', color='blue', alpha=0.7)

    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Ground Truth vs Predictions per Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_report(results: dict, speed_results: dict, output_dir: str, config: dict):
    """Generate markdown benchmark report."""
    report_path = os.path.join(output_dir, 'benchmark_report.md')

    with open(report_path, 'w') as f:
        f.write('# FCOS Banknote Detector Benchmark Report\n\n')
        f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')

        f.write('## Configuration\n\n')
        f.write(f'- **Image size**: {config["image_size"]}\n')
        f.write(f'- **Number of classes**: {config["num_classes"]}\n')
        f.write(f'- **FPN channels**: {config["fpn_channels"]}\n')
        f.write(f'- **Strides**: {config["strides"]}\n\n')

        f.write('## Overall Metrics\n\n')
        f.write(f'| Metric | Value |\n')
        f.write(f'|--------|-------|\n')
        f.write(f'| mAP@0.5 | {results["mAP@0.5"]:.4f} |\n')
        f.write(f'| mAP@0.5:0.95 | {results["mAP@0.5:0.95"]:.4f} |\n')
        f.write(f'| Total predictions | {results["total_predictions"]} |\n')
        f.write(f'| Total ground truth | {results["total_ground_truth"]} |\n')
        f.write(f'| Number of images | {results["num_images"]} |\n\n')

        f.write('## Inference Speed\n\n')
        f.write(f'| Metric | Value |\n')
        f.write(f'|--------|-------|\n')
        f.write(f'| FPS | {speed_results["fps"]:.2f} |\n')
        f.write(f'| Mean latency | {speed_results["mean_latency_ms"]:.2f} ms |\n')
        f.write(f'| Std latency | {speed_results["std_latency_ms"]:.2f} ms |\n')
        f.write(f'| Min latency | {speed_results["min_latency_ms"]:.2f} ms |\n')
        f.write(f'| Max latency | {speed_results["max_latency_ms"]:.2f} ms |\n\n')

        f.write('## Per-Class Performance\n\n')
        f.write('| Class | AP@0.5 | GT Count | Pred Count |\n')
        f.write('|-------|--------|----------|------------|\n')

        for class_name, stats in sorted(
            results['per_class_stats'].items(),
            key=lambda x: -x[1]['ap']
        ):
            f.write(f'| {class_name} | {stats["ap"]:.4f} | '
                    f'{stats["num_ground_truth"]} | {stats["num_predictions"]} |\n')

        f.write('\n## Visualizations\n\n')
        f.write('![Per-Class AP](per_class_ap.png)\n\n')
        f.write('![Detection Distribution](detection_distribution.png)\n\n')

    print(f'Report saved to: {report_path}')


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

    # Create dataloader
    _, val_loader = get_dataloaders(
        config_path=args.config,
        num_workers=args.num_workers
    )
    dataset = val_loader.dataset

    # Update config with actual number of classes
    config['num_classes'] = dataset.num_classes

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

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    # Measure inference speed
    print('\n' + '=' * 60)
    print('INFERENCE SPEED BENCHMARK')
    print('=' * 60)
    speed_results = measure_inference_speed(
        model, device, tuple(config['image_size']),
        args.warmup_iterations, args.timing_iterations
    )
    print(f'FPS: {speed_results["fps"]:.2f}')
    print(f'Mean latency: {speed_results["mean_latency_ms"]:.2f} ms')

    # Run full evaluation
    print('\n' + '=' * 60)
    print('ACCURACY BENCHMARK')
    print('=' * 60)
    results = evaluate_full(model, val_loader, dataset, device)

    print(f'\nmAP@0.5:      {results["mAP@0.5"]:.4f}')
    print(f'mAP@0.5:0.95: {results["mAP@0.5:0.95"]:.4f}')

    # Generate visualizations
    print('\nGenerating visualizations...')
    plot_per_class_ap(results, os.path.join(args.output_dir, 'per_class_ap.png'))
    plot_detection_distribution(
        results, os.path.join(args.output_dir, 'detection_distribution.png')
    )

    # Generate report
    generate_report(results, speed_results, args.output_dir, config)

    # Save raw results
    combined_results = {
        'accuracy': results,
        'speed': speed_results,
        'config': config,
        'total_params': total_params
    }

    with open(os.path.join(args.output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)

    print(f'\nAll results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
