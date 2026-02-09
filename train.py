"""
Training script for FCOS banknote detector.
Features: AMP, gradient clipping, checkpointing, early stopping, TensorBoard logging.
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import BanknoteDataset, get_dataloaders
from model import FCOS
from losses import FCOSLoss
from utils.metrics import compute_map_coco


def parse_args():
    parser = argparse.ArgumentParser(description='Train FCOS banknote detector')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs in config')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    loss_fn: FCOSLoss,
    device: torch.device,
    epoch: int,
    scaler: GradScaler,
    grad_clip: float,
    writer: SummaryWriter,
    image_size: tuple,
    gradient_accumulation_steps: int = 1
) -> dict:
    """Train for one epoch with optional gradient accumulation."""
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_ctr_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)

        # Forward pass with AMP
        with autocast('cuda'):
            cls_outputs, reg_outputs, ctr_outputs = model(images)

            loss, cls_loss, reg_loss, ctr_loss = loss_fn(
                cls_outputs, reg_outputs, ctr_outputs, targets,
                image_size
            )

            # Scale loss by accumulation steps for correct averaging
            loss = loss / gradient_accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Only update weights every N accumulation steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Accumulate losses
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()
        total_ctr_loss += ctr_loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'reg': f'{reg_loss.item():.4f}',
            'ctr': f'{ctr_loss.item():.4f}'
        })

        # Log to TensorBoard
        global_step = epoch * len(train_loader) + batch_idx
        if batch_idx % 10 == 0:
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/cls_loss', cls_loss.item(), global_step)
            writer.add_scalar('train/reg_loss', reg_loss.item(), global_step)
            writer.add_scalar('train/ctr_loss', ctr_loss.item(), global_step)

    # Compute averages
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_reg_loss = total_reg_loss / num_batches
    avg_ctr_loss = total_ctr_loss / num_batches

    return {
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'reg_loss': avg_reg_loss,
        'ctr_loss': avg_ctr_loss
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader,
    device: torch.device,
    num_classes: int
) -> dict:
    """Evaluate model on test set."""
    model.eval()

    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []

    for images, targets in tqdm(test_loader, desc='Evaluating'):
        images = images.to(device)

        # Run inference
        results = model.inference(images)

        for i, (result, target) in enumerate(zip(results, targets)):
            all_pred_boxes.append(result['boxes'].cpu())
            all_pred_scores.append(result['scores'].cpu())
            all_pred_labels.append(result['labels'].cpu())
            all_gt_boxes.append(target['boxes'])
            all_gt_labels.append(target['labels'])

    # Compute mAP
    mAP_coco, mAP_50, metrics = compute_map_coco(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_gt_boxes, all_gt_labels, num_classes
    )

    return {
        'mAP@0.5:0.95': mAP_coco,
        'mAP@0.5': mAP_50,
        **metrics
    }


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_map: float,
    config: dict,
    output_dir: str,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_map': best_map,
        'config': config
    }

    # Save latest checkpoint
    latest_path = os.path.join(output_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)
        print(f'  New best model saved with mAP@0.5: {best_map:.4f}')

    # Save periodic checkpoint
    if (epoch + 1) % 10 == 0:
        periodic_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, periodic_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler: GradScaler
) -> tuple:
    """Load checkpoint and return epoch and best_map."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint['epoch'], checkpoint['best_map']


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.epochs is not None:
        config['epochs'] = args.epochs

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Setup TensorBoard
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))

    # Create dataloaders
    print('Loading datasets...')
    merge_banknotes = config.get('merge_banknotes', True)
    train_loader, test_loader = get_dataloaders(
        config_path=args.config,
        num_workers=args.num_workers,
        merge_banknotes=merge_banknotes
    )
    train_dataset = train_loader.dataset

    print(f'Training samples: {len(train_dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    print(f'Number of classes: {train_dataset.num_classes}')

    # Update config with actual number of classes
    config['num_classes'] = train_dataset.num_classes

    # Compute class weights for handling imbalance
    print('Computing class weights...')
    class_weights = train_dataset.compute_class_weights().to(device)
    print(f'Class weights (min={class_weights.min():.2f}, max={class_weights.max():.2f}, mean={class_weights.mean():.2f})')

    # Create model
    print('Creating model...')
    model = FCOS(
        num_classes=config['num_classes'],
        fpn_channels=config['fpn_channels'],
        strides=config['strides'],
        score_threshold=config.get('score_threshold', 0.05),
        nms_threshold=config.get('nms_threshold', 0.5),
        max_detections=config.get('max_detections', 100),
        backbone_type=config.get('backbone_type', 'resnet18'),
        pretrained=config.get('use_pretrained', False)
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Backbone: {config.get("backbone_type", "resnet18")} (pretrained: {config.get("use_pretrained", False)})')

    # Optionally freeze backbone for initial epochs
    freeze_backbone_epochs = config.get('freeze_backbone_epochs', 0)
    if freeze_backbone_epochs > 0:
        print(f'Freezing backbone for first {freeze_backbone_epochs} epochs')
        for param in model.backbone.parameters():
            param.requires_grad = False

    # Enable gradient checkpointing to save memory
    use_checkpointing = config.get('use_gradient_checkpointing', False)
    if use_checkpointing:
        print('Enabling gradient checkpointing (trades compute for memory)')
        from apply_gradient_checkpointing import apply_gradient_checkpointing
        model = apply_gradient_checkpointing(model)

    # Create loss function with class weights
    loss_fn = FCOSLoss(
        num_classes=config['num_classes'],
        focal_alpha=config.get('focal_alpha', 0.25),
        focal_gamma=config.get('focal_gamma', 2.0),
        lambda_reg=config.get('lambda_reg', 1.0),
        lambda_ctr=config.get('lambda_ctr', 1.0),
        strides=config['strides'],
        class_weights=class_weights
    )

    # Create optimizer with differential learning rates
    backbone_lr = config.get('backbone_lr', config['lr'])
    main_lr = config['lr']

    param_groups = [
        {'params': model.backbone.parameters(), 'lr': backbone_lr, 'name': 'backbone'},
        {'params': model.fpn.parameters(), 'lr': main_lr, 'name': 'fpn'},
        {'params': model.head.parameters(), 'lr': main_lr, 'name': 'head'}
    ]

    optimizer = AdamW(
        param_groups,
        weight_decay=config['weight_decay']
    )
    print(f'Learning rates: backbone={backbone_lr:.2e}, head/fpn={main_lr:.2e}')

    # Create learning rate scheduler (warmup + cosine annealing)
    warmup_epochs = config.get('warmup_epochs', 3)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'] - warmup_epochs,
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # Create gradient scaler for AMP
    scaler = GradScaler('cuda')

    # Resume from checkpoint if specified
    start_epoch = 0
    best_map = 0.0

    if args.resume:
        print(f'Resuming from {args.resume}')
        start_epoch, best_map = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        start_epoch += 1
        print(f'Resuming from epoch {start_epoch}, best mAP: {best_map:.4f}')

    # Training loop
    patience = config.get('patience', 10)
    no_improve_count = 0

    print(f'\nStarting training for {config["epochs"]} epochs...')
    print(f'Output directory: {output_dir}')
    print('-' * 60)

    for epoch in range(start_epoch, config['epochs']):
        epoch_start = time.time()

        # Unfreeze backbone after specified epochs
        freeze_backbone_epochs = config.get('freeze_backbone_epochs', 0)
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs:
            print(f'\nUnfreezing backbone at epoch {epoch}')
            for param in model.backbone.parameters():
                param.requires_grad = True

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            device, epoch, scaler, config.get('grad_clip', 1.0), writer,
            tuple(config['image_size']),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1)
        )

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Free GPU memory before evaluation
        torch.cuda.empty_cache()

        # Evaluate
        test_metrics = evaluate(model, test_loader, device, config['num_classes'])

        epoch_time = time.time() - epoch_start

        # Log metrics
        print(f'\nEpoch {epoch+1}/{config["epochs"]} ({epoch_time:.1f}s)')
        print(f'  Train - Loss: {train_metrics["loss"]:.4f}, '
              f'Cls: {train_metrics["cls_loss"]:.4f}, '
              f'Reg: {train_metrics["reg_loss"]:.4f}, '
              f'Ctr: {train_metrics["ctr_loss"]:.4f}')
        print(f'  Test - mAP@0.5: {test_metrics["mAP@0.5"]:.4f}, '
              f'mAP@0.5:0.95: {test_metrics["mAP@0.5:0.95"]:.4f}')
        print(f'  LR: {current_lr:.6f}')

        # Log to TensorBoard
        writer.add_scalar('train/epoch_loss', train_metrics['loss'], epoch)
        writer.add_scalar('test/mAP_0.5', test_metrics['mAP@0.5'], epoch)
        writer.add_scalar('test/mAP_0.5_0.95', test_metrics['mAP@0.5:0.95'], epoch)
        writer.add_scalar('lr', current_lr, epoch)

        # Check for improvement
        is_best = test_metrics['mAP@0.5'] > best_map
        if is_best:
            best_map = test_metrics['mAP@0.5']
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch, best_map, config, output_dir, is_best
        )

        # Free GPU memory after checkpoint save
        torch.cuda.empty_cache()

        # Early stopping
        if no_improve_count >= patience:
            print(f'\nEarly stopping after {patience} epochs without improvement')
            break

    print('-' * 60)
    print(f'Training complete!')
    print(f'Best mAP@0.5: {best_map:.4f}')
    print(f'Checkpoints saved to: {output_dir}')

    writer.close()


if __name__ == '__main__':
    main()
