"""
Training script for RefineDet vehicle detection model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import logging
import os
from pathlib import Path
import csv
from datetime import datetime
import argparse
import sys
from tqdm import tqdm

# Import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from detection.refinedet import RefineDet
from detection.yolo_dataset import YOLODetectionDataset
from detection.loss import RefineDetLoss
from utils.metrics import calculate_detection_metrics

def setup_logging(log_dir: str):
    """Setup logging to file and console"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path: str):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path: str):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path: str):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    metrics = checkpoint.get('metrics', {})
    return start_epoch, metrics

def save_metrics(metrics_history, csv_path: str):
    """Save metrics history to CSV"""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        if metrics_history:
            # Write header
            writer.writerow(['epoch'] + list(metrics_history[0].keys()))
            # Write data
            for i, metrics in enumerate(metrics_history):
                writer.writerow([i] + list(metrics.values()))

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    # Create progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, (images, boxes, labels) in enumerate(pbar):
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, (boxes, labels))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Debug: Print first few losses to understand what's happening
        if batch_idx < 3:
            logging.info(f'Debug - Batch {batch_idx}: Loss={loss.item():.6f}, Boxes={len(boxes)}, Labels={len(labels)}')
    
    avg_loss = total_loss / num_batches
    return avg_loss

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    with torch.no_grad():
        for images, boxes, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, (boxes, labels))
            total_loss += loss.item()
    avg_loss = total_loss / num_batches
    return avg_loss

def calculate_comprehensive_metrics(model, dataloader, device, iou_threshold=0.5):
    """Calculate comprehensive detection metrics including MAE, MAP, R², etc."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, boxes, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            detections = model._decode_detections(outputs['odm_locs'], outputs['odm_confs'], images.shape[2:])
            
            for i in range(len(images)):
                # Extract predictions
                if len(detections[i]) > 0:
                    pred_boxes = detections[i][:, :4]
                    pred_scores = detections[i][:, 4]
                    pred_classes = detections[i][:, 5]
                    
                    # Filter by confidence
                    conf_mask = pred_scores > 0.5
                    if conf_mask.sum() > 0:
                        pred_boxes = pred_boxes[conf_mask]
                        pred_scores = pred_scores[conf_mask]
                        pred_classes = pred_classes[conf_mask]
                    else:
                        pred_boxes = torch.empty((0, 4))
                        pred_scores = torch.empty((0,))
                        pred_classes = torch.empty((0,))
                else:
                    pred_boxes = torch.empty((0, 4))
                    pred_scores = torch.empty((0,))
                    pred_classes = torch.empty((0,))
                
                all_predictions.append({
                    'boxes': pred_boxes.cpu(),
                    'scores': pred_scores.cpu(),
                    'classes': pred_classes.cpu()
                })
                
                all_targets.append({
                    'boxes': boxes[i].cpu(),
                    'classes': labels[i].cpu()
                })
    
    # Calculate comprehensive metrics
    metrics = calculate_detection_metrics(all_predictions, all_targets, iou_threshold)
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Train RefineDet for vehicle detection')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    parser.add_argument('--data', default='../traffic_wala_dataset', help='Path to dataset')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup directories
    log_dir = config.get('log_dir', 'logs')
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_dir)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Dataset paths
    dataset_path = Path(args.data)
    train_images = dataset_path / 'train' / 'images'
    train_labels = dataset_path / 'train' / 'labels'
    val_images = dataset_path / 'valid' / 'images'
    val_labels = dataset_path / 'valid' / 'labels'
    
    # Create datasets
    train_dataset = YOLODetectionDataset(
        images_dir=str(train_images),
        labels_dir=str(train_labels),
        transform=None  # Add transforms if needed
    )
    
    val_dataset = YOLODetectionDataset(
        images_dir=str(val_images),
        labels_dir=str(val_labels),
        transform=None
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=val_dataset.collate_fn
    )
    
    # Create model
    model = RefineDet(num_classes=config.get('num_classes', 2))  # background + vehicle
    model.to(device)
    
    # Loss function and optimizer
    criterion = RefineDetLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('lr_step_size', 30),
        gamma=config.get('lr_gamma', 0.1)
    )
    
    # Training parameters
    num_epochs = config.get('num_epochs', 100)
    start_epoch = 0
    metrics_history = []
    
    # Resume from checkpoint if specified
    if args.resume:
        start_epoch, metrics_history = load_checkpoint(model, optimizer, args.resume)
        logger.info(f"Resumed training from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Calculate comprehensive detection metrics (every 5 epochs to save time)
        if epoch % 5 == 0:
            detection_metrics = calculate_comprehensive_metrics(model, val_loader, device)
        else:
            detection_metrics = {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mAP': 0.0,
                'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'pearson': 0.0,
                'accuracy_10%': 0.0, 'accuracy_20%': 0.0, 'accuracy_30%': 0.0
            }
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'precision': detection_metrics['precision'],
            'recall': detection_metrics['recall'],
            'f1_score': detection_metrics['f1'],
            'mAP': detection_metrics['mAP'],
            'mae': detection_metrics['mae'],
            'rmse': detection_metrics['rmse'],
            'r2': detection_metrics['r2'],
            'pearson': detection_metrics['pearson'],
            'accuracy_10%': detection_metrics['accuracy_10%'],
            'accuracy_20%': detection_metrics['accuracy_20%'],
            'accuracy_30%': detection_metrics['accuracy_30%']
        }
        metrics_history.append(metrics)
        
        # Enhanced logging with comprehensive metrics (similar to main pipeline)
        if epoch % 5 == 0:
            logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            logger.info(f'  Detection - Precision: {detection_metrics["precision"]:.3f}, Recall: {detection_metrics["recall"]:.3f}, F1: {detection_metrics["f1"]:.3f}, mAP: {detection_metrics["mAP"]:.3f}')
            logger.info(f'  Counting - MAE: {detection_metrics["mae"]:.3f}, RMSE: {detection_metrics["rmse"]:.3f}, R²: {detection_metrics["r2"]:.3f}, Pearson: {detection_metrics["pearson"]:.3f}')
            logger.info(f'  Accuracy - 10%: {detection_metrics["accuracy_10%"]:.3f}, 20%: {detection_metrics["accuracy_20%"]:.3f}, 30%: {detection_metrics["accuracy_30%"]:.3f}')
        else:
            logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % config.get('save_interval', 10) == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save metrics
        metrics_csv = os.path.join(log_dir, 'metrics_history.csv')
        save_metrics(metrics_history, metrics_csv)
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, num_epochs-1, metrics, final_checkpoint_path)
    logger.info(f"Training complete. Final model saved: {final_checkpoint_path}")

if __name__ == "__main__":
    main() 