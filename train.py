import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import yaml
import torch.cuda.amp as amp
from pathlib import Path
import numpy as np
import logging
from datetime import datetime

from models.vehicle_counter import VehicleCounter
from models.loss import VehicleCountingLoss
from data.dataset import VehicleCountingDataset
from utils.metrics import calculate_metrics
from utils.logger import MetricsLogger

def get_transforms(is_train=True):
    return A.Compose([
        A.Resize(640, 640),
        A.HorizontalFlip(p=0.5) if is_train else A.NoOp(),
        A.RandomBrightnessContrast(p=0.2) if is_train else A.NoOp(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={'density_map': 'mask'})

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, writer, config):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        images = batch['image'].to(device)
        targets = {
            'density_map': batch['density_map'].to(device),
            'count': batch['count'].to(device),
            'global_count': batch['global_count'].to(device)
        }
        
        # Forward pass
        predictions = model(images)
        # Upsample predicted density map to match target
        pred_density = predictions['density_map']
        target_density = targets['density_map']
        if pred_density.shape[-2:] != target_density.shape[-2:]:
            predictions['density_map'] = torch.nn.functional.interpolate(
                pred_density, size=target_density.shape[-2:], mode='bilinear', align_corners=False
            )
        loss = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        if config['training']['grad_clip']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
        optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        # Log batch metrics
        if writer is not None:
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(dataloader) + pbar.n)
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            images = batch['image'].to(device)
            targets = {
                'density_map': batch['density_map'].to(device),
                'count': batch['count'].to(device),
                'global_count': batch['global_count'].to(device)
            }
            
            predictions = model(images)
            # Upsample predicted density map to match target
            pred_density = predictions['density_map']
            target_density = targets['density_map']
            if pred_density.shape[-2:] != target_density.shape[-2:]:
                predictions['density_map'] = torch.nn.functional.interpolate(
                    pred_density, size=target_density.shape[-2:], mode='bilinear', align_corners=False
                )
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            
            # Store predictions and targets for metrics (per sample)
            batch_size = targets['count'].shape[0]
            for i in range(batch_size):
                all_predictions.append({
                    'count': predictions['count'][i].cpu().squeeze(),
                    'global_count': predictions['global_count'][i].cpu().squeeze()
                })
                all_targets.append({
                    'count': targets['count'][i].cpu().squeeze(),
                    'global_count': targets['global_count'][i].cpu().squeeze()
                })
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)
    metrics['val_loss'] = total_loss / len(dataloader)
    
    return metrics

def collate_fn(batch):
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density_map': torch.stack([item['density_map'] for item in batch]),
        'count': torch.stack([item['count'] for item in batch]),
        'global_count': torch.stack([item['global_count'] for item in batch]),
        'points': [item['points'] for item in batch],
        'image_name': [item['image_name'] for item in batch]
    }

def build_transform(transform_config):
    """Build an Albumentations Compose transform from a config list."""
    transforms = []
    for t in transform_config:
        name = t['name']
        if name == 'Resize':
            transforms.append(A.Resize(height=t['height'], width=t['width']))
        elif name == 'Normalize':
            transforms.append(A.Normalize(mean=t['mean'], std=t['std']))
        elif name == 'ToTensor':
            transforms.append(ToTensorV2())
        # Add more transforms as needed
    return A.Compose(transforms)

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Create datasets and dataloaders
    train_transform = build_transform(config['data']['train_transform'])
    val_transform = build_transform(config['data']['val_transform'])
    train_dataset = VehicleCountingDataset(
        root_dir="traffic_wala_dataset",
        transform=train_transform,
        split='train'
    )
    val_dataset = VehicleCountingDataset(
        root_dir="traffic_wala_dataset",
        transform=val_transform,
        split='valid'
    )
    
    # Calculate optimal number of workers
    num_workers = min(os.cpu_count(), 4) if os.cpu_count() else 0
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Create model
    model = VehicleCounter(
        backbone_channels=config['model']['backbone_channels'],
        fpn_channels=config['model']['fpn_channels']
    ).to(device)
    
    # Create loss function
    criterion = VehicleCountingLoss(
        density_weight=config['loss']['density_weight'],
        count_weight=config['loss']['count_weight'],
        global_weight=config['loss']['global_weight']
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['scheduler']['max_lr'],
        epochs=config['training']['scheduler']['epochs'],
        steps_per_epoch=config['training']['scheduler']['steps_per_epoch'],
        pct_start=config['training']['scheduler']['pct_start'],
        div_factor=config['training']['scheduler']['div_factor'],
        final_div_factor=config['training']['scheduler']['final_div_factor']
    )
    
    # Create tensorboard writer and metrics logger
    writer = SummaryWriter(log_dir)
    metrics_logger = MetricsLogger(log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['training']['epochs']):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            writer=writer,
            config=config
        )
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Save the model after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_metrics['val_loss']
        }, os.path.join(log_dir, f'model_epoch_{epoch}.pth'))
        
        # Save the best model based on validation loss
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_metrics['val_loss']
            }, os.path.join(log_dir, 'best_model.pth'))
        
        # Log metrics
        metrics_logger.log_metrics(epoch, train_loss, val_metrics)
        print(f'Epoch {epoch} Metrics:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_metrics["val_loss"]:.4f}')
        print(f'MAE: {val_metrics["mae"]:.4f}')
        print(f'RMSE: {val_metrics["rmse"]:.4f}')
        print(f'MAP: {val_metrics["map"]:.4f}')
        print(f'Accuracy (10%): {val_metrics["accuracy_10%"]:.4f}')
        print(f'Accuracy (20%): {val_metrics["accuracy_20%"]:.4f}')
        print(f'Accuracy (30%): {val_metrics["accuracy_30%"]:.4f}')
        print(f'Mean Relative Error: {val_metrics["mean_rel_error"]:.4f}')
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
        writer.add_scalar('Metrics/MAE', val_metrics['mae'], epoch)
        writer.add_scalar('Metrics/RMSE', val_metrics['rmse'], epoch)
        writer.add_scalar('Metrics/MAP', val_metrics['map'], epoch)
        writer.add_scalar('Metrics/Accuracy_10', val_metrics['accuracy_10%'], epoch)
        writer.add_scalar('Metrics/Accuracy_20', val_metrics['accuracy_20%'], epoch)
        writer.add_scalar('Metrics/Accuracy_30', val_metrics['accuracy_30%'], epoch)
        writer.add_scalar('Metrics/Mean_Relative_Error', val_metrics['mean_rel_error'], epoch)
    
    writer.close()

if __name__ == '__main__':
    main() 