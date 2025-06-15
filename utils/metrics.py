import torch
import numpy as np
from typing import List, Dict

def calculate_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict:
    """
    Calculate various metrics for vehicle counting.
    
    Args:
        predictions: List of dictionaries containing 'count' and 'global_count'
        targets: List of dictionaries containing ground truth 'count' and 'global_count'
    
    Returns:
        Dictionary containing various metrics
    """
    # Convert predictions and targets to tensors
    pred_counts = torch.stack([p['count'] for p in predictions])
    target_counts = torch.stack([t['count'] for t in targets])
    
    pred_global = torch.stack([p['global_count'] for p in predictions])
    target_global = torch.stack([t['global_count'] for t in targets])
    
    # Calculate absolute errors
    count_abs_error = torch.abs(pred_counts - target_counts)
    global_abs_error = torch.abs(pred_global - target_global)
    
    # Calculate relative errors
    count_rel_error = count_abs_error / (target_counts + 1e-6)  # Add small epsilon to avoid division by zero
    global_rel_error = global_abs_error / (target_global + 1e-6)
    
    # Calculate MAP (Mean Average Precision)
    # For counting, we'll use a threshold-based approach
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    map_scores = []
    
    for threshold in thresholds:
        # Consider prediction correct if relative error is less than threshold
        correct = (count_rel_error < threshold).float()
        precision = correct.mean()
        map_scores.append(precision)
    
    map_score = sum(map_scores) / len(thresholds)
    
    metrics = {
        'mae': count_abs_error.mean().item(),
        'rmse': torch.sqrt(torch.mean((pred_counts - target_counts) ** 2)).item(),
        'mean_rel_error': count_rel_error.mean().item(),
        'global_mae': global_abs_error.mean().item(),
        'global_rmse': torch.sqrt(torch.mean((pred_global - target_global) ** 2)).item(),
        'global_mean_rel_error': global_rel_error.mean().item(),
        'map': map_score.item(),
        'accuracy_10%': (count_rel_error < 0.1).float().mean().item(),
        'accuracy_20%': (count_rel_error < 0.2).float().mean().item(),
        'accuracy_30%': (count_rel_error < 0.3).float().mean().item()
    }
    
    return metrics 