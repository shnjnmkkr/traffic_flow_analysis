import torch
import numpy as np
from typing import List, Dict
from sklearn.metrics import r2_score
import scipy.stats

class DetectionMetrics:
    """Class for computing detection metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_detections = 0
        self.correct_detections = 0
        self.total_ground_truth = 0
    
    def update(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """Update metrics with new predictions and ground truth"""
        self.total_ground_truth += len(gt_boxes)
        self.total_detections += len(pred_boxes)
        
        # Count correct detections
        for pb in pred_boxes:
            for gb in gt_boxes:
                iou = compute_iou(pb, gb)
                if iou >= iou_threshold:
                    self.correct_detections += 1
                    break
    
    def compute(self):
        """Compute final metrics"""
        precision = self.correct_detections / max(1, self.total_detections)
        recall = self.correct_detections / max(1, self.total_ground_truth)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_detections': self.total_detections,
            'correct_detections': self.correct_detections,
            'total_ground_truth': self.total_ground_truth
        }

def mean_average_precision(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Compute mean Average Precision (mAP) for detection.
    Args:
        pred_boxes: list of [x1, y1, x2, y2] predicted boxes
        gt_boxes: list of [x1, y1, x2, y2] ground truth boxes
        iou_threshold: IoU threshold for a correct detection
    Returns:
        mAP (float)
    """
    # Dummy implementation for illustration
    # Replace with real mAP calculation for your use case
    if not pred_boxes or not gt_boxes:
        return 0.0
    correct = 0
    for pb in pred_boxes:
        for gb in gt_boxes:
            iou = compute_iou(pb, gb)
            if iou >= iou_threshold:
                correct += 1
                break
    return correct / max(1, len(gt_boxes))

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
    return iou

def mae_rmse(pred_counts, gt_counts):
    """
    Compute Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for counting.
    Args:
        pred_counts: list of predicted counts
        gt_counts: list of ground truth counts
    Returns:
        mae (float), rmse (float)
    """
    pred_counts = np.array(pred_counts)
    gt_counts = np.array(gt_counts)
    mae = np.mean(np.abs(pred_counts - gt_counts))
    rmse = np.sqrt(np.mean((pred_counts - gt_counts) ** 2))
    return mae, rmse

def classification_metrics(pred_labels, gt_labels, num_classes):
    """
    Compute accuracy and confusion matrix for classification.
    Args:
        pred_labels: list of predicted class indices
        gt_labels: list of ground truth class indices
        num_classes: number of classes
    Returns:
        accuracy (float), conf_matrix (np.ndarray)
    """
    pred_labels = np.array(pred_labels)
    gt_labels = np.array(gt_labels)
    accuracy = np.mean(pred_labels == gt_labels)
    conf_matrix = confusion_matrix(gt_labels, pred_labels, labels=list(range(num_classes)))
    return accuracy, conf_matrix 

def calculate_detection_metrics(predictions: List[Dict], targets: List[Dict], iou_threshold=0.5) -> Dict:
    """
    Calculate comprehensive detection metrics similar to main pipeline.
    
    Args:
        predictions: List of dictionaries containing 'boxes', 'scores', 'classes'
        targets: List of dictionaries containing ground truth 'boxes', 'classes'
        iou_threshold: IoU threshold for positive detection
    
    Returns:
        Dictionary containing various detection metrics
    """
    if not predictions or not targets:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mAP': 0.0,
            'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'pearson': 0.0
        }
    
    # Extract counts for regression metrics
    pred_counts = torch.tensor([len(p['boxes']) for p in predictions], dtype=torch.float32)
    target_counts = torch.tensor([len(t['boxes']) for t in targets], dtype=torch.float32)
    
    # Calculate regression metrics (similar to main pipeline)
    count_abs_error = torch.abs(pred_counts - target_counts)
    count_rel_error = count_abs_error / (target_counts + 1e-6)
    
    # Calculate MAP using multiple IoU thresholds
    iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    ap_scores = []
    
    for iou_thresh in iou_thresholds:
        ap = calculate_ap(predictions, targets, iou_thresh)
        ap_scores.append(ap)
    
    mAP = np.mean(ap_scores)
    
    # Calculate precision, recall, F1 at default IoU threshold
    precision, recall, f1 = calculate_pr_f1(predictions, targets, iou_threshold)
    
    # Calculate R² and Pearson correlation for count regression
    pred_counts_np = pred_counts.numpy()
    target_counts_np = target_counts.numpy()
    
    if len(pred_counts_np) > 1:
        r2 = r2_score(target_counts_np, pred_counts_np)
        pearson = scipy.stats.pearsonr(target_counts_np, pred_counts_np)[0]
    else:
        r2 = 0.0
        pearson = 0.0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mAP': mAP,
        'mae': count_abs_error.mean().item(),
        'rmse': torch.sqrt(torch.mean((pred_counts - target_counts) ** 2)).item(),
        'mean_rel_error': count_rel_error.mean().item(),
        'r2': r2,
        'pearson': pearson,
        'accuracy_10%': (count_rel_error < 0.1).float().mean().item(),
        'accuracy_20%': (count_rel_error < 0.2).float().mean().item(),
        'accuracy_30%': (count_rel_error < 0.3).float().mean().item(),
        'total_gt': target_counts.sum().item(),
        'total_pred': pred_counts.sum().item()
    }
    
    return metrics

def calculate_ap(predictions: List[Dict], targets: List[Dict], iou_threshold=0.5) -> float:
    """Calculate Average Precision at given IoU threshold"""
    all_predictions = []
    all_targets = []
    
    for pred, target in zip(predictions, targets):
        if len(pred['boxes']) > 0:
            all_predictions.extend([
                {'box': box, 'score': score, 'class': cls}
                for box, score, cls in zip(pred['boxes'], pred['scores'], pred['classes'])
            ])
        if len(target['boxes']) > 0:
            all_targets.extend([
                {'box': box, 'class': cls}
                for box, cls in zip(target['boxes'], target['classes'])
            ])
    
    if not all_predictions or not all_targets:
        return 0.0
    
    # Sort predictions by score
    all_predictions.sort(key=lambda x: x['score'], reverse=True)
    
    # Calculate precision and recall
    tp = 0
    fp = 0
    fn = len(all_targets)
    
    precisions = []
    recalls = []
    
    for pred in all_predictions:
        # Check if this prediction matches any ground truth
        matched = False
        for i, target in enumerate(all_targets):
            if target.get('matched', False):
                continue
            
            iou = calculate_iou(pred['box'], target['box'])
            if iou >= iou_threshold and pred['class'] == target['class']:
                tp += 1
                fn -= 1
                target['matched'] = True
                matched = True
                break
        
        if not matched:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP using interpolation
    ap = 0.0
    for i in range(len(precisions)):
        if i == 0:
            ap += precisions[i] * recalls[i]
        else:
            ap += precisions[i] * (recalls[i] - recalls[i-1])
    
    return ap

def calculate_pr_f1(predictions: List[Dict], targets: List[Dict], iou_threshold=0.5) -> tuple:
    """Calculate Precision, Recall, and F1 score"""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred, target in zip(predictions, targets):
        if len(pred['boxes']) == 0 and len(target['boxes']) == 0:
            continue
        
        if len(pred['boxes']) == 0:
            total_fn += len(target['boxes'])
            continue
        
        if len(target['boxes']) == 0:
            total_fp += len(pred['boxes'])
            continue
        
        # Calculate IoU matrix
        ious = torch.zeros(len(pred['boxes']), len(target['boxes']))
        for i, pred_box in enumerate(pred['boxes']):
            for j, target_box in enumerate(target['boxes']):
                ious[i, j] = calculate_iou(pred_box, target_box)
        
        # Match predictions to ground truth
        matched_targets = set()
        for i in range(len(pred['boxes'])):
            best_iou, best_j = ious[i].max(dim=0)
            if best_iou >= iou_threshold and best_j not in matched_targets:
                total_tp += 1
                matched_targets.add(best_j)
            else:
                total_fp += 1
        
        total_fn += len(target['boxes']) - len(matched_targets)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().numpy()
    
    # Convert to x1, y1, x2, y2 format if needed
    if len(box1) == 4:
        x1_1, y1_1, x2_1, y2_1 = box1
    else:
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    
    if len(box2) == 4:
        x1_2, y1_2, x2_2, y2_2 = box2
    else:
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict:
    """
    Main metrics calculation function - wrapper for detection metrics
    """
    return calculate_detection_metrics(predictions, targets) 