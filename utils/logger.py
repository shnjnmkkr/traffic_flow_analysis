import csv
import os
from datetime import datetime
from typing import Dict

class MetricsLogger:
    def __init__(self, log_dir: str):
        """
        Initialize the metrics logger.
        
        Args:
            log_dir: Directory to save the metrics CSV file
        """
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, 'metrics_history.csv')
        self.fieldnames = [
            'epoch',
            'timestamp',
            'train_loss',
            'val_loss',
            'mae',
            'rmse',
            'map',
            'accuracy_10%',
            'accuracy_20%',
            'accuracy_30%',
            'mean_rel_error',
            'global_mae',
            'global_rmse',
            'global_mean_rel_error'
        ]
        
        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def log_metrics(self, epoch: int, train_loss: float, val_metrics: Dict):
        """
        Log metrics for the current epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss for the epoch
            val_metrics: Dictionary containing validation metrics
        """
        metrics = {'epoch': epoch, 'train_loss': train_loss}
        metrics.update(val_metrics)
        # Dynamically update fieldnames if new keys are present
        new_keys = [k for k in metrics.keys() if k not in self.fieldnames]
        if new_keys:
            self.fieldnames.extend(new_keys)
            with open(self.csv_path, 'r') as f:
                reader = list(csv.DictReader(f))
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                for row in reader:
                    writer.writerow(row)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics) 