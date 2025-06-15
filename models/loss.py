import torch
import torch.nn as nn
import torch.nn.functional as F

class VehicleCountingLoss(nn.Module):
    def __init__(self, density_weight=1.0, count_weight=0.1, global_weight=0.1):
        super().__init__()
        self.density_weight = density_weight
        self.count_weight = count_weight
        self.global_weight = global_weight
        
        # Initialize loss components
        self.density_criterion = nn.MSELoss()
        self.count_criterion = nn.SmoothL1Loss()
        self.global_criterion = nn.SmoothL1Loss()
        
    def forward(self, predictions, targets):
        # Extract predictions
        density_map = predictions['density_map']
        count = predictions['count']
        global_count = predictions['global_count']
        
        # Extract targets
        target_density = targets['density_map']
        target_count = targets['count']
        target_global = targets['global_count']
        
        # Calculate density loss
        density_loss = self.density_criterion(density_map, target_density)
        
        # Calculate count loss with normalization
        count_scale = target_count.mean().clamp(min=1.0)
        count_loss = self.count_criterion(count / count_scale, target_count / count_scale)
        
        # Calculate global count loss with normalization
        global_scale = target_global.mean().clamp(min=1.0)
        global_loss = self.global_criterion(global_count / global_scale, target_global / global_scale)
        
        # Combine losses
        total_loss = (
            self.density_weight * density_loss +
            self.count_weight * count_loss +
            self.global_weight * global_loss
        )
        
        # Check for NaN or Inf values
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: NaN or Inf detected in loss calculation")
            return torch.tensor(0.0, device=total_loss.device, requires_grad=True)
        
        return total_loss 