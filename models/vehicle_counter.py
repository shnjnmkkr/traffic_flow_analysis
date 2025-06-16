import torch
import torch.nn as nn
from .backbone import CustomBackbone
from .fpn import FPN
from .counting_head import DensityHead, CountHead

class VehicleCounter(nn.Module):
    def __init__(self, backbone_channels=[128, 256, 512, 1024], fpn_channels=256, dropout_rate=0.3, backbone_type="custom"):
        super().__init__()
        
        # Initialize backbone and FPN
        self.backbone = CustomBackbone(in_channels=3, backbone_type=backbone_type)
        self.fpn = FPN(backbone_channels, fpn_channels, dropout_rate=dropout_rate)
        
        # Initialize heads
        self.density_head = DensityHead(fpn_channels, fpn_channels, dropout_rate=dropout_rate)
        self.count_head = CountHead(fpn_channels, fpn_channels, dropout_rate=dropout_rate)
        self.global_head = CountHead(fpn_channels, fpn_channels, dropout_rate=dropout_rate)
        
    def forward(self, x):
        # Get backbone features
        features = self.backbone(x)
        
        # Get FPN features
        fpn_features = self.fpn(features)
        
        # Use highest resolution features for predictions
        features = fpn_features[0]  # Use P1 (1/4 resolution)
        
        # Generate predictions
        density_map = self.density_head(features)
        count = self.count_head(features)
        global_count = self.global_head(features)
        
        return {
            'density_map': density_map,
            'count': count,
            'global_count': global_count
        } 