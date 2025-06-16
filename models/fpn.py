import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, dropout_rate=0.2):
        super(FPN, self).__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            for in_channels in in_channels_list
        ])
        
        # FPN output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout2d(dropout_rate)
            )
            for _ in range(len(in_channels_list))
        ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: List of features from backbone [c1, c2, c3, c4]
                     c1: 1/4 resolution
                     c2: 1/8 resolution
                     c3: 1/16 resolution
                     c4: 1/32 resolution
        Returns:
            List of FPN features [p1, p2, p3, p4]
        """
        # Process features from top to bottom
        laterals = []
        for i, feature in enumerate(features):
            laterals.append(self.lateral_convs[i](feature))
        
        # Top-down pathway
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i],
                size=laterals[i-1].shape[-2:],
                mode='nearest'
            )
        
        # FPN output
        outputs = []
        for i, lateral in enumerate(laterals):
            outputs.append(self.fpn_convs[i](lateral))
        
        return outputs 