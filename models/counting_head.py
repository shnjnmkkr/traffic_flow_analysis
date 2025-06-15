import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class DensityHead(nn.Module):
    def __init__(self, in_channels, fpn_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, fpn_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(fpn_channels)
        self.conv2 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(fpn_channels)
        self.conv3 = nn.Conv2d(fpn_channels, 1, 1)
        self.dropout = nn.Dropout(0.2)
        
        # Attention mechanisms
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(fpn_channels)
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Apply attention before final conv
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        
        # Final conv to get density map
        x = self.conv3(x)
        return x

class CountHead(nn.Module):
    def __init__(self, in_channels, fpn_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, fpn_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(fpn_channels)
        self.conv2 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(fpn_channels)
        self.dropout = nn.Dropout(0.2)
        
        # Attention mechanisms
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(fpn_channels)
        
        # Global pooling and FC layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(fpn_channels, fpn_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fpn_channels // 2, 1)
        )
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Apply attention before pooling
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        
        # Global pooling and FC
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(-1)

class RegressionHead(nn.Module):
    def __init__(self, in_channels, fpn_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, fpn_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(fpn_channels)
        self.conv2 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(fpn_channels)
        self.dropout = nn.Dropout(0.2)
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(fpn_channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(fpn_channels, fpn_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fpn_channels // 2, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.spatial_attention(x)
        x = self.channel_attention(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.squeeze(-1)
        return x 