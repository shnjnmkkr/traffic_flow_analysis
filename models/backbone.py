import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1)
        
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=1)
        )
        
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=5, padding=2)
        )
        
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=2, dilation=2)
        )
        
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_dilated = self.branch_dilated(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_dilated]
        return F.relu(self.bn(torch.cat(outputs, 1)))

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv_out(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CustomBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super(CustomBackbone, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 1 (1/4)
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64),
            InceptionBlock(64, 128),
            ResidualBlock(128, 128)
        )
        
        # Stage 2 (1/8)
        self.stage2 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            InceptionBlock(256, 256),
            ResidualBlock(256, 256)
        )
        
        # Stage 3 (1/16)
        self.stage3 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            InceptionBlock(512, 512),
            ResidualBlock(512, 512)
        )
        
        # Stage 4 (1/32)
        self.stage4 = nn.Sequential(
            ResidualBlock(512, 1024, stride=2),
            ASPP(1024, 1024)
        )

    def forward(self, x):
        # Initial features
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # Stage features
        c1 = self.stage1(x)      # 1/4
        c2 = self.stage2(c1)     # 1/8
        c3 = self.stage3(c2)     # 1/16
        c4 = self.stage4(c3)     # 1/32
        
        return [c1, c2, c3, c4] 