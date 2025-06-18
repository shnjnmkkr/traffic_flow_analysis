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
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(ASPP, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv_out(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels + i * out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            x = layer(torch.cat(features, 1))
            features.append(x)
        return torch.cat(features, 1)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        q = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, H * W)
        v = self.value(x).view(batch_size, -1, H * W)
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=2)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        out = self.gamma * out + x
        return out

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CustomBackbone(nn.Module):
    def __init__(self, in_channels=3, backbone_type="custom"):
        super(CustomBackbone, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 1 (1/4)
        if backbone_type == "custom":
            self.stage1 = nn.Sequential(
                ResidualBlock(64, 64),
                InceptionBlock(64, 128),
                ResidualBlock(128, 128)
            )
        elif backbone_type == "dense":
            self.stage1 = DenseBlock(64, 128)
        elif backbone_type == "attention":
            self.stage1 = nn.Sequential(
                ResidualBlock(64, 64),
                SelfAttention(64),
                InceptionBlock(64, 128),
                ResidualBlock(128, 128)
            )
        elif backbone_type == "se":
            self.stage1 = nn.Sequential(
                ResidualBlock(64, 64),
                SEBlock(64),
                InceptionBlock(64, 128),
                ResidualBlock(128, 128)
            )
        elif backbone_type == "hybrid":
            dense1_in, dense1_out, dense1_layers = 64, 128, 4
            dense1_final = dense1_in + dense1_layers * dense1_out
            self.stage1 = nn.Sequential(
                DenseBlock(dense1_in, dense1_out, num_layers=dense1_layers),
                SEBlock(dense1_final)
            )
        
        # Stage 2 (1/8)
        if backbone_type == "custom":
            self.stage2 = nn.Sequential(
                ResidualBlock(128, 256, stride=2),
                InceptionBlock(256, 256),
                ResidualBlock(256, 256)
            )
        elif backbone_type == "dense":
            self.stage2 = DenseBlock(128, 256)
        elif backbone_type == "attention":
            self.stage2 = nn.Sequential(
                ResidualBlock(128, 256, stride=2),
                SelfAttention(256),
                InceptionBlock(256, 256),
                ResidualBlock(256, 256)
            )
        elif backbone_type == "se":
            self.stage2 = nn.Sequential(
                ResidualBlock(128, 256, stride=2),
                SEBlock(256),
                InceptionBlock(256, 256),
                ResidualBlock(256, 256)
            )
        elif backbone_type == "hybrid":
            dense2_in, dense2_out, dense2_layers = dense1_final, 256, 4
            dense2_final = dense2_in + dense2_layers * dense2_out
            self.stage2 = nn.Sequential(
                DenseBlock(dense2_in, dense2_out, num_layers=dense2_layers),
                SEBlock(dense2_final)
            )
        
        # Stage 3 (1/16)
        if backbone_type == "custom":
            self.stage3 = nn.Sequential(
                ResidualBlock(256, 512, stride=2),
                InceptionBlock(512, 512),
                ResidualBlock(512, 512)
            )
        elif backbone_type == "dense":
            self.stage3 = DenseBlock(256, 512)
        elif backbone_type == "attention":
            self.stage3 = nn.Sequential(
                ResidualBlock(256, 512, stride=2),
                SelfAttention(512),
                InceptionBlock(512, 512),
                ResidualBlock(512, 512)
            )
        elif backbone_type == "se":
            self.stage3 = nn.Sequential(
                ResidualBlock(256, 512, stride=2),
                SEBlock(512),
                InceptionBlock(512, 512),
                ResidualBlock(512, 512)
            )
        elif backbone_type == "hybrid":
            dense3_in, dense3_out, dense3_layers = dense2_final, 512, 4
            dense3_final = dense3_in + dense3_layers * dense3_out
            self.stage3 = nn.Sequential(
                DenseBlock(dense3_in, dense3_out, num_layers=dense3_layers),
                SEBlock(dense3_final)
            )
        
        # Stage 4 (1/32)
        if backbone_type == "custom":
            self.stage4 = nn.Sequential(
                ResidualBlock(512, 1024, stride=2),
                ASPP(1024, 1024)
            )
        elif backbone_type == "dense":
            self.stage4 = DenseBlock(512, 1024)
        elif backbone_type == "attention":
            self.stage4 = nn.Sequential(
                ResidualBlock(512, 1024, stride=2),
                SelfAttention(1024),
                ASPP(1024, 1024)
            )
        elif backbone_type == "se":
            self.stage4 = nn.Sequential(
                ResidualBlock(512, 1024, stride=2),
                SEBlock(1024),
                ASPP(1024, 1024)
            )
        elif backbone_type == "hybrid":
            dense4_in, dense4_out, dense4_layers = dense3_final, 1024, 4
            dense4_final = dense4_in + dense4_layers * dense4_out
            self.stage4 = nn.Sequential(
                DenseBlock(dense4_in, dense4_out, num_layers=dense4_layers),
                SEBlock(dense4_final),
                SelfAttention(dense4_final)
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