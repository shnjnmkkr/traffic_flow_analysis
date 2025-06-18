# Main Pipeline - Density-Based Vehicle Counting

## 🏗️ Architecture Overview

The Main pipeline implements a **density-based vehicle counting** approach using a custom hybrid architecture that combines multiple state-of-the-art deep learning techniques.

## 🧠 Custom Architecture Components

### 1. Hybrid Backbone Architecture

Our custom backbone combines four powerful architectural elements:

#### **ResNet-Style Blocks**
- **Deep residual connections** for gradient flow
- **Bottleneck design** for efficient computation
- **Batch normalization** for stable training
- **ReLU activation** for non-linearity

```python
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
```

#### **Inception-Style Blocks**
- **Multi-scale feature processing** (1x1, 3x3, 5x5 convolutions)
- **Parallel pathways** for different receptive fields
- **Concatenation** of multi-scale features
- **Efficient computation** with 1x1 bottlenecks

```python
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1x1 pathway
        self.branch1 = nn.Conv2d(in_channels, out_channels//4, 1)
        # 3x3 pathway
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1),
            nn.Conv2d(out_channels//4, out_channels//4, 3, padding=1)
        )
        # 5x5 pathway
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1),
            nn.Conv2d(out_channels//4, out_channels//4, 5, padding=2)
        )
        # Pool pathway
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_channels, out_channels//4, 1)
        )
```

#### **ASPP (Atrous Spatial Pyramid Pooling)**
- **Multi-scale context aggregation** with different dilation rates
- **Global context** through global average pooling
- **Atrous convolutions** to maintain spatial resolution
- **Contextual information** for better understanding

```python
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Different dilation rates: 1, 6, 12, 18
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1)
```

#### **SE (Squeeze-and-Excitation) Blocks**
- **Channel attention mechanism** for adaptive feature recalibration
- **Squeeze operation** (global average pooling)
- **Excitation operation** (FC layers with sigmoid)
- **Channel-wise scaling** for feature enhancement

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

### 2. Feature Pyramid Network (FPN)

- **Multi-scale feature fusion** from different backbone levels
- **Top-down pathway** for semantic information
- **Lateral connections** for spatial details
- **Consistent feature scales** across all levels

### 3. Multi-Head Architecture

#### **Density Head**
- Generates **density maps** for vehicle localization
- **1x1 convolution** for channel reduction
- **Sigmoid activation** for probability output
- **Spatial resolution** matching input image

#### **Count Head**
- Predicts **local vehicle counts** per region
- **Global average pooling** + **FC layers**
- **Regression output** for count prediction
- **Multiple scales** for robustness

#### **Global Count Head**
- Predicts **total vehicle count** in image
- **Global context** aggregation
- **Final count regression**
- **Cross-validation** with local counts

## 🎯 Loss Function

### Multi-Task Loss Combination

```python
class VehicleCountingLoss(nn.Module):
    def __init__(self, density_weight=1.0, count_weight=0.7, global_weight=0.1):
        super().__init__()
        self.density_weight = density_weight
        self.count_weight = count_weight
        self.global_weight = global_weight
        
        self.density_loss = nn.MSELoss()
        self.count_loss = nn.MSELoss()
        self.global_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        # Density map loss
        density_loss = self.density_loss(predictions['density_map'], targets['density_map'])
        
        # Local count loss
        count_loss = self.count_loss(predictions['count'], targets['count'])
        
        # Global count loss
        global_loss = self.global_loss(predictions['global_count'], targets['global_count'])
        
        # Weighted combination
        total_loss = (self.density_weight * density_loss + 
                     self.count_weight * count_loss + 
                     self.global_weight * global_loss)
        
        return total_loss
```

## 📊 Training Configuration

### Model Architecture Selection

```yaml
model:
  backbone_type: "se"              # Full hybrid: SE + ResNet + Inception + ASPP
  backbone_channels: [128, 256, 512, 1024]  # Feature channels at each level
  fpn_channels: 256                # FPN output channels
  dropout_rate: 0.3                # Dropout for regularization
```

### Training Parameters

```yaml
training:
  batch_size: 8                    # Optimal for GPU memory
  epochs: 100                      # Full training cycle
  learning_rate: 0.00005           # Low LR for stable training
  weight_decay: 0.0001             # L2 regularization
  
  scheduler:
    type: 'OneCycleLR'             # Advanced learning rate scheduling
    max_lr: 0.0005                 # Peak learning rate
    div_factor: 25                 # Initial LR division
    final_div_factor: 100          # Final LR division
    pct_start: 0.3                 # Warmup percentage
```

### Loss Weights

```yaml
loss:
  density_weight: 1.0              # Primary loss component
  count_weight: 0.7                # Local count importance
  global_weight: 0.1               # Global count validation
```

## 🚀 Usage

### Training

```bash
cd Main
python train.py
```

### Inference

```bash
cd Main
python inference.py
```

### Custom Model Creation

```python
from models.vehicle_counter import VehicleCounter

# Create model with different architectures
model = VehicleCounter(
    backbone_type="se",           # Full hybrid architecture
    backbone_channels=[128, 256, 512, 1024],
    fpn_channels=256,
    dropout_rate=0.3
)

# Alternative architectures
model_resnet = VehicleCounter(backbone_type="resnet")      # ResNet only
model_inception = VehicleCounter(backbone_type="inception") # Inception only
```

## 📈 Performance Metrics

The Main pipeline provides comprehensive evaluation:

- **MAE (Mean Absolute Error)**: Average counting error
- **RMSE (Root Mean Square Error)**: Counting precision
- **R² Score**: Model fit quality
- **Pearson Correlation**: Linear relationship strength
- **Accuracy at thresholds**: 10%, 20%, 30% error tolerance

## 🔧 Customization

### Adding New Backbone Blocks

```python
# In models/backbone.py
class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Your custom architecture here
        pass
    
    def forward(self, x):
        # Your forward pass here
        return x
```

### Modifying Loss Weights

```python
# In train.py
criterion = VehicleCountingLoss(
    density_weight=1.0,    # Adjust based on your needs
    count_weight=0.7,      # Increase for better counting
    global_weight=0.1      # Decrease if global count is less important
)
```

### Data Augmentation

```yaml
# In config.yaml
train_transform:
  - name: "Resize"
    height: 512
    width: 512
  - name: "HorizontalFlip"
    p: 0.5
  - name: "RandomBrightnessContrast"
    p: 0.2
  - name: "ColorJitter"
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
    p: 0.3
```

## 🎯 Key Advantages

1. **Hybrid Architecture**: Combines best of multiple approaches
2. **Multi-Scale Processing**: Handles vehicles of different sizes
3. **Attention Mechanisms**: Focuses on relevant features
4. **Density-Based**: Robust to occlusion and overlap
5. **Fast Inference**: Single forward pass
6. **Scalable**: Works with varying image sizes
7. **Configurable**: Easy to modify and extend

## 📚 References

- **ResNet**: Deep Residual Learning for Image Recognition
- **Inception**: Going Deeper with Convolutions
- **ASPP**: DeepLab: Semantic Image Segmentation
- **SE Blocks**: Squeeze-and-Excitation Networks
- **FPN**: Feature Pyramid Networks for Object Detection
- **Density Maps**: CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes

---

**This architecture represents the culmination of multiple state-of-the-art techniques, providing robust and accurate vehicle counting capabilities.** 