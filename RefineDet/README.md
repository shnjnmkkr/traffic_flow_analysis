# RefineDet Pipeline - Detection-Based Vehicle Counting

## 🎯 Overview

The RefineDet pipeline implements a **detection-based vehicle counting** approach using the RefineDet architecture, which combines the efficiency of one-stage detectors with the accuracy of two-stage methods through a novel two-step refinement process.

## 🏗️ RefineDet Architecture

### Core Components

RefineDet consists of three main modules that work together in a two-stage refinement process:

#### 1. **ARM (Anchor Refinement Module)**
- **Purpose**: Generates coarse anchor boxes and objectness scores
- **Input**: Feature maps from VGG16 backbone
- **Output**: Anchor adjustments and binary objectness predictions
- **Function**: First-stage refinement for anchor quality

```python
class AnchorRefinementModule(nn.Module):
    def __init__(self, in_channels, num_anchors=6):
        super().__init__()
        # Predict anchor deltas (dx, dy, dw, dh) for each anchor
        self.loc = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        # Predict objectness score for each anchor
        self.conf = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=3, padding=1)
    
    def forward(self, x):
        loc = self.loc(x)    # (batch, num_anchors*4, H, W)
        conf = self.conf(x)  # (batch, num_anchors*2, H, W)
        return loc, conf
```

#### 2. **TCB (Transfer Connection Block)**
- **Purpose**: Refines and transfers features from ARM to ODM
- **Input**: ARM features + upsampled features from deeper TCB
- **Output**: Refined features for ODM
- **Function**: Feature refinement and upsampling

```python
class TransferConnectionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, x, up=None):
        out = self.conv1(x)
        out = self.relu(out)
        if up is not None:
            out = out + self.upsample(up)  # Feature fusion
        out = self.conv2(out)
        out = self.relu(out)
        return out
```

#### 3. **ODM (Object Detection Module)**
- **Purpose**: Final bounding box and class predictions
- **Input**: TCB-refined features
- **Output**: Precise bounding box regressions and class scores
- **Function**: Second-stage refinement for final detections

```python
class ObjectDetectionModule(nn.Module):
    def __init__(self, in_channels, num_anchors=6, num_classes=2):
        super().__init__()
        # Predict bounding box deltas (dx, dy, dw, dh) for each anchor
        self.loc = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        # Predict class scores for each anchor
        self.conf = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
    
    def forward(self, x):
        loc = self.loc(x)    # (batch, num_anchors*4, H, W)
        conf = self.conf(x)  # (batch, num_anchors*num_classes, H, W)
        return loc, conf
```

### Two-Stage Refinement Process

```
Input Image
    ↓
VGG16 Backbone
    ↓
Feature Maps (4 levels)
    ↓
┌─────────────────┐
│   ARM Stage 1   │ → Coarse anchors & objectness
└─────────────────┘
    ↓
┌─────────────────┐
│   TCB Stage 2   │ → Feature refinement & upsampling
└─────────────────┘
    ↓
┌─────────────────┐
│   ODM Stage 2   │ → Final detections & classifications
└─────────────────┘
    ↓
Post-processing (NMS, counting)
```

## 🔧 Implementation Details

### VGG16 Backbone
- **Pre-trained weights**: ImageNet initialization
- **Feature extraction**: Multiple scales (64x64, 32x32, 16x16, 8x8)
- **Channel dimensions**: [512, 512, 512, 512]
- **Adaptation**: Modified for detection task

### Anchor Generation
- **Scales**: [0.1, 0.2, 0.4, 0.6] for different feature levels
- **Aspect ratios**: [1.0] (square anchors for vehicles)
- **Total anchors**: ~16,320 for 512x512 input
- **Matching**: IoU-based assignment to ground truth

### Loss Function

```python
class RefineDetLoss(nn.Module):
    def __init__(self, iou_threshold=0.5, neg_pos_ratio=3, alpha=1.0):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        
        # Loss functions
        self.loc_loss = nn.SmoothL1Loss(reduction='sum')
        self.conf_loss = nn.CrossEntropyLoss(reduction='sum')
    
    def forward(self, predictions, targets):
        # Extract predictions
        odm_locs = predictions['odm_locs']
        odm_confs = predictions['odm_confs']
        
        # Flatten and concatenate all feature levels
        locs = torch.cat([loc.view(loc.size(0), -1, 4) for loc in odm_locs], dim=1)
        confs = torch.cat([conf.view(conf.size(0), -1, 2) for conf in odm_confs], dim=1)
        
        # Match anchors to ground truth
        # Compute localization and classification losses
        # Return weighted combination
        return total_loss
```

## 🚀 Usage

### Training

```bash
cd RefineDet/alt_pipeline/detection
python train_refinedet.py --config config.yaml --data ../../traffic_wala_dataset
```

### Configuration

```yaml
# RefineDet/alt_pipeline/detection/config.yaml
num_epochs: 50
batch_size: 8
learning_rate: 0.001
weight_decay: 0.0001
lr_step_size: 20
lr_gamma: 0.1

# Model parameters
num_classes: 2  # background + vehicle
num_anchors: 6
iou_threshold: 0.5
neg_pos_ratio: 3

# Directories
log_dir: logs
checkpoint_dir: checkpoints
```

### Inference

```bash
cd RefineDet/alt_pipeline
python pipeline.py
```

## 🔄 Pipeline Integration

### GMM Background Subtraction
- **Purpose**: Motion detection and background modeling
- **Integration**: Pre-processing step for video sequences
- **Benefits**: Reduces false positives from static objects

```python
class GMMBackgroundSubtractor:
    def __init__(self, history=500, var_threshold=16):
        self.history = history
        self.var_threshold = var_threshold
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold
        )
    
    def process_frame(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        return fg_mask
```

### Vehicle Counter
- **Purpose**: Post-processing for counting detected vehicles
- **Input**: RefineDet detections
- **Output**: Vehicle count and tracking information
- **Features**: Duplicate removal, temporal consistency

## 📊 Performance Metrics

### Detection Metrics
- **Precision**: Ratio of correct detections to total detections
- **Recall**: Ratio of detected vehicles to total vehicles
- **F1-Score**: Harmonic mean of precision and recall
- **mAP**: Mean Average Precision across IoU thresholds

### Counting Metrics
- **MAE**: Mean Absolute Error in vehicle count
- **RMSE**: Root Mean Square Error
- **Accuracy**: Percentage of correct counts within tolerance

## 🎯 Advantages and Limitations

### Advantages
- ✅ **Precise localization**: Bounding box predictions
- ✅ **Class-aware**: Can distinguish vehicle types
- ✅ **Standard detection**: Uses established detection pipeline
- ✅ **Two-stage refinement**: Combines efficiency and accuracy
- ✅ **Multi-scale processing**: Handles vehicles of different sizes

### Limitations
- ❌ **Complex architecture**: More parameters than density-based
- ❌ **Anchor dependency**: Performance depends on anchor design
- ❌ **Slower inference**: Multiple refinement stages
- ❌ **Training complexity**: Requires careful hyperparameter tuning

## 🔧 Customization

### Modifying Anchor Configuration

```python
# In refinedet.py
class RefineDet(nn.Module):
    def __init__(self, num_classes=2, num_anchors=6):
        # Change anchor scales for different vehicle sizes
        self.scales = [0.1, 0.2, 0.4, 0.6]  # Adjust based on dataset
        self.aspect_ratios = [1.0]  # Add more ratios if needed
```

### Adjusting Loss Weights

```python
# In loss.py
class RefineDetLoss(nn.Module):
    def __init__(self, loc_weight=1.0, conf_weight=1.0):
        self.loc_weight = loc_weight    # Localization loss weight
        self.conf_weight = conf_weight  # Classification loss weight
```

### Adding New Backbone

```python
# Replace VGG16 with other backbones
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Implement ResNet backbone
        pass
```

## 📚 References

- **RefineDet Paper**: Single-Shot Refinement Neural Network for Object Detection (CVPR 2018)
- **Original Implementation**: [https://github.com/sfzhang15/RefineDet](https://github.com/sfzhang15/RefineDet)
- **VGG16**: Very Deep Convolutional Networks for Large-Scale Image Recognition
- **SSD**: Single Shot MultiBox Detector
- **Faster R-CNN**: Towards Real-Time Object Detection with Region Proposal Networks

## 🚨 Troubleshooting

### Common Issues

1. **Low detection accuracy**
   - Check anchor scales and aspect ratios
   - Verify IoU threshold settings
   - Ensure proper data augmentation

2. **Training instability**
   - Reduce learning rate
   - Adjust loss weights
   - Check gradient clipping

3. **Memory issues**
   - Reduce batch size
   - Use gradient accumulation
   - Optimize anchor generation

### Performance Optimization

```python
# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(images)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

**The RefineDet pipeline provides a robust detection-based approach for vehicle counting, offering precise localization capabilities with the efficiency of single-shot detectors.** 