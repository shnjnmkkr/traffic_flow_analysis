# Vehicle Detection and Counting System

A comprehensive deep learning-based vehicle detection and counting system with multiple architectural approaches for traffic flow analysis.

## 🚗 Project Overview

This project implements multiple deep learning architectures for vehicle detection and counting in traffic scenarios. It features both a **density-based approach** (Main) and a **detection-based approach** (RefineDet) to provide comprehensive solutions for traffic analysis.

## 📁 Project Structure

```
Vehicle Detection/
├── Main/                           # Primary density-based approach
│   ├── models/                     # Custom architectures
│   │   ├── backbone.py            # ResNet + Inception + ASPP + SE blocks
│   │   ├── fpn.py                 # Feature Pyramid Network
│   │   ├── counting_head.py       # Density and count heads
│   │   ├── loss.py                # Multi-task loss functions
│   │   └── vehicle_counter.py     # Main model class
│   ├── data/
│   │   └── dataset.py             # Custom dataset with density maps
│   ├── utils/
│   │   ├── logger.py              # Training logger
│   │   └── metrics.py             # Evaluation metrics
│   ├── train.py                   # Training script
│   ├── inference.py               # Inference and visualization
│   ├── config.yaml                # Configuration file
│   └── requirements.txt           # Dependencies
├── RefineDet/                      # Alternative detection-based approach
│   └── alt_pipeline/
│       ├── detection/              # RefineDet implementation
│       │   ├── refinedet.py       # Main RefineDet model
│       │   ├── arm.py             # Anchor Refinement Module
│       │   ├── tcb.py             # Transfer Connection Block
│       │   ├── odm.py             # Object Detection Module
│       │   ├── vgg16.py           # VGG16 backbone
│       │   ├── loss.py            # RefineDet loss function
│       │   ├── train_refinedet.py # Training script
│       │   └── config.yaml        # RefineDet configuration
│       ├── counting/               # Counting components
│       │   ├── gmm.py             # GMM background subtraction
│       │   └── count.py           # Vehicle counter
│       └── pipeline.py            # End-to-end pipeline
├── traffic_wala_dataset/           # Dataset directory
├── results/                        # Inference results
├── logs/                          # Training logs
├── checkpoints/                   # Model checkpoints
└── README.md                      # This file
```

## 🏗️ Architecture Details

### Main Pipeline (Density-Based Approach)

The primary approach uses a **density map regression** method for vehicle counting, inspired by the [traffic flow analysis repository](https://github.com/shnjnmkkr/traffic_flow_analysis).

#### Core Components:

1. **Custom Backbone Architecture**
   - **ResNet-style blocks**: Deep residual connections for feature extraction
   - **Inception-style blocks**: Multi-scale feature processing
   - **ASPP (Atrous Spatial Pyramid Pooling)**: Multi-scale context aggregation
   - **SE (Squeeze-and-Excitation) blocks**: Channel attention mechanisms
   - **FPN (Feature Pyramid Network)**: Multi-scale feature fusion

2. **Multi-Head Architecture**
   - **Density Head**: Generates density maps for vehicle localization
   - **Count Head**: Predicts local vehicle counts
   - **Global Count Head**: Predicts total vehicle count in image

3. **Loss Function**
   - **Density Loss**: MSE loss for density map regression
   - **Count Loss**: MSE loss for count prediction
   - **Global Loss**: MSE loss for global count prediction
   - **Weighted combination** for optimal training

#### Advantages:
- ✅ **Accurate counting**: Direct density map regression
- ✅ **Robust to occlusion**: Handles overlapping vehicles well
- ✅ **Fast inference**: Single forward pass
- ✅ **Scalable**: Works with varying image sizes

### RefineDet Pipeline (Detection-Based Approach)

The alternative approach uses **RefineDet architecture** for object detection and counting.

#### Core Components:

1. **RefineDet Architecture**
   - **VGG16 Backbone**: Feature extraction
   - **ARM (Anchor Refinement Module)**: Coarse anchor generation
   - **TCB (Transfer Connection Block)**: Feature refinement and upsampling
   - **ODM (Object Detection Module)**: Final bounding box and class predictions

2. **Two-Stage Refinement Process**
   - **Stage 1**: ARM generates coarse anchors and objectness scores
   - **Stage 2**: TCB refines features, ODM produces precise detections

3. **Supporting Components**
   - **GMM Background Subtraction**: Motion detection
   - **Vehicle Counter**: Post-processing for counting

#### Advantages:
- ✅ **Precise localization**: Bounding box predictions
- ✅ **Class-aware**: Can distinguish vehicle types
- ✅ **Standard detection**: Uses established detection pipeline

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies for Main pipeline
cd Main
pip install -r requirements.txt

# Install additional dependencies for RefineDet
cd ../RefineDet
pip install torch torchvision opencv-python albumentations
```

### Dataset Preparation

Organize your dataset as follows:

```
traffic_wala_dataset/
├── train/
│   ├── images/                    # Training images
│   └── labels/_annotations.csv    # YOLO format annotations
├── valid/
│   ├── images/                    # Validation images
│   └── labels/_annotations.csv    # YOLO format annotations
└── test/
    ├── images/                    # Test images
    └── labels/_annotations.csv    # YOLO format annotations
```

### Training

#### Main Pipeline (Recommended)

```bash
cd Main
python train.py
```

**Configuration**: Edit `config.yaml` to customize:
- Model architecture (backbone type, channels, etc.)
- Training parameters (batch size, learning rate, epochs)
- Data augmentation (transforms, normalization)
- Loss weights (density, count, global)

#### RefineDet Pipeline

```bash
cd RefineDet/alt_pipeline/detection
python train_refinedet.py --config config.yaml --data ../../traffic_wala_dataset
```

### Inference

#### Main Pipeline

```bash
cd Main
python inference.py
```

#### RefineDet Pipeline

```bash
cd RefineDet/alt_pipeline
python pipeline.py
```

## 📊 Performance Comparison

| Architecture | Approach | Counting Accuracy | Detection Precision | Speed | Memory Usage |
|--------------|----------|------------------|-------------------|-------|--------------|
| **Main Pipeline** | Density-based | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **RefineDet** | Detection-based | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

## 🔧 Configuration

### Main Pipeline Configuration (`Main/config.yaml`)

```yaml
model:
  backbone_type: "se"              # "se", "resnet", "inception"
  backbone_channels: [128, 256, 512, 1024]
  fpn_channels: 256
  dropout_rate: 0.3

training:
  batch_size: 8
  epochs: 100
  learning_rate: 0.00005
  weight_decay: 0.0001

loss:
  density_weight: 1.0
  count_weight: 0.7
  global_weight: 0.1
```

### RefineDet Configuration (`RefineDet/alt_pipeline/detection/config.yaml`)

```yaml
num_epochs: 50
batch_size: 8
learning_rate: 0.001
num_classes: 2  # background + vehicle
num_anchors: 6
```

## 📈 Results and Metrics

The system provides comprehensive evaluation metrics:

- **MAE (Mean Absolute Error)**: Average counting error
- **RMSE (Root Mean Square Error)**: Counting precision
- **R² Score**: Model fit quality
- **Pearson Correlation**: Linear relationship strength
- **Accuracy at different thresholds**: 10%, 20%, 30% error tolerance
- **Precision/Recall**: Detection performance (RefineDet)

## 🛠️ Customization

### Adding New Architectures

1. **Backbone**: Add new blocks in `Main/models/backbone.py`
2. **Heads**: Extend counting heads in `Main/models/counting_head.py`
3. **Loss**: Implement new loss functions in `Main/models/loss.py`

### Data Augmentation

Modify `Main/config.yaml` under `train_transform`:

```yaml
train_transform:
  - name: "Resize"
    height: 512
    width: 512
  - name: "HorizontalFlip"
    p: 0.5
  - name: "RandomBrightnessContrast"
    p: 0.2
```

### Model Architecture Selection

Choose from available architectures:

```python
# In config.yaml
model:
  backbone_type: "se"        # SE + ResNet + Inception + ASPP
  # backbone_type: "resnet"  # ResNet only
  # backbone_type: "inception" # Inception only
```

## 📝 Usage Examples

### Basic Training

```python
# Main pipeline
from Main.models.vehicle_counter import VehicleCounter
from Main.data.dataset import VehicleCountingDataset

model = VehicleCounter(backbone_type="se")
dataset = VehicleCountingDataset("traffic_wala_dataset", split='train')
```

### Custom Inference

```python
# Load trained model
model = VehicleCounter()
model.load_state_dict(torch.load("checkpoints/final_model.pth"))

# Run inference
with torch.no_grad():
    predictions = model(images)
    density_map = predictions['density_map']
    count = predictions['count']
    global_count = predictions['global_count']
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes
4. **Test** thoroughly
5. **Submit** a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Inspired by the [traffic flow analysis repository](https://github.com/shnjnmkkr/traffic_flow_analysis)
- RefineDet implementation based on the [official RefineDet paper](https://github.com/sfzhang15/RefineDet)
- Uses Albumentations for data augmentation
- Built with PyTorch framework

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in each pipeline folder
- Review the configuration files for customization options

---

**Happy vehicle counting! 🚗📊** 