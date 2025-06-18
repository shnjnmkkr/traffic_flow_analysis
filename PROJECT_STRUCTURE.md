# Vehicle Detection Project Structure

## 📁 Complete Directory Organization

```
Vehicle Detection/
├── 📋 README.md                    # Main project documentation
├── 📋 PROJECT_STRUCTURE.md         # This file - project organization guide
├── 📋 requirements.txt             # Global dependencies
├── 📋 .gitignore                   # Git ignore rules
│
├── 🏠 Main/                        # Primary density-based approach
│   ├── 📋 README.md               # Detailed Main pipeline documentation
│   ├── 📋 requirements.txt        # Main pipeline dependencies
│   ├── 📋 config.yaml            # Main pipeline configuration
│   ├── 📋 train.py               # Main training script
│   ├── 📋 inference.py           # Main inference script
│   │
│   ├── 🧠 models/                 # Custom architecture implementations
│   │   ├── 📋 backbone.py        # ResNet + Inception + ASPP + SE blocks
│   │   ├── 📋 fpn.py             # Feature Pyramid Network
│   │   ├── 📋 counting_head.py   # Density and count heads
│   │   ├── 📋 loss.py            # Multi-task loss functions
│   │   └── 📋 vehicle_counter.py # Main model class
│   │
│   ├── 📊 data/                   # Dataset handling
│   │   └── 📋 dataset.py         # Custom dataset with density maps
│   │
│   └── 🛠️ utils/                  # Utility functions
│       ├── 📋 logger.py          # Training logger
│       └── 📋 metrics.py         # Evaluation metrics
│
├── 🎯 RefineDet/                   # Alternative detection-based approach
│   ├── 📋 README.md              # Detailed RefineDet documentation
│   │
│   └── alt_pipeline/              # RefineDet implementation
│       ├── 📋 pipeline.py        # End-to-end pipeline
│       │
│       ├── 🔍 detection/         # RefineDet core components
│       │   ├── 📋 refinedet.py   # Main RefineDet model
│       │   ├── 📋 arm.py         # Anchor Refinement Module
│       │   ├── 📋 tcb.py         # Transfer Connection Block
│       │   ├── 📋 odm.py         # Object Detection Module
│       │   ├── 📋 vgg16.py       # VGG16 backbone
│       │   ├── 📋 loss.py        # RefineDet loss function
│       │   ├── 📋 train_refinedet.py # Training script
│       │   └── 📋 config.yaml    # RefineDet configuration
│       │
│       ├── 🔢 counting/          # Counting components
│       │   ├── 📋 gmm.py         # GMM background subtraction
│       │   └── 📋 count.py       # Vehicle counter
│       │
│       └── 📊 utils/             # RefineDet utilities
│           └── 📋 metrics.py     # Detection metrics
│
├── 📊 traffic_wala_dataset/        # Dataset directory
│   ├── 📋 data.yaml              # Dataset configuration
│   ├── 📁 train/                 # Training data
│   │   ├── 📁 images/           # Training images
│   │   └── 📁 labels/           # Training annotations
│   ├── 📁 valid/                 # Validation data
│   │   ├── 📁 images/           # Validation images
│   │   └── 📁 labels/           # Validation annotations
│   └── 📁 test/                  # Test data
│       ├── 📁 images/           # Test images
│       └── 📁 labels/           # Test annotations
│
├── 📈 results/                     # Inference results and visualizations
├── 📝 logs/                       # Training logs and metrics
├── 💾 checkpoints/                # Model checkpoints
├── 🗄️ cache/                      # Cached processed data
│
└── 📚 Documentation/               # Additional documentation
    ├── 📋 Zhang_Single-Shot_Refinement_Neural_CVPR_2018_paper.pdf
    └── 📋 TSP_CSSE_37928.pdf
```

## 🎯 Architecture Comparison

### Main Pipeline (Density-Based)
- **Approach**: Density map regression for vehicle counting
- **Architecture**: Custom hybrid (ResNet + Inception + ASPP + SE + FPN)
- **Advantages**: Fast, robust to occlusion, accurate counting
- **Use Case**: Primary solution for vehicle counting

### RefineDet Pipeline (Detection-Based)
- **Approach**: Object detection with bounding box predictions
- **Architecture**: RefineDet (ARM + TCB + ODM) with VGG16 backbone
- **Advantages**: Precise localization, class-aware, standard detection
- **Use Case**: Alternative approach for detailed vehicle analysis

## 🚀 Quick Start Guide

### 1. Choose Your Approach

**For Vehicle Counting (Recommended):**
```bash
cd Main
python train.py
```

**For Vehicle Detection:**
```bash
cd RefineDet/alt_pipeline/detection
python train_refinedet.py --config config.yaml --data ../../traffic_wala_dataset
```

### 2. Configuration

**Main Pipeline:**
- Edit `Main/config.yaml` for model architecture, training parameters, and data augmentation
- Key settings: `backbone_type`, `batch_size`, `learning_rate`, `loss_weights`

**RefineDet Pipeline:**
- Edit `RefineDet/alt_pipeline/detection/config.yaml` for detection parameters
- Key settings: `num_classes`, `num_anchors`, `iou_threshold`

### 3. Dataset Preparation

Both pipelines use the same dataset structure:
```
traffic_wala_dataset/
├── train/images/     # Training images
├── train/labels/     # YOLO format annotations
├── valid/images/     # Validation images
├── valid/labels/     # Validation annotations
├── test/images/      # Test images
└── test/labels/      # Test annotations
```

## 📊 Performance Metrics

### Main Pipeline Metrics
- **MAE**: Mean Absolute Error in counting
- **RMSE**: Root Mean Square Error
- **R² Score**: Model fit quality
- **Pearson Correlation**: Linear relationship strength
- **Accuracy at thresholds**: 10%, 20%, 30% error tolerance

### RefineDet Pipeline Metrics
- **Precision**: Detection accuracy
- **Recall**: Detection completeness
- **F1-Score**: Harmonic mean of precision and recall
- **mAP**: Mean Average Precision
- **Counting accuracy**: Post-processing vehicle count

## 🔧 Customization Guide

### Adding New Architectures

1. **Main Pipeline**: Add new blocks in `Main/models/backbone.py`
2. **RefineDet Pipeline**: Modify `RefineDet/alt_pipeline/detection/refinedet.py`

### Modifying Loss Functions

1. **Main Pipeline**: Edit `Main/models/loss.py`
2. **RefineDet Pipeline**: Edit `RefineDet/alt_pipeline/detection/loss.py`

### Data Augmentation

1. **Main Pipeline**: Modify `Main/config.yaml` under `train_transform`
2. **RefineDet Pipeline**: Add transforms in training script

## 📝 File Descriptions

### Core Model Files

| File | Purpose | Pipeline |
|------|---------|----------|
| `backbone.py` | Custom hybrid architecture | Main |
| `fpn.py` | Feature Pyramid Network | Main |
| `counting_head.py` | Density and count heads | Main |
| `refinedet.py` | RefineDet main model | RefineDet |
| `arm.py` | Anchor Refinement Module | RefineDet |
| `tcb.py` | Transfer Connection Block | RefineDet |
| `odm.py` | Object Detection Module | RefineDet |

### Training Scripts

| File | Purpose | Pipeline |
|------|---------|----------|
| `train.py` | Main training loop | Main |
| `train_refinedet.py` | RefineDet training | RefineDet |
| `inference.py` | Inference and visualization | Main |
| `pipeline.py` | End-to-end RefineDet pipeline | RefineDet |

### Configuration Files

| File | Purpose | Pipeline |
|------|---------|----------|
| `config.yaml` | All training parameters | Main |
| `config.yaml` | RefineDet parameters | RefineDet |

## 🎯 Usage Examples

### Training Main Pipeline
```bash
cd Main
# Edit config.yaml for your needs
python train.py
```

### Training RefineDet Pipeline
```bash
cd RefineDet/alt_pipeline/detection
# Edit config.yaml for your needs
python train_refinedet.py --config config.yaml --data ../../traffic_wala_dataset
```

### Inference
```bash
# Main pipeline
cd Main
python inference.py

# RefineDet pipeline
cd RefineDet/alt_pipeline
python pipeline.py
```

### Custom Model Creation
```python
# Main pipeline
from Main.models.vehicle_counter import VehicleCounter
model = VehicleCounter(backbone_type="se")

# RefineDet pipeline
from RefineDet.alt_pipeline.detection.refinedet import RefineDet
model = RefineDet(num_classes=2)
```

## 🔍 Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the correct directory
2. **CUDA out of memory**: Reduce batch size in config files
3. **Training instability**: Check learning rate and loss weights
4. **Poor performance**: Verify dataset format and annotations

### Performance Optimization

1. **Mixed precision training**: Enable in training scripts
2. **Gradient accumulation**: For larger effective batch sizes
3. **Data loading optimization**: Adjust `num_workers` in DataLoader
4. **Model checkpointing**: Regular saves to resume training

## 📚 References

- **Main Pipeline**: Inspired by [traffic flow analysis](https://github.com/shnjnmkkr/traffic_flow_analysis)
- **RefineDet Pipeline**: Based on [official RefineDet implementation](https://github.com/sfzhang15/RefineDet)
- **Architecture Papers**: ResNet, Inception, ASPP, SE Blocks, FPN

---

**This project provides two complementary approaches for vehicle detection and counting, allowing users to choose the best method for their specific use case.** 