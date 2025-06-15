# Deep Learning-Based Vehicle Counting System

## Overview & Ideology
This project implements a robust deep learning model for vehicle counting in static images, designed for flexibility, extensibility, and ease of use. The architecture combines:
- **Custom CNN backbone** (ResNet-style, Inception-style, ASPP)
- **Feature Pyramid Network (FPN)** for multi-scale feature fusion
- **Density-based and regression-based counting heads**
- **Multi-task loss** for improved accuracy

**Ideology:**
- Modular codebase for easy experimentation
- Clear separation of model, data, and utility code
- Designed for both research and production
- Easy to extend (e.g., for data augmentation, new heads, or loss functions)

## Quick Navigation
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Augmentation](#data-augmentation)
- [File-by-File Guide](#file-by-file-guide)
- [Configuration](#configuration)
- [Contributing](#contributing)

## Project Structure
```
.
├── models/
│   ├── backbone.py         # Custom CNN backbone (ResNet, Inception, ASPP)
│   ├── fpn.py              # Feature Pyramid Network
│   ├── counting_head.py    # Density and count heads with attention
│   ├── loss.py             # Multi-task loss function
│   └── vehicle_counter.py  # Main model class
├── data/
│   └── dataset.py          # Custom dataset & augmentation
├── utils/
│   ├── logger.py           # Training logger
│   ├── metrics.py          # Evaluation metrics
│   └── __init__.py
├── train.py                # Training script
├── inference.py            # Inference & visualization
├── config.yaml             # All configuration
├── requirements.txt        # Dependencies
├── traffic_wala_dataset/   # Example dataset structure
└── results/                # Inference results
```

## Installation
1. **Clone the repository:**
```bash
git clone https://github.com/shnjnmkkr/traffic_flow_analysis.git
cd traffic_flow_analysis
```
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage
### 1. Prepare Your Dataset
Organize your data as:
```
traffic_wala_dataset/
├── train/
│   ├── images/
│   └── labels/_annotations.csv
├── valid/
│   ├── images/
│   └── labels/_annotations.csv
└── test/
    ├── images/
    └── labels/_annotations.csv
```

### 2. Configure the Project
Edit `config.yaml` to set paths, model, and training parameters. All transforms, including augmentation, are defined here.

### 3. Training
```bash
python train.py
```
- Logs and checkpoints are saved in `logs/` and `checkpoints/`.

### 4. Inference
```bash
python inference.py
```
- Results are saved in `results/`.

## Data Augmentation
- **Where to modify:**
  - All data augmentation is handled in `data/dataset.py` and configured via `config.yaml` under `train_transform` and `val_transform`.
  - Uses [Albumentations](https://albumentations.ai/) for flexible, powerful augmentation.
- **How to add/modify:**
  - Edit the `train_transform` list in `config.yaml` to add, remove, or change augmentations.
  - Example (in `config.yaml`):
    ```yaml
    train_transform:
      - name: "Resize"
        height: 512
        width: 512
      - name: "HorizontalFlip"
        p: 0.5
      - name: "RandomBrightnessContrast"
        p: 0.2
      - name: "Normalize"
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      - name: "ToTensor"
    ```
  - To implement custom augmentations, edit the `build_transform` function in `train.py`.

## File-by-File Guide
### models/
- **backbone.py**: Custom CNN backbone (ResNet, Inception, ASPP blocks)
- **fpn.py**: Feature Pyramid Network for multi-scale feature fusion
- **counting_head.py**: Density and count heads with attention mechanisms
- **loss.py**: Multi-task loss (density, count, global regression)
- **vehicle_counter.py**: Main model class combining backbone, FPN, and heads

### data/
- **dataset.py**: Custom PyTorch Dataset. Handles loading, density map generation, and applies augmentations. **Edit here for advanced data loading or custom augmentation logic.**

### utils/
- **logger.py**: Logs training metrics
- **metrics.py**: Calculates MAE, RMSE, and other metrics

### Root Scripts
- **train.py**: Main training loop. Loads config, sets up data, model, optimizer, and handles training/validation.
- **inference.py**: Loads trained model, runs inference, and visualizes/saves results.
- **config.yaml**: All configuration (paths, transforms, model, training, loss)
- **requirements.txt**: Python dependencies

## Configuration
- All settings are in `config.yaml`:
  - Data paths, image size
  - Augmentation (see `train_transform`)
  - Model architecture
  - Training parameters (batch size, epochs, learning rate, scheduler, etc.)
  - Loss weights

## Contributing & Extending
- **For data augmentation:**
  - Edit `config.yaml` and/or `data/dataset.py`.
  - Use Albumentations for easy, powerful transforms.
- **For new model features:**
  - Add new modules in `models/` and update `vehicle_counter.py`.
- **For new loss functions:**
  - Add to `models/loss.py` and update training logic in `train.py`.
- **For evaluation/visualization:**
  - Extend `inference.py` or add scripts in `utils/`.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---
**For questions or contributions, please open an issue or pull request!** 