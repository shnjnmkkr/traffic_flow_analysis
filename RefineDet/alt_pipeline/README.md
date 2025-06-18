# Alternative Vehicle Detection and Counting Pipeline

This alternative pipeline is inspired by the RefineDet architecture and focuses on **vehicle detection and counting** using your single-class YOLO dataset.

## Overview

Unlike the main pipeline which includes classification, this alternative approach focuses on:
- **Vehicle Detection**: Using RefineDet with VGG16 backbone
- **Vehicle Counting**: Using GMM background subtraction and tracking
- **Single Class**: Optimized for your dataset with only vehicle class (class 0)

## Architecture

### Detection (RefineDet)
- **Backbone**: VGG16 with feature extraction
- **ARM (Anchor Refinement Module)**: Refines anchor boxes
- **TCB (Transfer Connection Block)**: Connects ARM and ODM
- **ODM (Object Detection Module)**: Final detection predictions
- **Output**: Bounding boxes for vehicles (class 0)

### Counting
- **GMM Background Subtraction**: Identifies moving objects
- **Vehicle Counter**: Tracks and counts detected vehicles
- **Integration**: Combines detection results with motion analysis

## Dataset Compatibility

This pipeline is specifically designed for your dataset:
- **Format**: YOLO format with `.txt` label files
- **Classes**: Single vehicle class (class 0)
- **Structure**: 
  ```
  traffic_wala_dataset/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── valid/
  │   ├── images/
  │   └── labels/
  └── data.yaml
  ```

## Quick Start

### 1. Test the Pipeline
```bash
cd alt_pipeline
python test_pipeline.py
```

### 2. Train the Detection Model
```bash
python detection/train_refinedet.py --data ../traffic_wala_dataset
```

### 3. Run Inference
```bash
# Single image
python pipeline.py --model checkpoints/final_model.pth --input your_image.jpg

# Video
python pipeline.py --model checkpoints/final_model.pth --input your_video.mp4 --output result.mp4
```

## File Structure

```
alt_pipeline/
├── detection/
│   ├── refinedet.py      # Main RefineDet model
│   ├── vgg16.py          # VGG16 backbone
│   ├── arm.py            # Anchor Refinement Module
│   ├── tcb.py            # Transfer Connection Block
│   ├── odm.py            # Object Detection Module
│   ├── train_refinedet.py # Training script
│   └── yolo_dataset.py   # YOLO dataset loader
├── counting/
│   ├── gmm.py            # GMM background subtraction
│   └── count.py          # Vehicle counting logic
├── utils/
│   └── metrics.py        # Evaluation metrics
├── pipeline.py           # Main pipeline integration
├── config.yaml           # Training configuration
├── test_pipeline.py      # Testing script
└── README.md            # This file
```

## Key Features

### Detection Training
- **Real Dataset Support**: Works with your YOLO-annotated images
- **Logging**: Comprehensive training logs to file and console
- **Checkpointing**: Save/load training progress
- **Metrics History**: CSV export for analysis
- **Resume Training**: Continue from any checkpoint

### Pipeline Integration
- **End-to-End**: Detection → Counting → Results
- **Image/Video Support**: Process both images and videos
- **Visualization**: Draw bounding boxes and counts
- **Metrics**: Track detection performance

### Configuration
- **YAML Config**: Easy parameter tuning
- **Device Selection**: Automatic CUDA/CPU detection
- **Flexible Paths**: Configurable dataset and output paths

## Training Configuration

The `config.yaml` file contains all training parameters:

```yaml
# Dataset
dataset:
  num_classes: 1  # Only vehicle class
  img_size: 512
  batch_size: 8

# Training
training:
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 1e-4
```

## Output

### Detection Results
- **Bounding Boxes**: Vehicle locations with confidence scores
- **Count**: Total number of vehicles in frame
- **Metrics**: Detection accuracy and performance

### Training Output
- **Logs**: Detailed training progress
- **Checkpoints**: Model snapshots for resume
- **Metrics CSV**: Training history for plotting

## Advantages Over Main Pipeline

1. **Simplified**: No classification complexity
2. **Focused**: Optimized for vehicle counting
3. **Efficient**: Single-class detection is faster
4. **Compatible**: Works directly with your dataset
5. **Modular**: Easy to modify and extend

## Next Steps

1. **Test**: Run `test_pipeline.py` to verify setup
2. **Train**: Train the detection model on your dataset
3. **Evaluate**: Test on validation images
4. **Deploy**: Use for real-time vehicle counting

## Troubleshooting

### Common Issues
- **Import Errors**: Ensure all dependencies are installed
- **Dataset Path**: Check that dataset paths are correct
- **Memory Issues**: Reduce batch size in config
- **CUDA Issues**: Set device to 'cpu' if GPU unavailable

### Getting Help
- Check the logs in `logs/` directory
- Verify dataset structure matches expected format
- Test individual components with `test_pipeline.py` 