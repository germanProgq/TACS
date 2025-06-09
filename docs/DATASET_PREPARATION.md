# TACSNet Dataset Preparation Guide

## Overview

This guide explains how to prepare training data for TACSNet multi-class object detection system targeting cars, pedestrians, and cyclists with >92% mAP.

## Dataset Format

TACSNet uses YOLO format annotations with the following structure:

```
datasets/
└── traffic_detection/
    ├── train/
    │   ├── images/
    │   │   ├── 000001.jpg
    │   │   ├── 000002.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── 000001.txt
    │       ├── 000002.txt
    │       └── ...
    └── val/
        ├── images/
        └── labels/
```

## Annotation Format

Each `.txt` file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: 0=car, 1=pedestrian, 2=cyclist
- Coordinates are normalized to [0, 1]
- Center coordinates (x_center, y_center) relative to image width/height
- Box dimensions (width, height) relative to image width/height

Example:
```
0 0.456 0.543 0.123 0.234
1 0.234 0.678 0.045 0.089
2 0.789 0.321 0.067 0.134
```

## Generating Synthetic Data

For initial testing, use the provided synthetic data generator:

```bash
cd scripts
chmod +x generate_synthetic_dataset.py
python3 generate_synthetic_dataset.py --output-dir ../datasets/traffic_detection/ --num-train 5000 --num-val 1000
```

## Using Real Data

### Recommended Datasets

1. **COCO Dataset** (cars, pedestrians, cyclists subset)
   - Filter relevant classes
   - Convert to YOLO format
   
2. **Cityscapes Dataset**
   - Urban traffic scenes
   - High-quality annotations
   
3. **BDD100K**
   - Diverse driving conditions
   - Day/night variations

### Data Requirements

For production-grade performance (>92% mAP):
- **Minimum**: 10,000 training images
- **Recommended**: 50,000+ training images
- **Class balance**: ~40% cars, ~40% pedestrians, ~20% cyclists
- **Variations**: Different lighting, weather, angles, occlusions

### Annotation Guidelines

1. **Bounding Boxes**
   - Tight boxes around visible portions
   - Include partially occluded objects (>30% visible)
   - Exclude heavily occluded objects (<30% visible)

2. **Class Definitions**
   - **Cars**: All 4-wheeled vehicles (sedans, SUVs, vans, trucks)
   - **Pedestrians**: Standing/walking people
   - **Cyclists**: People on bicycles/motorcycles

3. **Edge Cases**
   - Parked cars: Include if >50% visible
   - Groups: Annotate individuals separately
   - Distant objects: Include if >20 pixels in size

## Data Augmentation

The training pipeline includes:
- Random horizontal flip
- Random brightness/contrast adjustment
- Random scale (0.8-1.2x)
- Random crop
- Synthetic noise addition

## Quality Control

Before training:
1. Verify annotation format
2. Check class distribution
3. Visualize random samples
4. Remove corrupted images
5. Ensure consistent image dimensions

## Training Command

Once data is prepared:

```bash
./build/train_tacsnet \
    --dataset ./datasets/traffic_detection/ \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.001 \
    --checkpoint-dir ./checkpoints/
```

## Performance Expectations

With proper data:
- 10k images: ~85% mAP
- 25k images: ~90% mAP  
- 50k+ images: >92% mAP (target)

## Notes

- Image size: All images resized to 416x416 during training
- Format: JPEG recommended for smaller dataset size
- GPU memory: 16 batch size requires ~8GB VRAM
- Training time: ~2 hours per 10k images on modern GPU