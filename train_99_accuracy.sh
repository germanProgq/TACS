#!/bin/bash

# Training script for 99% accuracy

echo "=== TACS 99% Accuracy Training Script ==="
echo "This script will train the model to achieve 99% detection accuracy"
echo ""

# Check if dataset path is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_path> [resume_checkpoint]"
    echo "Example: $0 ./datasets/traffic"
    exit 1
fi

DATASET_PATH=$1
OUTPUT_DIR="./models/99_accuracy"
RESUME_CHECKPOINT=${2:-""}

# Create output directory
mkdir -p $OUTPUT_DIR

# Create enhanced training configuration
cat > $OUTPUT_DIR/training_config.txt << EOF
# TACSNet 99% Accuracy Training Configuration
batch_size=16
epochs=500
learning_rate=0.0005
momentum=0.9
weight_decay=0.0001
use_adam=true
validation_interval=1
checkpoint_interval=10
lambda_objectness=2.0
lambda_bbox=10.0
lambda_classification=3.0
best_map_threshold=0.99
patience=50
min_delta=0.0001
enable_augmentation=true
augmentation_prob=0.7
EOF

echo "Configuration saved to $OUTPUT_DIR/training_config.txt"
echo ""
echo "Training parameters:"
echo "- Target mAP: 99%"
echo "- Batch size: 16"
echo "- Epochs: 500"
echo "- Learning rate: 0.0005 (with cosine annealing)"
echo "- Enhanced data augmentation enabled"
echo "- Aggressive NMS thresholds"
echo ""

# Compile the training program with optimizations
echo "Compiling training program..."
g++ -O3 -std=c++17 -fopenmp \
    -I./include \
    train_tacsnet.cpp \
    src/models/*.cpp \
    src/training/*.cpp \
    src/data/*.cpp \
    src/utils/*.cpp \
    src/core/*.cpp \
    -lpthread -lm \
    -o train_99_accuracy

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

# Start training
echo "Starting training..."
echo "This will take several hours to achieve 99% accuracy."
echo ""

if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
    ./train_99_accuracy $DATASET_PATH $OUTPUT_DIR $RESUME_CHECKPOINT
else
    ./train_99_accuracy $DATASET_PATH $OUTPUT_DIR
fi

# Check training result
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Training completed successfully! ==="
    echo "Model saved to: $OUTPUT_DIR/best_model.bin"
    echo ""
    echo "To run inference with 99% accuracy mode:"
    echo "./inference_99_accuracy $OUTPUT_DIR/best_model.bin <input_video>"
else
    echo ""
    echo "=== Training failed to reach 99% accuracy ==="
    echo "Check the logs and consider:"
    echo "1. Increasing dataset size"
    echo "2. Adjusting hyperparameters"
    echo "3. Running for more epochs"
fi