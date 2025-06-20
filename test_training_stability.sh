#\!/bin/bash

echo "=== Testing TACSNet Training Stability ==="
echo "Running training for 5 epochs to check if loss stabilizes..."
echo ""

# Create necessary directories
mkdir -p ./models/pretrained
mkdir -p ./checkpoints

# Run training with pretrained weights (using initialized weights)
echo "Test 1: With pretrained flag (uses proper initialization)"
./build/train_tacsnet --pretrained 2>&1 | head -n 50

echo ""
echo "If loss values are reasonable (not 700+), the training is stable."
echo "Expected initial loss: 5-50 (with proper initialization)"
echo "If loss is still exploding, further debugging is needed."
