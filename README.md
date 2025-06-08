# TACS - Traffic-Aware Control System

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-blue.svg)](https://isocpp.org/)
[![Build System](https://img.shields.io/badge/build-CMake-green.svg)](https://cmake.org/)

## Overview

The Traffic-Aware Control System (TACS) is a fully offline, C++-based AI platform for intelligent traffic management. Built from scratch without reliance on paid cloud services, TACS provides real-time traffic optimization through advanced AI techniques including object detection, accident classification, weather recognition, and reinforcement learning.

## Key Features

### 🚗 Multi-Object Detection
- **TACSNet**: Custom YOLO-inspired architecture for detecting vehicles, pedestrians, and cyclists
- Real-time detection with ≥92% mAP accuracy
- Extensible to new object classes (e-scooters, buses, motorcycles)

### 🎯 Intelligent Tracking
- **MemoryTracker**: Kalman filter-based multi-object tracking with Hungarian algorithm
- 6D state tracking (position, velocity, dimensions)
- Unified tracking for all object types

### 🚨 Accident Detection
- **AccidentNet**: CNN + GRU architecture for incident classification
- Detects rear-end, side-impact, and pile-up accidents
- ≥88% classification accuracy

### 🌤️ Weather Recognition
- **WeatherNet**: ResNet10-based weather classification
- Real-time weather condition detection
- ≥85% accuracy across weather conditions

### 🤖 Reinforcement Learning
- **RLPolicyNet**: Advantage Actor-Critic (A2C) for traffic signal optimization
- Peer-to-peer agent voting for emergency overrides
- Real-time traffic flow optimization

### 🔗 V2X Integration
- Vehicle-to-Infrastructure (V2I) communication
- Inter-intersection synchronization
- Swarm drone coverage with Voronoi-based zoning

### 🌐 Federated Learning
- Distributed learning across multiple intersections
- Server distillation bank for knowledge aggregation
- Rollback support for model safety

## Technical Specifications

### Performance Targets
- **Latency**: ≤50ms per frame (detection to decision)
- **Detection mAP**: ≥92% for vehicles, pedestrians, cyclists
- **Accident Classification**: ≥88% accuracy
- **Weather Classification**: ≥85% accuracy
- **Plugin Training**: ≤120 seconds for new object classes

### System Requirements
- C++17 compatible compiler
- CMake 3.10+
- Optional: CUDA-capable GPU for acceleration
- Optional: SDL2/OpenGL for simulation frontend

## Quick Start

### Build Instructions

```bash
# Clone the repository
git clone <repository-url>
cd traffic_ai

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Run the system
./tacs --config ../config.yaml
```

### Basic Usage

```bash
# Start TACS with default configuration
./tacs

# Run with custom config
./tacs --config /path/to/config.yaml

# Enable simulation mode
./tacs --simulation
```

## Architecture Overview

### Core Components

1. **TACSNet** - Multi-class object detection network
2. **MemoryTracker** - Kalman filter-based object tracking
3. **AccidentNet** - Temporal accident classification
4. **WeatherNet** - Environmental condition recognition
5. **RLPolicyNet** - Traffic signal optimization via RL
6. **Federator** - Distributed learning coordination

### Network Architecture

```
Input Frame (RGB) → TACSNet → Detection Results
                             ↓
MemoryTracker → Object Tracks → RLPolicyNet → Signal Control
                             ↓
AccidentNet → Incident Classification
                             ↓
WeatherNet → Weather Conditions
```

## Advanced Features

### ONNX Export
All models support ONNX export for edge deployment:

```cpp
// Export trained model to ONNX
exportToONNX(tacsnet, "TACSNet.onnx");

// Load on edge device
auto model = loadONNXModel("TACSNet.onnx");
```

### Plugin Object Training
Add new object classes in under 2 minutes:

```bash
# Add new object with image and metadata
./tacs --add-object electric_scooter --image scooter.png --meta meta.json
```

### Federated Learning
Deploy across multiple intersections:

```bash
# Start as federated node
./tacs --federated --node-id intersection_001

# Connect to distillation server
./tacs --server-url http://traffic-server:8080/api/sync
```

## Deployment

### Production Deployment

1. **Compile for production**:
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make
   ```

2. **Install as systemd service**:
   ```bash
   sudo cp scripts/tacs.service /etc/systemd/system/
   sudo systemctl enable tacs
   sudo systemctl start tacs
   ```

3. **ONNX Edge Deployment**:
   ```bash
   # Export models
   ./tacs --export-onnx

   # Deploy to edge device
   scp *.onnx edge-device:/opt/tacs/models/
   ```

### Configuration

Key configuration options in `config.yaml`:

```yaml
detection:
  confidence_threshold: 0.5
  nms_threshold: 0.4
  input_size: 416

tracking:
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3

rl_policy:
  learning_rate: 0.001
  gamma: 0.99
  entropy_coef: 0.01

v2x:
  enabled: true
  protocol: "dsrc"
  broadcast_interval: 100ms
```

## Simulation Frontend

Interactive traffic simulation with real-time visualization:

### Controls
- **Mouse**: Place new entities
- **O**: Add obstacle
- **X**: Create accident scenario
- **R**: Reload/retrain model
- **ESC**: Exit simulation

### Visualizations
- Color-coded bounding boxes by object type
- Real-time signal state indicators
- Weather condition overlays
- Accident classification labels

## Development

### Project Structure

```
traffic_ai/
├── include/          # Header files
│   ├── core/         # Core tensor and memory management
│   ├── layers/       # Neural network layers
│   ├── models/       # Model architectures
│   ├── training/     # Training utilities
│   └── utils/        # Utility functions
├── src/              # Source implementations
├── tests/            # Unit tests
├── build/            # Build directory
└── scripts/          # Deployment scripts
```

### Building Tests

```bash
cd build
make test_runner
./test_runner
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Ensure all tests pass
5. Submit a pull request

## Performance Optimization

### Quantization Support
- Manual FP16 and INT8 quantization
- Up to 4× speedup with minimal accuracy loss
- Custom SIMD optimizations

### Memory Management
- Memory-mapped replay buffers
- Pinned memory for real-time constraints
- Efficient tensor operations

### Hardware Acceleration
- ONNX-TensorRT hybrid deployment
- CUDA kernel optimization
- Multi-threading support

## Research & Citations

This system implements several state-of-the-art techniques:
- Custom YOLO-inspired object detection
- Kalman filtering with Hungarian assignment
- Advantage Actor-Critic reinforcement learning
- Federated learning with knowledge distillation

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Traffic management research community
- Open-source computer vision libraries
- Contributors to this project

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact the development team
- Check the documentation wiki

---

**TACS** - Revolutionizing traffic management through intelligent AI systems.