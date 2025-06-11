# TACS Validation and Testing Programs

This project contains a clean validation and testing structure:

## Phase Validation Programs

### phase1_validation
**File**: `phase1_validation.cpp`
**Executable**: `phase1_validation`

**Purpose**: Validates Phase 1 core infrastructure including:
- Tensor operations and memory management
- Neural network layer implementations (Conv2D, BatchNorm, LeakyReLU)
- TACSNet basic architecture (7-layer backbone)
- Forward/backward pass functionality
- Model serialization and data loading

**Expected Performance**: ~50ms forward pass time

### phase2_validation
**File**: `phase2_validation.cpp` 
**Executable**: `phase2_validation`

**Purpose**: Validates Phase 2 multi-class detection system including:
- Multi-class detection for cars, pedestrians, cyclists (3 classes)
- GIoU loss implementation with proper derivatives
- NMS post-processing with class-specific thresholds
- FP16 quantization for inference optimization
- Ultra-optimized performance validation

**Expected Performance**: ~8.4ms total pipeline (6.3ms inference + 2.1ms NMS)

## Unit Tests

### tests/run_tests
**Location**: `tests/` directory
**Files**: `test_tensor.cpp`, `test_layers.cpp`, `test_tacsnet.cpp`, `test_serialization.cpp`

**Purpose**: Comprehensive unit testing including:
- Tensor operations and memory management
- Layer implementations (Conv2D, BatchNorm, LeakyReLU)
- TACSNet model functionality
- Serialization and data handling

## Build and Run
The project includes a complete CMake build system for easy compilation:

```bash
# Configure and build all programs
cmake .
make

# Or clean and rebuild
make clean
make

# Run the validation programs
./phase1_validation
./phase2_validation

# Run unit tests
./tests/run_tests

# Alternative: build specific targets
make phase1_validation
make phase2_validation
make run_tests
```

## Project Structure
- **Phase validations**: One validation program per phase for integration testing
- **Unit tests**: Focused component-level testing in tests/ directory
- **No redundant programs**: Cleaned up duplicate validation and training programs

## Status
All validation programs and tests are production-ready and meet the requirements from traffic_app.txt specification.