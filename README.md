# TACS - Traffic-Aware Control System

A C++17 project that explores AI-based traffic management, written from scratch without an external machine learning framework. It covers object detection, object tracking, accident and weather classification, and reinforcement learning for signal control, plus a traffic simulation you can run to watch the pieces work together.

This is a research and learning project. The networks, training loops, and math are all hand-written, so treat it as experimental rather than a finished product.

## Components

- TACSNet: a small YOLO-style detector for vehicles, pedestrians, and cyclists
- MemoryTracker: multi-object tracking with an Extended Kalman Filter and Hungarian assignment
- AccidentNet: a convolutional plus GRU classifier over short frame windows
- WeatherNet: a compact image classifier for weather conditions
- RLPolicyNet: an advantage actor-critic (A2C) policy for signal phase control
- Plugin learning: a feature extractor plus shallow classifier for adding new object classes
- Simulation: an SDL2 traffic simulation frontend (optional, built when SDL2 is found)
- Supporting code for layers, training, federated aggregation, drones, and utilities

Source lives under `src/` and `include/`, grouped by area (`layers`, `tracking`, `rl`, `training`, `plugin`, `simulation`, `federated`, `drone`, `utils`).

## Requirements

- A C++17 compiler (GCC 8+, Clang 10+, or MSVC 2019+)
- CMake 3.16+
- OpenMP (optional, used when available)
- OpenCV (optional)
- SDL2 (optional, needed for the simulation frontend)

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

The build produces several executables in `build/`:

- `phase1_validation` through `phase10_validation`: validation programs for each part of the system
- `phase10_console_demo`: a console demo of the full pipeline
- `tacs_simulation`: the SDL2 simulation (only built when SDL2 is available)
- `tacs_edge_runtime`: the edge runtime entry point
- `train_tacsnet` and `train_tacsnet_dataset`: detector training programs
- `create_pretrained_weights`: generates starter weight files

For example:

```bash
./build/phase10_console_demo
./build/tacs_simulation
```

A default configuration file is in `config/`.

## Training

`train.py` and `train_tacsnet.cpp` drive detector training. Weights can be generated with `create_pretrained_weights`. Training is CPU-based by default.

## License

Released under CC BY-NC-SA 4.0 (see `LICENSE`). Noncommercial use with attribution and share-alike.
