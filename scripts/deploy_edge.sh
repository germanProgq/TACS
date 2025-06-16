#!/bin/bash

# Production-ready edge deployment script for TACS
# Deploys ONNX models to edge devices with optimal configurations

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

DEVICE_TYPE=""
MODEL_PATH=""
CONFIG_FILE=""
OPTIMIZE_LEVEL="3"
DEPLOYMENT_DIR="/opt/tacs"
SERVICE_NAME="tacs-edge"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Deploy TACS models to edge devices with hardware-specific optimizations.

OPTIONS:
    -m, --model PATH         Path to ONNX model file (required)
    -d, --device TYPE        Device type: cpu, gpu-cuda, gpu-opencl, npu, tpu (auto-detect if not specified)
    -c, --config FILE        Configuration file path
    -o, --optimize LEVEL     Optimization level 0-3 (default: 3)
    -t, --target DIR         Target deployment directory (default: /opt/tacs)
    -s, --service NAME       Service name (default: tacs-edge)
    -h, --help               Show this help message

EXAMPLES:
    $0 -m model.onnx -d gpu-cuda
    $0 -m model.onnx -c edge_config.yaml
    $0 -m model.onnx -d cpu -o 2 -t /home/user/tacs

EOF
    exit 1
}

detect_device() {
    echo "Detecting best available device..."
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected"
        DEVICE_TYPE="gpu-cuda"
        return
    fi
    
    # Check for OpenCL devices
    if command -v clinfo &> /dev/null && clinfo 2>/dev/null | grep -q "Device Type.*GPU"; then
        echo "OpenCL GPU detected"
        DEVICE_TYPE="gpu-opencl"
        return
    fi
    
    # Check for NPU/TPU (platform specific)
    if [ -e "/dev/npu0" ] || [ -e "/sys/class/npu" ]; then
        echo "NPU detected"
        DEVICE_TYPE="npu"
        return
    fi
    
    # Default to CPU
    echo "No accelerator detected, using CPU"
    DEVICE_TYPE="cpu"
}

check_dependencies() {
    echo "Checking dependencies..."
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Error: Model file not found: $MODEL_PATH"
        exit 1
    fi
    
    if [ ! -d "$BUILD_DIR" ]; then
        echo "Error: Build directory not found. Please build the project first."
        exit 1
    fi
    
    if [ ! -f "$BUILD_DIR/tacs_edge_runtime" ]; then
        echo "Error: Edge runtime binary not found. Building..."
        (cd "$PROJECT_ROOT" && mkdir -p build && cd build && cmake .. && make tacs_edge_runtime)
    fi
}

optimize_model() {
    echo "Optimizing model for $DEVICE_TYPE..."
    
    case $DEVICE_TYPE in
        gpu-cuda)
            if [ "$OPTIMIZE_LEVEL" -ge 2 ]; then
                echo "Enabling TensorRT optimizations..."
                EXTRA_ARGS="--enable-tensorrt --fp16"
            fi
            ;;
        cpu)
            if [ "$OPTIMIZE_LEVEL" -ge 1 ]; then
                echo "Enabling CPU-specific optimizations..."
                EXTRA_ARGS="--enable-simd"
                
                # Detect CPU features
                if grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
                    EXTRA_ARGS="$EXTRA_ARGS --avx2"
                elif grep -q "neon" /proc/cpuinfo 2>/dev/null; then
                    EXTRA_ARGS="$EXTRA_ARGS --neon"
                fi
            fi
            ;;
    esac
    
    if [ "$OPTIMIZE_LEVEL" -ge 3 ]; then
        echo "Applying aggressive optimizations..."
        EXTRA_ARGS="$EXTRA_ARGS --fuse-ops --eliminate-dead-nodes --constant-folding"
    fi
}

create_config() {
    echo "Creating deployment configuration..."
    
    cat > "$DEPLOYMENT_DIR/config.yaml" <<EOF
# TACS Edge Deployment Configuration
# Generated on $(date)

runtime:
  device: $DEVICE_TYPE
  optimization_level: $OPTIMIZE_LEVEL
  num_threads: $(nproc)
  
model:
  path: $DEPLOYMENT_DIR/model.onnx
  input_size: [1, 3, 416, 416]
  output_names: ["detection", "classification"]
  
inference:
  batch_size: 1
  enable_profiling: false
  max_latency_ms: 50
  
memory:
  enable_memory_pool: true
  max_workspace_mb: 512
  
logging:
  level: info
  file: /var/log/tacs/edge.log
  
monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_port: 8080
EOF

    if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
        echo "Merging with user config..."
        # Simple merge - user config overrides defaults
        cat "$CONFIG_FILE" >> "$DEPLOYMENT_DIR/config.yaml"
    fi
}

deploy_files() {
    echo "Deploying files to $DEPLOYMENT_DIR..."
    
    # Create deployment directory
    sudo mkdir -p "$DEPLOYMENT_DIR"
    sudo mkdir -p "$DEPLOYMENT_DIR/lib"
    sudo mkdir -p /var/log/tacs
    
    # Copy runtime binary
    sudo cp "$BUILD_DIR/tacs_edge_runtime" "$DEPLOYMENT_DIR/"
    sudo chmod +x "$DEPLOYMENT_DIR/tacs_edge_runtime"
    
    # Copy model
    sudo cp "$MODEL_PATH" "$DEPLOYMENT_DIR/model.onnx"
    
    # Copy libraries if needed
    if [ -d "$BUILD_DIR/lib" ]; then
        sudo cp -r "$BUILD_DIR/lib/"* "$DEPLOYMENT_DIR/lib/"
    fi
    
    # Set permissions
    sudo chown -R root:root "$DEPLOYMENT_DIR"
    sudo chmod -R 755 "$DEPLOYMENT_DIR"
}

create_service() {
    echo "Creating systemd service..."
    
    sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" > /dev/null <<EOF
[Unit]
Description=TACS Edge Runtime Service
After=network.target

[Service]
Type=simple
ExecStart=$DEPLOYMENT_DIR/tacs_edge_runtime --config $DEPLOYMENT_DIR/config.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment="LD_LIBRARY_PATH=$DEPLOYMENT_DIR/lib"

# Resource limits
LimitNOFILE=65536
MemoryLimit=2G
CPUQuota=80%

# Security
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/tacs

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable "${SERVICE_NAME}.service"
}

setup_monitoring() {
    echo "Setting up monitoring..."
    
    # Create monitoring script
    sudo tee "$DEPLOYMENT_DIR/monitor.sh" > /dev/null <<'EOF'
#!/bin/bash

# Simple monitoring script for TACS edge deployment

check_service() {
    if systemctl is-active --quiet $1; then
        echo "OK: $1 is running"
        return 0
    else
        echo "ERROR: $1 is not running"
        return 1
    fi
}

check_latency() {
    local latency=$(curl -s http://localhost:8080/metrics | grep "inference_latency_ms" | awk '{print $2}')
    if [ -n "$latency" ] && (( $(echo "$latency < 50" | bc -l) )); then
        echo "OK: Inference latency ${latency}ms < 50ms"
        return 0
    else
        echo "WARNING: Inference latency ${latency}ms"
        return 1
    fi
}

check_memory() {
    local mem_usage=$(ps aux | grep tacs_edge_runtime | grep -v grep | awk '{print $4}')
    echo "INFO: Memory usage: ${mem_usage}%"
}

# Main monitoring loop
while true; do
    echo "=== TACS Edge Monitor - $(date) ==="
    check_service "$SERVICE_NAME"
    check_latency
    check_memory
    echo ""
    sleep 60
done
EOF

    sudo chmod +x "$DEPLOYMENT_DIR/monitor.sh"
}

run_benchmark() {
    echo "Running deployment benchmark..."
    
    "$DEPLOYMENT_DIR/tacs_edge_runtime" \
        --benchmark \
        --config "$DEPLOYMENT_DIR/config.yaml" \
        --iterations 100 \
        --warmup 10
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE_TYPE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -o|--optimize)
            OPTIMIZE_LEVEL="$2"
            shift 2
            ;;
        -t|--target)
            DEPLOYMENT_DIR="$2"
            shift 2
            ;;
        -s|--service)
            SERVICE_NAME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path is required"
    usage
fi

# Main deployment process
echo "=== TACS Edge Deployment Script ==="
echo "Model: $MODEL_PATH"

# Auto-detect device if not specified
if [ -z "$DEVICE_TYPE" ]; then
    detect_device
fi

echo "Device: $DEVICE_TYPE"
echo "Optimization Level: $OPTIMIZE_LEVEL"
echo "Deployment Directory: $DEPLOYMENT_DIR"
echo ""

check_dependencies
optimize_model
deploy_files
create_config
create_service
setup_monitoring

echo ""
echo "Deployment complete!"
echo ""
echo "To start the service:"
echo "  sudo systemctl start ${SERVICE_NAME}"
echo ""
echo "To check status:"
echo "  sudo systemctl status ${SERVICE_NAME}"
echo ""
echo "To view logs:"
echo "  sudo journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "To run monitoring:"
echo "  sudo ${DEPLOYMENT_DIR}/monitor.sh"
echo ""

# Optional: Run benchmark
read -p "Run benchmark now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    run_benchmark
fi