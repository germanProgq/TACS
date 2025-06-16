/**
 * @file edge_runtime.cpp
 * @brief Production-ready edge runtime for TACS ONNX model deployment
 * 
 * Standalone executable for running TACS models on edge devices with
 * hardware-specific optimizations and monitoring capabilities.
 */

#include "utils/onnx_runtime.h"
#include "core/tensor.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <csignal>
#include <atomic>
#include <cstring>
#include <vector>
#include <iomanip>
#include <algorithm>

// Simple HTTP server for metrics and health checks
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

// Camera capture support (platform-specific)
#ifdef __linux__
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#endif

#ifdef __APPLE__
#include <CoreMedia/CoreMedia.h>
#include <AVFoundation/AVFoundation.h>
#endif

#ifdef _WIN32
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#endif

using namespace tacs;

std::atomic<bool> g_running(true);
std::atomic<uint64_t> g_total_inferences(0);
std::atomic<double> g_total_latency_ms(0);
std::atomic<double> g_max_latency_ms(0);

// Camera capture implementation
class CameraCapture {
public:
    CameraCapture() : fd_(-1), initialized_(false), width_(0), height_(0) {}
    
    ~CameraCapture() {
        if (initialized_) {
            release();
        }
    }
    
    bool initialize(int device_id, int width, int height) {
#ifdef __linux__
        std::string device_path = "/dev/video" + std::to_string(device_id);
        fd_ = open(device_path.c_str(), O_RDWR);
        if (fd_ < 0) {
            std::cerr << "Failed to open camera device: " << device_path << std::endl;
            return false;
        }
        
        // Query capabilities
        struct v4l2_capability cap;
        if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
            std::cerr << "Failed to query camera capabilities" << std::endl;
            close(fd_);
            return false;
        }
        
        // Set format
        struct v4l2_format fmt;
        memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width;
        fmt.fmt.pix.height = height;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;
        
        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
            std::cerr << "Failed to set camera format" << std::endl;
            close(fd_);
            return false;
        }
        
        width_ = fmt.fmt.pix.width;
        height_ = fmt.fmt.pix.height;
        
        // Request buffers
        struct v4l2_requestbuffers req;
        memset(&req, 0, sizeof(req));
        req.count = 1;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        
        if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
            std::cerr << "Failed to request buffers" << std::endl;
            close(fd_);
            return false;
        }
        
        // Map buffer
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = 0;
        
        if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            std::cerr << "Failed to query buffer" << std::endl;
            close(fd_);
            return false;
        }
        
        buffer_length_ = buf.length;
        buffer_ = mmap(NULL, buffer_length_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
        
        if (buffer_ == MAP_FAILED) {
            std::cerr << "Failed to map buffer" << std::endl;
            close(fd_);
            return false;
        }
        
        // Queue buffer
        if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            std::cerr << "Failed to queue buffer" << std::endl;
            munmap(buffer_, buffer_length_);
            close(fd_);
            return false;
        }
        
        // Start streaming
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
            std::cerr << "Failed to start streaming" << std::endl;
            munmap(buffer_, buffer_length_);
            close(fd_);
            return false;
        }
        
        initialized_ = true;
        return true;
#elif defined(__APPLE__)
        // macOS camera support requires Objective-C++ implementation
        // Would use AVFoundation framework for camera access
        // For production, implement in separate .mm file
        std::cerr << "Camera support on macOS requires AVFoundation (not implemented)" << std::endl;
        return false;
#elif defined(_WIN32)
        // Windows camera support via Media Foundation
        // Initialize COM and Media Foundation
        HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
        if (SUCCEEDED(hr)) {
            hr = MFStartup(MF_VERSION);
            if (SUCCEEDED(hr)) {
                // Would enumerate video capture devices and initialize
                std::cerr << "Windows camera support requires Media Foundation implementation" << std::endl;
                MFShutdown();
            }
            CoUninitialize();
        }
        return false;
#else
        // Camera not supported on this platform
        std::cerr << "Camera support not available on this platform" << std::endl;
        return false;
#endif
    }
    
    bool capture_frame(core::Tensor& output) {
#ifdef __linux__
        if (!initialized_) return false;
        
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        
        // Dequeue buffer
        if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
            std::cerr << "Failed to dequeue buffer" << std::endl;
            return false;
        }
        
        // Convert RGB24 to normalized float and resize to 416x416
        const uint8_t* src = static_cast<const uint8_t*>(buffer_);
        float* dst = output.data_float();
        
        // Simple bilinear resize from captured size to 416x416
        float x_scale = width_ / 416.0f;
        float y_scale = height_ / 416.0f;
        
        for (int y = 0; y < 416; ++y) {
            for (int x = 0; x < 416; ++x) {
                int src_x = static_cast<int>(x * x_scale);
                int src_y = static_cast<int>(y * y_scale);
                
                // Clamp to valid range
                src_x = std::min(src_x, width_ - 1);
                src_y = std::min(src_y, height_ - 1);
                
                int src_idx = (src_y * width_ + src_x) * 3;
                
                // Convert RGB to CHW format and normalize to [0, 1]
                dst[0 * 416 * 416 + y * 416 + x] = src[src_idx + 0] / 255.0f;     // R
                dst[1 * 416 * 416 + y * 416 + x] = src[src_idx + 1] / 255.0f;     // G
                dst[2 * 416 * 416 + y * 416 + x] = src[src_idx + 2] / 255.0f;     // B
            }
        }
        
        // Queue buffer again
        if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            std::cerr << "Failed to queue buffer" << std::endl;
            return false;
        }
        
        return true;
#elif defined(__APPLE__)
        // macOS camera capture would use AVFoundation
        if (initialized_) {
            // Would capture frame using AVCaptureSession
            return false;
        }
        return false;
#elif defined(_WIN32)
        // Windows camera capture would use Media Foundation
        if (initialized_) {
            // Would capture frame using IMFSourceReader
            return false;
        }
        return false;
#else
        // No camera support on this platform
        return false;
#endif
    }
    
    void release() {
#ifdef __linux__
        if (initialized_) {
            // Stop streaming
            enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctl(fd_, VIDIOC_STREAMOFF, &type);
            
            // Unmap buffer
            if (buffer_ != MAP_FAILED) {
                munmap(buffer_, buffer_length_);
            }
            
            // Close device
            if (fd_ >= 0) {
                close(fd_);
            }
            
            initialized_ = false;
        }
#endif
    }
    
private:
    int fd_;
    bool initialized_;
    int width_;
    int height_;
    void* buffer_;
    size_t buffer_length_;
};

struct EdgeConfig {
    std::string model_path;
    std::string config_path;
    utils::DeviceType device = utils::DeviceType::CPU;
    utils::OptimizationLevel opt_level = utils::OptimizationLevel::O3_AGGRESSIVE;
    int num_threads = std::thread::hardware_concurrency();
    bool enable_profiling = false;
    bool benchmark_mode = false;
    int benchmark_iterations = 100;
    int benchmark_warmup = 10;
    int metrics_port = 9090;
    int health_port = 8080;
    bool camera_enabled = false;
    int camera_device_id = 0;
};

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

EdgeConfig parse_args(int argc, char* argv[]) {
    EdgeConfig config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config.config_path = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            std::string device = argv[++i];
            if (device == "cpu") config.device = utils::DeviceType::CPU;
            else if (device == "gpu-cuda") config.device = utils::DeviceType::GPU_CUDA;
            else if (device == "gpu-opencl") config.device = utils::DeviceType::GPU_OPENCL;
            else if (device == "npu") config.device = utils::DeviceType::NPU;
            else if (device == "tpu") config.device = utils::DeviceType::TPU;
        } else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::stoi(argv[++i]);
        } else if (arg == "--optimize" && i + 1 < argc) {
            int level = std::stoi(argv[++i]);
            config.opt_level = static_cast<utils::OptimizationLevel>(level);
        } else if (arg == "--profile") {
            config.enable_profiling = true;
        } else if (arg == "--benchmark") {
            config.benchmark_mode = true;
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.benchmark_iterations = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            config.benchmark_warmup = std::stoi(argv[++i]);
        } else if (arg == "--metrics-port" && i + 1 < argc) {
            config.metrics_port = std::stoi(argv[++i]);
        } else if (arg == "--health-port" && i + 1 < argc) {
            config.health_port = std::stoi(argv[++i]);
        } else if (arg == "--camera") {
            config.camera_enabled = true;
        } else if (arg == "--camera-id" && i + 1 < argc) {
            config.camera_device_id = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "TACS Edge Runtime\n\n";
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n\n";
            std::cout << "OPTIONS:\n";
            std::cout << "  --model PATH         Path to ONNX model file\n";
            std::cout << "  --config PATH        Path to configuration file\n";
            std::cout << "  --device TYPE        Device type: cpu, gpu-cuda, gpu-opencl, npu, tpu\n";
            std::cout << "  --threads N          Number of threads (default: auto)\n";
            std::cout << "  --optimize LEVEL     Optimization level 0-3 (default: 3)\n";
            std::cout << "  --profile            Enable profiling\n";
            std::cout << "  --benchmark          Run in benchmark mode\n";
            std::cout << "  --iterations N       Benchmark iterations (default: 100)\n";
            std::cout << "  --warmup N           Benchmark warmup iterations (default: 10)\n";
            std::cout << "  --metrics-port PORT  Metrics server port (default: 9090)\n";
            std::cout << "  --health-port PORT   Health check port (default: 8080)\n";
            std::cout << "  --camera             Enable camera input\n";
            std::cout << "  --camera-id ID       Camera device ID (default: 0)\n";
            std::cout << "  --help               Show this help message\n";
            exit(0);
        }
    }
    
    if (config.model_path.empty() && config.config_path.empty()) {
        std::cerr << "Error: Either --model or --config must be specified\n";
        exit(1);
    }
    
    return config;
}

void load_config_file(EdgeConfig& config) {
    if (config.config_path.empty()) return;
    
    std::ifstream file(config.config_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file: " << config.config_path << std::endl;
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        size_t pos = line.find(':');
        if (pos == std::string::npos) continue;
        
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        
        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        if (key == "model.path" && config.model_path.empty()) {
            config.model_path = value;
        } else if (key == "runtime.device") {
            if (value == "cpu") config.device = utils::DeviceType::CPU;
            else if (value == "gpu-cuda") config.device = utils::DeviceType::GPU_CUDA;
            else if (value == "gpu-opencl") config.device = utils::DeviceType::GPU_OPENCL;
            else if (value == "npu") config.device = utils::DeviceType::NPU;
            else if (value == "tpu") config.device = utils::DeviceType::TPU;
        } else if (key == "runtime.optimization_level") {
            config.opt_level = static_cast<utils::OptimizationLevel>(std::stoi(value));
        } else if (key == "runtime.num_threads") {
            config.num_threads = std::stoi(value);
        }
    }
}

void serve_metrics(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "Failed to create metrics socket" << std::endl;
        return;
    }
    
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);
    
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Failed to bind metrics socket to port " << port << std::endl;
        close(server_fd);
        return;
    }
    
    listen(server_fd, 3);
    
    while (g_running) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(server_fd, &readfds);
        
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        
        int activity = select(server_fd + 1, &readfds, NULL, NULL, &timeout);
        
        if (activity > 0 && FD_ISSET(server_fd, &readfds)) {
            int client_fd = accept(server_fd, NULL, NULL);
            if (client_fd >= 0) {
                // Generate metrics in Prometheus format
                std::stringstream metrics;
                metrics << "# HELP tacs_inference_total Total number of inferences\n";
                metrics << "# TYPE tacs_inference_total counter\n";
                metrics << "tacs_inference_total " << g_total_inferences.load() << "\n\n";
                
                uint64_t count = g_total_inferences.load();
                double avg_latency = count > 0 ? g_total_latency_ms.load() / count : 0;
                
                metrics << "# HELP tacs_inference_latency_ms Average inference latency in milliseconds\n";
                metrics << "# TYPE tacs_inference_latency_ms gauge\n";
                metrics << "tacs_inference_latency_ms " << std::fixed << std::setprecision(2) << avg_latency << "\n\n";
                
                metrics << "# HELP tacs_inference_latency_max_ms Maximum inference latency in milliseconds\n";
                metrics << "# TYPE tacs_inference_latency_max_ms gauge\n";
                metrics << "tacs_inference_latency_max_ms " << std::fixed << std::setprecision(2) << g_max_latency_ms.load() << "\n";
                
                std::string response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n" + metrics.str();
                send(client_fd, response.c_str(), response.length(), 0);
                close(client_fd);
            }
        }
    }
    
    close(server_fd);
}

void serve_health(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "Failed to create health check socket" << std::endl;
        return;
    }
    
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);
    
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Failed to bind health check socket to port " << port << std::endl;
        close(server_fd);
        return;
    }
    
    listen(server_fd, 3);
    
    while (g_running) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(server_fd, &readfds);
        
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        
        int activity = select(server_fd + 1, &readfds, NULL, NULL, &timeout);
        
        if (activity > 0 && FD_ISSET(server_fd, &readfds)) {
            int client_fd = accept(server_fd, NULL, NULL);
            if (client_fd >= 0) {
                std::string response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nOK\n";
                send(client_fd, response.c_str(), response.length(), 0);
                close(client_fd);
            }
        }
    }
    
    close(server_fd);
}

void run_benchmark(utils::ONNXRuntime& runtime, const EdgeConfig& config) {
    std::cout << "Running benchmark..." << std::endl;
    std::cout << "Device: " << utils::HardwareDetector::get_device_name(config.device) << std::endl;
    std::cout << "Optimization Level: " << static_cast<int>(config.opt_level) << std::endl;
    std::cout << "Threads: " << config.num_threads << std::endl;
    std::cout << "Iterations: " << config.benchmark_iterations << std::endl;
    std::cout << "Warmup: " << config.benchmark_warmup << std::endl;
    std::cout << std::endl;
    
    // Create dummy input
    std::unordered_map<std::string, core::Tensor> inputs;
    inputs["input"] = core::Tensor({1, 3, 416, 416}, core::DataType::FLOAT32);
    
    // Fill with random data
    float* data = static_cast<float*>(inputs["input"].data());
    for (size_t i = 0; i < inputs["input"].size(); ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Warmup
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < config.benchmark_warmup; ++i) {
        std::unordered_map<std::string, core::Tensor> outputs;
        runtime.run(inputs, outputs);
    }
    
    // Benchmark
    std::cout << "Benchmarking..." << std::endl;
    std::vector<double> latencies;
    
    for (int i = 0; i < config.benchmark_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::unordered_map<std::string, core::Tensor> outputs;
        runtime.run(inputs, outputs);
        
        auto end = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(latency);
        
        if ((i + 1) % 10 == 0) {
            std::cout << "Progress: " << (i + 1) << "/" << config.benchmark_iterations << "\r" << std::flush;
        }
    }
    std::cout << std::endl;
    
    // Calculate statistics
    std::sort(latencies.begin(), latencies.end());
    
    double sum = 0;
    for (double l : latencies) sum += l;
    double avg = sum / latencies.size();
    
    double p50 = latencies[latencies.size() * 0.50];
    double p90 = latencies[latencies.size() * 0.90];
    double p95 = latencies[latencies.size() * 0.95];
    double p99 = latencies[latencies.size() * 0.99];
    double min = latencies.front();
    double max = latencies.back();
    
    std::cout << "\nBenchmark Results:" << std::endl;
    std::cout << "==================" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Average latency: " << avg << " ms" << std::endl;
    std::cout << "Min latency:     " << min << " ms" << std::endl;
    std::cout << "Max latency:     " << max << " ms" << std::endl;
    std::cout << "P50 latency:     " << p50 << " ms" << std::endl;
    std::cout << "P90 latency:     " << p90 << " ms" << std::endl;
    std::cout << "P95 latency:     " << p95 << " ms" << std::endl;
    std::cout << "P99 latency:     " << p99 << " ms" << std::endl;
    std::cout << "Throughput:      " << (1000.0 / avg) << " FPS" << std::endl;
    
    if (avg <= 50.0) {
        std::cout << "\n✓ PASSED: Average latency " << avg << " ms <= 50 ms target" << std::endl;
    } else {
        std::cout << "\n✗ FAILED: Average latency " << avg << " ms > 50 ms target" << std::endl;
    }
    
    if (config.enable_profiling) {
        std::cout << "\nPer-layer profiling:" << std::endl;
        auto profiling_results = runtime.get_profiling_results();
        for (const auto& [layer, time] : profiling_results) {
            std::cout << "  " << layer << ": " << time << " ms" << std::endl;
        }
    }
}

void run_inference_loop(utils::ONNXRuntime& runtime, const EdgeConfig& config) {
    std::cout << "Starting inference loop..." << std::endl;
    std::cout << "Press Ctrl+C to stop" << std::endl;
    
    // Create dummy input for continuous inference
    std::unordered_map<std::string, core::Tensor> inputs;
    inputs["input"] = core::Tensor({1, 3, 416, 416}, core::DataType::FLOAT32);
    
    // Initialize camera capture (if available)
    CameraCapture camera;
    bool use_camera = false;
    if (config.camera_enabled && camera.initialize(config.camera_device_id, 416, 416)) {
        std::cout << "Camera initialized successfully" << std::endl;
        use_camera = true;
    } else if (config.camera_enabled) {
        std::cout << "Warning: Failed to initialize camera, using dummy data" << std::endl;
    }
    
    while (g_running) {
        // Get frame from camera or use dummy data
        if (use_camera) {
            if (!camera.capture_frame(inputs["input"])) {
                std::cerr << "Failed to capture frame from camera" << std::endl;
                // Fall back to dummy data
                float* data = static_cast<float*>(inputs["input"].data());
                for (size_t i = 0; i < inputs["input"].size(); ++i) {
                    data[i] = static_cast<float>(rand()) / RAND_MAX;
                }
            }
        } else {
            // Use dummy data for testing
            float* data = static_cast<float*>(inputs["input"].data());
            for (size_t i = 0; i < inputs["input"].size(); ++i) {
                data[i] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::unordered_map<std::string, core::Tensor> outputs;
        if (runtime.run(inputs, outputs)) {
            auto end = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration<double, std::milli>(end - start).count();
            
            // Update metrics
            g_total_inferences++;
            g_total_latency_ms.store(g_total_latency_ms.load() + latency);
            
            double current_max = g_max_latency_ms.load();
            while (latency > current_max && !g_max_latency_ms.compare_exchange_weak(current_max, latency));
            
            if (g_total_inferences % 100 == 0) {
                uint64_t count = g_total_inferences.load();
                double avg_latency = g_total_latency_ms.load() / count;
                std::cout << "Inferences: " << count 
                          << ", Avg latency: " << std::fixed << std::setprecision(2) << avg_latency << " ms"
                          << ", Max latency: " << g_max_latency_ms.load() << " ms\r" << std::flush;
            }
        }
        
        // Frame rate limiting (30 FPS target)
        // In production with real camera, this would be handled by camera frame rate
        if (!use_camera) {
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    }
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    EdgeConfig config = parse_args(argc, argv);
    
    // Load configuration file
    load_config_file(config);
    
    // Set up signal handling
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Print configuration
    std::cout << "TACS Edge Runtime" << std::endl;
    std::cout << "=================" << std::endl;
    std::cout << "Model: " << config.model_path << std::endl;
    std::cout << "Device: " << utils::HardwareDetector::get_device_name(config.device) << std::endl;
    std::cout << "Threads: " << config.num_threads << std::endl;
    std::cout << "Optimization: " << static_cast<int>(config.opt_level) << std::endl;
    std::cout << std::endl;
    
    // Check device availability
    if (!utils::HardwareDetector::is_device_available(config.device)) {
        std::cerr << "Error: Selected device is not available" << std::endl;
        std::cerr << "Falling back to CPU" << std::endl;
        config.device = utils::DeviceType::CPU;
    }
    
    // Initialize runtime
    auto runtime = std::make_unique<utils::ONNXRuntime>();
    
    utils::InferenceOptions inference_opts;
    inference_opts.device = config.device;
    inference_opts.opt_level = config.opt_level;
    inference_opts.num_threads = config.num_threads;
    inference_opts.enable_profiling = config.enable_profiling;
    inference_opts.enable_memory_optimization = true;
    inference_opts.max_workspace_size = 512 * 1024 * 1024; // 512 MB
    
    std::cout << "Loading model..." << std::endl;
    if (!runtime->load_model(config.model_path, inference_opts)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    std::cout << "Model loaded successfully" << std::endl;
    std::cout << "Memory usage: " << (runtime->get_memory_usage() / (1024 * 1024)) << " MB" << std::endl;
    std::cout << std::endl;
    
    // Print model info
    std::cout << "Input tensors:" << std::endl;
    for (const auto& name : runtime->get_input_names()) {
        auto info = runtime->get_input_info(name);
        if (info) {
            std::cout << "  " << name << ": [";
            for (size_t i = 0; i < info->shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << info->shape[i];
            }
            std::cout << "]" << std::endl;
        }
    }
    
    std::cout << "Output tensors:" << std::endl;
    for (const auto& name : runtime->get_output_names()) {
        auto info = runtime->get_output_info(name);
        if (info) {
            std::cout << "  " << name << ": [";
            for (size_t i = 0; i < info->shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << info->shape[i];
            }
            std::cout << "]" << std::endl;
        }
    }
    std::cout << std::endl;
    
    if (config.benchmark_mode) {
        // Run benchmark
        run_benchmark(*runtime, config);
    } else {
        // Start monitoring servers
        std::thread metrics_thread(serve_metrics, config.metrics_port);
        std::thread health_thread(serve_health, config.health_port);
        
        std::cout << "Metrics available at: http://localhost:" << config.metrics_port << "/metrics" << std::endl;
        std::cout << "Health check at: http://localhost:" << config.health_port << "/health" << std::endl;
        std::cout << std::endl;
        
        // Run inference loop
        run_inference_loop(*runtime, config);
        
        // Wait for threads to finish
        metrics_thread.join();
        health_thread.join();
    }
    
    std::cout << "\nShutdown complete" << std::endl;
    return 0;
}