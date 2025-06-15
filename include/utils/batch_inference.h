/**
 * @file batch_inference.h
 * @brief Optimized batch processing for multi-scale object detection
 * 
 * Implements efficient batch inference with memory pooling and parallel processing
 * to maximize GPU/CPU utilization and achieve sub-50ms latency requirement.
 * Supports dynamic batching and zero-copy tensor operations.
 */
#pragma once

#include "core/tensor.h"
#include "models/tacsnet.h"
#include "utils/nms.h"
#include <vector>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>

namespace tacs {
namespace utils {

struct InferenceRequest {
    core::Tensor image;
    int request_id;
    std::function<void(const std::vector<NMSDetection>&)> callback;
};

struct BatchConfig {
    int max_batch_size = 8;
    int timeout_ms = 10;
    bool use_fp16 = false;
    bool use_int8 = false;
    int num_threads = 4;
};

class BatchInference {
public:
    explicit BatchInference(std::shared_ptr<models::TACSNet> model, 
                           const BatchConfig& config = BatchConfig{});
    ~BatchInference();
    
    // Submit single image for inference
    void submit(const core::Tensor& image, int request_id,
                std::function<void(const std::vector<NMSDetection>&)> callback);
    
    // Process batch of images synchronously
    std::vector<std::vector<NMSDetection>> process_batch(const std::vector<core::Tensor>& images);
    
    // Start/stop background processing thread
    void start();
    void stop();
    
    // Performance metrics
    float get_average_latency() const { return avg_latency_ms_; }
    float get_throughput() const { return throughput_fps_; }

private:
    std::shared_ptr<models::TACSNet> model_;
    BatchConfig config_;
    NonMaxSuppression nms_;
    
    // Request queue
    std::queue<InferenceRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Worker thread
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{false};
    
    // Memory pool for zero-copy operations
    std::vector<core::Tensor> tensor_pool_;
    std::mutex pool_mutex_;
    
    // Performance tracking
    std::atomic<float> avg_latency_ms_{0.0f};
    std::atomic<float> throughput_fps_{0.0f};
    
    void worker_loop();
    std::vector<InferenceRequest> collect_batch();
    void process_batch_internal(const std::vector<InferenceRequest>& batch);
    core::Tensor get_tensor_from_pool(const std::vector<int>& shape);
    void return_tensor_to_pool(core::Tensor&& tensor);
    
    // Optimized batch operations
    core::Tensor stack_images(const std::vector<core::Tensor>& images);
    void split_outputs(const std::vector<models::DetectionOutput>& batch_outputs,
                      std::vector<std::vector<models::DetectionOutput>>& individual_outputs);
    
    // Production-ready quantization helpers
    core::Tensor quantizeToINT8(const core::Tensor& input);
    core::Tensor dequantizeFromINT8(const core::Tensor& input);
    core::Tensor quantizeToFP16(const core::Tensor& input);
};

// Dynamic batching scheduler for optimal throughput
class DynamicBatchScheduler {
public:
    explicit DynamicBatchScheduler(int max_batch_size = 8, 
                                  int max_latency_ms = 50);
    
    // Adaptive batch size based on current load
    int get_optimal_batch_size(int queue_size, float current_latency_ms);
    
    // Update performance statistics
    void update_stats(int batch_size, float latency_ms);

private:
    int max_batch_size_;
    int max_latency_ms_;
    
    // Adaptive parameters
    float alpha_ = 0.1f;  // Learning rate
    std::vector<float> batch_latencies_;
    std::mutex stats_mutex_;
    
    float predict_latency(int batch_size);
};

}
}