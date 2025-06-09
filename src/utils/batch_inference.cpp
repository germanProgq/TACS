#include "utils/batch_inference.h"
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <thread>
#include <atomic>

namespace tacs {
namespace utils {

BatchInference::BatchInference(std::shared_ptr<models::TACSNet> model, const BatchConfig& config)
    : model_(model), config_(config) {
    // Pre-allocate tensor pool
    for (int i = 0; i < config_.max_batch_size * 2; ++i) {
        tensor_pool_.emplace_back(std::vector<int>{1, 3, 416, 416});
    }
}

BatchInference::~BatchInference() {
    stop();
}

void BatchInference::submit(const core::Tensor& image, int request_id,
                           std::function<void(const std::vector<NMSDetection>&)> callback) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push({image, request_id, callback});
    }
    queue_cv_.notify_one();
}

std::vector<std::vector<NMSDetection>> BatchInference::process_batch(const std::vector<core::Tensor>& images) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Stack images into batch tensor
    core::Tensor batch_input = stack_images(images);
    
    // Run inference
    model_->set_training(false);
    auto outputs = model_->forward(batch_input);
    
    // Split outputs and apply NMS
    std::vector<std::vector<NMSDetection>> results;
    results.reserve(images.size());
    
    std::vector<std::vector<models::DetectionOutput>> individual_outputs;
    split_outputs(outputs, individual_outputs);
    
    for (const auto& ind_output : individual_outputs) {
        auto detections = nms_.apply(ind_output, model_->get_anchors(), 416, 416);
        results.push_back(detections);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    float latency_ms = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Update performance metrics
    avg_latency_ms_ = 0.9f * avg_latency_ms_ + 0.1f * latency_ms;
    throughput_fps_ = 1000.0f * images.size() / latency_ms;
    
    return results;
}

void BatchInference::start() {
    running_ = true;
    
    for (int i = 0; i < config_.num_threads; ++i) {
        worker_threads_.emplace_back(&BatchInference::worker_loop, this);
    }
}

void BatchInference::stop() {
    running_ = false;
    queue_cv_.notify_all();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

void BatchInference::worker_loop() {
    while (running_) {
        auto batch = collect_batch();
        if (!batch.empty()) {
            process_batch_internal(batch);
        }
    }
}

std::vector<InferenceRequest> BatchInference::collect_batch() {
    std::vector<InferenceRequest> batch;
    
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    // Wait for requests with timeout
    queue_cv_.wait_for(lock, std::chrono::milliseconds(config_.timeout_ms),
                       [this] { return !request_queue_.empty() || !running_; });
    
    if (!running_) return batch;
    
    // Collect up to max_batch_size requests
    while (!request_queue_.empty() && batch.size() < config_.max_batch_size) {
        batch.push_back(std::move(request_queue_.front()));
        request_queue_.pop();
    }
    
    return batch;
}

void BatchInference::process_batch_internal(const std::vector<InferenceRequest>& batch) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Extract images from requests
    std::vector<core::Tensor> images;
    images.reserve(batch.size());
    for (const auto& req : batch) {
        images.push_back(req.image);
    }
    
    // Process batch
    auto results = process_batch(images);
    
    // Invoke callbacks
    for (size_t i = 0; i < batch.size(); ++i) {
        if (batch[i].callback) {
            batch[i].callback(results[i]);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    float latency_ms = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Update metrics
    avg_latency_ms_ = 0.9f * avg_latency_ms_ + 0.1f * latency_ms;
    throughput_fps_ = 1000.0f * batch.size() / latency_ms;
}

core::Tensor BatchInference::stack_images(const std::vector<core::Tensor>& images) {
    if (images.empty()) {
        throw std::runtime_error("Cannot stack empty image vector");
    }
    
    const auto& first_shape = images[0].shape();
    int batch_size = images.size();
    int channels = first_shape[0];
    int height = first_shape[1];
    int width = first_shape[2];
    
    core::Tensor batch({batch_size, channels, height, width});
    float* batch_data = batch.data_float();
    
    // Optimized memory copy with parallel processing
    size_t image_size = channels * height * width;
    
    #pragma omp parallel for if(batch_size > 4)
    for (int i = 0; i < batch_size; ++i) {
        const float* src = images[i].data_float();
        float* dst = batch_data + i * image_size;
        std::memcpy(dst, src, image_size * sizeof(float));
    }
    
    return batch;
}

void BatchInference::split_outputs(const std::vector<models::DetectionOutput>& batch_outputs,
                                  std::vector<std::vector<models::DetectionOutput>>& individual_outputs) {
    if (batch_outputs.empty()) return;
    
    const auto& first_bbox_shape = batch_outputs[0].bbox_predictions.shape();
    int batch_size = first_bbox_shape[0];
    int num_scales = batch_outputs.size();
    
    individual_outputs.resize(batch_size);
    
    for (int b = 0; b < batch_size; ++b) {
        individual_outputs[b].reserve(num_scales);
        
        for (int s = 0; s < num_scales; ++s) {
            const auto& scale_output = batch_outputs[s];
            const auto& bbox_shape = scale_output.bbox_predictions.shape();
            const auto& obj_shape = scale_output.objectness_scores.shape();
            const auto& cls_shape = scale_output.class_probabilities.shape();
            
            // Create shapes for individual detection
            std::vector<int> ind_bbox_shape = {1, bbox_shape[1], bbox_shape[2], bbox_shape[3], bbox_shape[4]};
            std::vector<int> ind_obj_shape = {1, obj_shape[1], obj_shape[2], obj_shape[3], obj_shape[4]};
            std::vector<int> ind_cls_shape = {1, cls_shape[1], cls_shape[2], cls_shape[3], cls_shape[4]};
            
            models::DetectionOutput ind_output(ind_bbox_shape, ind_obj_shape, ind_cls_shape);
            
            // Copy data for this batch element
            const float* bbox_src = scale_output.bbox_predictions.data_float();
            const float* obj_src = scale_output.objectness_scores.data_float();
            const float* cls_src = scale_output.class_probabilities.data_float();
            
            float* bbox_dst = ind_output.bbox_predictions.data_float();
            float* obj_dst = ind_output.objectness_scores.data_float();
            float* cls_dst = ind_output.class_probabilities.data_float();
            
            size_t bbox_stride = ind_output.bbox_predictions.size();
            size_t obj_stride = ind_output.objectness_scores.size();
            size_t cls_stride = ind_output.class_probabilities.size();
            
            std::memcpy(bbox_dst, bbox_src + b * bbox_stride, bbox_stride * sizeof(float));
            std::memcpy(obj_dst, obj_src + b * obj_stride, obj_stride * sizeof(float));
            std::memcpy(cls_dst, cls_src + b * cls_stride, cls_stride * sizeof(float));
            
            individual_outputs[b].push_back(std::move(ind_output));
        }
    }
}

core::Tensor BatchInference::get_tensor_from_pool(const std::vector<int>& shape) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    for (auto it = tensor_pool_.begin(); it != tensor_pool_.end(); ++it) {
        if (it->shape() == shape) {
            core::Tensor tensor = std::move(*it);
            tensor_pool_.erase(it);
            return tensor;
        }
    }
    
    // If no suitable tensor in pool, create new one
    return core::Tensor(shape);
}

void BatchInference::return_tensor_to_pool(core::Tensor&& tensor) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Only keep reasonable number of tensors in pool
    if (tensor_pool_.size() < config_.max_batch_size * 4) {
        tensor_pool_.push_back(std::move(tensor));
    }
}

// Dynamic Batch Scheduler Implementation

DynamicBatchScheduler::DynamicBatchScheduler(int max_batch_size, int max_latency_ms)
    : max_batch_size_(max_batch_size), max_latency_ms_(max_latency_ms) {
    batch_latencies_.resize(max_batch_size + 1, 0.0f);
}

int DynamicBatchScheduler::get_optimal_batch_size(int queue_size, float current_latency_ms) {
    if (queue_size == 0) return 1;
    
    // Find largest batch size that meets latency constraint
    int optimal_size = 1;
    
    for (int size = std::min(queue_size, max_batch_size_); size > 0; --size) {
        float predicted_latency = predict_latency(size);
        if (predicted_latency < max_latency_ms_) {
            optimal_size = size;
            break;
        }
    }
    
    return optimal_size;
}

void DynamicBatchScheduler::update_stats(int batch_size, float latency_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (batch_size > 0 && batch_size <= max_batch_size_) {
        // Exponential moving average update
        batch_latencies_[batch_size] = (1.0f - alpha_) * batch_latencies_[batch_size] + 
                                       alpha_ * latency_ms;
    }
}

float DynamicBatchScheduler::predict_latency(int batch_size) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (batch_latencies_[batch_size] > 0) {
        return batch_latencies_[batch_size];
    }
    
    // Linear interpolation for unseen batch sizes
    float base_latency = 10.0f;  // Base latency in ms
    float per_sample_latency = 5.0f;  // Additional latency per sample
    
    return base_latency + per_sample_latency * batch_size;
}

}
}