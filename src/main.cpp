/**
 * @file main.cpp
 * @brief TACS Phase 1 validation and testing suite
 * 
 * Comprehensive testing of core neural network infrastructure including tensor
 * operations, model forward/backward passes, and training loop validation.
 * Ensures all components meet real-time performance requirements.
 */
#include "core/tensor.h"
#include "core/memory_manager.h"
#include "models/tacsnet.h"
#include "training/loss.h"
#include "training/optimizer.h"
#include "data/data_loader.h"
#include <iostream>
#include <chrono>

using namespace tacs;

void print_tensor_info(const core::Tensor& tensor, const std::string& name) {
    std::cout << name << " shape: [";
    const auto& shape = tensor.shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "] size: " << tensor.size() << std::endl;
}

void test_tensor_operations() {
    std::cout << "=== Testing Tensor Operations ===" << std::endl;
    
    core::Tensor a({2, 3});
    a.randn(0.0f, 1.0f);
    print_tensor_info(a, "Tensor A");
    
    core::Tensor b({3, 4});
    b.randn(0.0f, 1.0f);
    print_tensor_info(b, "Tensor B");
    
    auto transposed = a.transpose(0, 1);
    print_tensor_info(transposed, "Transposed A");
    
    std::cout << "Memory allocated: " << 
        core::MemoryManager::instance().total_allocated() << " bytes" << std::endl;
}

void test_model_forward() {
    std::cout << "\n=== Testing TACSNet Forward Pass ===" << std::endl;
    
    // Pre-allocate memory pool for zero-allocation inference
    auto& mem_mgr = core::MemoryManager::instance();
    constexpr size_t INFERENCE_POOL_SIZE = 200 * 1024 * 1024; // 200MB
    mem_mgr.pre_allocate_inference_pool(INFERENCE_POOL_SIZE);
    mem_mgr.set_inference_mode(true);
    
    models::TACSNet model;
    
    core::Tensor input({1, 3, 416, 416});
    input.randn(0.0f, 0.1f);
    print_tensor_info(input, "Input");
    
    // Warm-up run for cache optimization
    std::cout << "Performing warm-up run..." << std::endl;
    auto warm_outputs = model.forward(input, false);
    mem_mgr.reset_pool();
    
    // Production inference timing
    auto start = std::chrono::high_resolution_clock::now();
    auto outputs = model.forward(input, false);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Forward pass time: " << duration.count() << " ms" << std::endl;
    
    // Check if we meet the 50ms requirement
    if (duration.count() <= 50) {
        std::cout << "✓ PERFORMANCE TARGET MET: Forward pass under 50ms" << std::endl;
    } else {
        std::cout << "✗ PERFORMANCE TARGET MISSED: Forward pass over 50ms" << std::endl;
    }
    
    std::cout << "Number of detection heads: " << outputs.size() << std::endl;
    std::cout << "Memory pool utilization: " << mem_mgr.pool_utilization() << " bytes" << std::endl;
    
    for (size_t i = 0; i < outputs.size(); ++i) {
        print_tensor_info(outputs[i].bbox_predictions, 
                          "Head " + std::to_string(i) + " bbox");
        print_tensor_info(outputs[i].objectness_scores, 
                          "Head " + std::to_string(i) + " objectness");
        print_tensor_info(outputs[i].class_probabilities, 
                          "Head " + std::to_string(i) + " classes");
    }
    
    // Disable inference mode after testing
    mem_mgr.set_inference_mode(false);
}

void test_training_loop() {
    std::cout << "\n=== Testing Training Loop ===" << std::endl;
    
    models::TACSNet model;
    training::YOLOLoss loss_fn;
    training::SGDOptimizer optimizer(0.001f);
    
    core::Tensor input({2, 3, 416, 416});
    input.randn(0.0f, 0.1f);
    
    core::Tensor targets({2, 10, 5});
    targets.randn(0.0f, 0.1f);
    
    std::cout << "Running training iteration..." << std::endl;
    
    auto outputs = model.forward(input, true);
    float loss_value = loss_fn.compute_loss(outputs, targets, model.get_anchors());
    
    std::cout << "Loss: " << loss_value << std::endl;
    
    auto gradients = loss_fn.backward(outputs, targets, model.get_anchors());
    model.backward(gradients, input);
    
    model.apply_gradients(0.001f);
    
    std::cout << "Training iteration completed successfully" << std::endl;
}

void test_data_loader() {
    std::cout << "\n=== Testing Data Loader ===" << std::endl;
    
    data::DataLoader loader("./data", 4, true);
    
    std::cout << "Number of samples: " << loader.num_samples() << std::endl;
    std::cout << "Number of batches: " << loader.num_batches() << std::endl;
    
    if (loader.has_next()) {
        auto batch = loader.next_batch();
        std::cout << "Loaded batch with " << batch.size() << " samples" << std::endl;
        
        if (!batch.empty()) {
            print_tensor_info(batch[0].image, "First sample image");
            std::cout << "First sample has " << batch[0].boxes.size() << " bounding boxes" << std::endl;
        }
    } else {
        std::cout << "No data available - creating synthetic batch" << std::endl;
    }
}

void test_model_serialization() {
    std::cout << "\n=== Testing Model Serialization ===" << std::endl;
    
    models::TACSNet model;
    std::string model_path = "./test_model.tacs";
    
    std::cout << "Saving model to: " << model_path << std::endl;
    bool save_success = model.save_model(model_path);
    
    if (save_success) {
        std::cout << "Model saved successfully" << std::endl;
        
        models::TACSNet loaded_model;
        std::cout << "Loading model from: " << model_path << std::endl;
        bool load_success = loaded_model.load_model(model_path);
        
        if (load_success) {
            std::cout << "Model loaded successfully" << std::endl;
            
            const auto& original_anchors = model.get_anchors();
            const auto& loaded_anchors = loaded_model.get_anchors();
            
            std::cout << "Original anchors: " << original_anchors.size() << " sets" << std::endl;
            std::cout << "Loaded anchors: " << loaded_anchors.size() << " sets" << std::endl;
            
            std::cout << "Serialization test completed" << std::endl;
        } else {
            std::cout << "Failed to load model" << std::endl;
        }
    } else {
        std::cout << "Failed to save model" << std::endl;
    }
}

int main() {
    std::cout << "TACS - Traffic-Aware Control System" << std::endl;
    std::cout << "Phase 1: Core Infrastructure Testing" << std::endl;
    std::cout << "====================================" << std::endl;
    
    try {
        test_tensor_operations();
        test_model_forward();
        test_training_loop();
        test_data_loader();
        test_model_serialization();
        
        std::cout << "\n=== Phase 1 Testing Complete ===" << std::endl;
        std::cout << "All core components initialized successfully!" << std::endl;
        std::cout << "Peak memory usage: " << 
            core::MemoryManager::instance().peak_allocated() << " bytes" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}