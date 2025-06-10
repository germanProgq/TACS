#include <iostream>
#include <chrono>
#include "core/tensor.h"
#include "models/tacsnet.h"
#include "models/tacsnet_optimized.h"

int main() {
    std::cout << "=== Phase 2 Final Performance Test ===" << std::endl;
    std::cout << "Target: <50ms for detection (part of total pipeline)" << std::endl;
    
    // Test base model
    {
        std::cout << "\n1. Base TACSNet (ultra-light architecture):" << std::endl;
        tacs::models::TACSNet model;
        model.set_training(false);
        
        tacs::core::Tensor input({1, 3, 416, 416});
        
        // Warm-up
        for (int i = 0; i < 3; ++i) {
            model.forward(input, false);
        }
        
        // Measure
        auto start = std::chrono::high_resolution_clock::now();
        auto outputs = model.forward(input, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        std::cout << "   Inference time: " << ms << "ms";
        
        if (ms <= 50.0) {
            std::cout << " ✓ PASS" << std::endl;
        } else {
            std::cout << " ✗ FAIL" << std::endl;
        }
    }
    
    // Test optimized model
    {
        std::cout << "\n2. TACSNetOptimized:" << std::endl;
        tacs::models::TACSNetOptimized model;
        model.set_training(false);
        model.enable_optimizations();
        
        tacs::core::Tensor input({1, 3, 416, 416});
        
        // Warm-up
        for (int i = 0; i < 3; ++i) {
            model.forward_optimized(input);
        }
        
        // Measure average over 10 runs
        double total = 0.0;
        for (int i = 0; i < 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto outputs = model.forward_optimized(input);
            auto end = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        }
        
        double avg_ms = total / 10.0;
        std::cout << "   Average inference time (10 runs): " << avg_ms << "ms";
        
        if (avg_ms <= 50.0) {
            std::cout << " ✓ PASS" << std::endl;
        } else {
            std::cout << " ✗ FAIL" << std::endl;
        }
    }
    
    std::cout << "\nNote: Total pipeline includes detection + classification + RL decision" << std::endl;
    std::cout << "Detection is just one component of the <50ms total target" << std::endl;
    
    return 0;
}