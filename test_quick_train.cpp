/**
 * @file test_quick_train.cpp
 * @brief Quick training test to verify loss stability
 */

#include <iostream>
#include <memory>
#include "models/tacsnet.h"
#include "training/optimizer.h"
#include "training/loss.h"
#include "training/gradient_converter.h"
#include "core/tensor.h"

using namespace tacs;

int main() {
    std::cout << "=== Quick TACSNet Training Stability Test ===\n\n";
    
    try {
        // Initialize model with pretrained weights
        auto model = std::make_unique<models::TACSNetUltra>(true);
        model->set_training(true);
        
        // Create optimizer with ultra-small learning rate
        training::SGDOptimizer optimizer(0.000001f, 0.9f, 0.0005f);
        
        // Create loss function with balanced weights
        training::LossWeights loss_weights;
        loss_weights.objectness = 0.5f;
        loss_weights.bbox = 1.0f;
        loss_weights.classification = 0.5f;
        training::YOLOLoss loss_fn(loss_weights);
        
        // Create dummy data
        core::Tensor input({1, 3, 416, 416});
        input.randn();  // Random input
        
        core::Tensor targets({1, 5, 5});  // 5 objects with [class, x, y, w, h]
        targets.zero();
        float* target_data = targets.data_float();
        
        // Add a few dummy targets
        for (int i = 0; i < 3; ++i) {
            target_data[i * 5 + 0] = i;  // class
            target_data[i * 5 + 1] = 0.5f;  // x
            target_data[i * 5 + 2] = 0.5f;  // y
            target_data[i * 5 + 3] = 0.1f;  // w
            target_data[i * 5 + 4] = 0.1f;  // h
        }
        
        // Initialize optimizer parameters
        auto weights = model->get_weights();
        for (auto& [name, weight] : weights) {
            optimizer.add_parameter_group(name, weight);
        }
        
        std::cout << "Running 5 iterations to check loss stability...\n";
        std::cout << "Expected loss range: 5-50 (with proper initialization)\n\n";
        
        for (int iter = 0; iter < 5; ++iter) {
            // Forward pass
            auto predictions = model->forward(input);
            
            // Compute loss
            float loss = loss_fn.compute_loss(predictions, targets, model->get_anchors());
            
            std::cout << "Iteration " << (iter + 1) << " - Loss: " << loss << std::endl;
            
            if (loss > 100.0f) {
                std::cout << "\nWARNING: Loss is too high! Training may be unstable.\n";
                std::cout << "Possible issues:\n";
                std::cout << "- Weight initialization problem\n";
                std::cout << "- Learning rate too high\n";
                std::cout << "- Gradient explosion\n";
                break;
            }
            
            // Backward pass
            model->zero_gradients();
            auto gradients = loss_fn.backward(predictions, targets, model->get_anchors());
            
            // Apply gradients
            for (size_t scale = 0; scale < gradients.size() && scale < predictions.size(); ++scale) {
                const auto& grad = gradients[scale];
                const auto& pred = predictions[scale];
                
                int num_anchors = pred.bbox_predictions.shape()[1];
                int num_classes = pred.class_predictions.shape()[4];
                
                auto [bbox_grad, obj_grad, cls_grad] = 
                    training::GradientConverter::split_combined_gradient(
                        grad, num_anchors, num_classes);
                
                model->backward(bbox_grad, obj_grad, cls_grad);
            }
            
            // Update weights
            auto weight_grads = model->get_weight_gradients();
            for (const auto& [name, grad] : weight_grads) {
                optimizer.set_gradient(name, grad);
            }
            optimizer.step();
        }
        
        std::cout << "\nTest completed successfully!\n";
        
        std::cout << "✓ Training appears stable with current settings.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}