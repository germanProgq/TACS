/**
 * @file batch_norm.h
 * @brief Batch normalization layer for training stabilization
 * 
 * Implements batch normalization with running statistics for inference mode.
 * Essential for training stability and convergence of deep neural networks.
 * Supports both training and evaluation modes with proper momentum updates.
 */
#pragma once

#include "core/tensor.h"

namespace tacs {
namespace layers {

class BatchNorm2D {
public:
    explicit BatchNorm2D(int num_features, float eps = 1e-5f, float momentum = 0.1f);
    ~BatchNorm2D() = default;

    core::Tensor forward(const core::Tensor& input, bool training = true);
    core::Tensor backward(const core::Tensor& grad_output, const core::Tensor& input, 
                  const core::Tensor& normalized);
    
    const core::Tensor& weight() const { return weight_; }
    const core::Tensor& bias() const { return bias_; }
    const core::Tensor& running_mean() const { return running_mean_; }
    const core::Tensor& running_var() const { return running_var_; }
    
    void zero_grad();
    void apply_gradients(float learning_rate);
    void set_training(bool training) { training_ = training; }
    
    // Setter methods for model serialization
    void set_weight(const core::Tensor& weight);
    void set_bias(const core::Tensor& bias);
    void set_running_mean(const core::Tensor& running_mean);
    void set_running_var(const core::Tensor& running_var);

private:
    int num_features_;
    float eps_;
    float momentum_;
    bool training_;
    
    core::Tensor weight_;
    core::Tensor bias_;
    core::Tensor running_mean_;
    core::Tensor running_var_;
    core::Tensor weight_grad_;
    core::Tensor bias_grad_;
    
    void initialize_parameters();
};

}
}