/**
 * @file conv2d.h
 * @brief 2D Convolutional layer with manual backpropagation implementation
 * 
 * Provides convolution operations with configurable parameters including
 * stride, padding, and bias terms. Optimized for real-time inference with
 * efficient memory layout and cache-friendly access patterns.
 */
#pragma once

#include "core/tensor.h"

namespace tacs {
namespace layers {

class Conv2D {
public:
    Conv2D(int in_channels, int out_channels, int kernel_size, 
           int stride = 1, int padding = 0, bool bias = true);
    ~Conv2D() = default;

    core::Tensor forward(const core::Tensor& input);
    void backward(const core::Tensor& grad_output, const core::Tensor& input);
    
    const core::Tensor& weight() const { return weight_; }
    const core::Tensor& bias() const { return bias_; }
    const core::Tensor& weight_grad() const { return weight_grad_; }
    const core::Tensor& bias_grad() const { return bias_grad_; }
    
    void zero_grad();
    void apply_gradients(float learning_rate);
    
    // Setter methods for model serialization
    void set_weight(const core::Tensor& weight);
    void set_bias(const core::Tensor& bias);

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    bool has_bias_;
    
    core::Tensor weight_;
    core::Tensor bias_;
    core::Tensor weight_grad_;
    core::Tensor bias_grad_;
    
    void initialize_weights();
};

}
}