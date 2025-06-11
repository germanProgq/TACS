/**
 * @file depthwise_conv2d.h
 * @brief Depth-wise separable convolution for ultra-fast inference
 * 
 * Implements depth-wise convolution followed by point-wise convolution
 * to achieve 8-9x computational reduction compared to standard convolution
 * while maintaining similar representational capacity.
 */
#pragma once

#include "core/tensor.h"
#include <vector>
#include <memory>

namespace tacs {
namespace layers {

class DepthwiseConv2D {
public:
    DepthwiseConv2D(int in_channels, int out_channels, int kernel_size, 
                    int stride = 1, int padding = 0, float multiplier = 1.0f);
    
    core::Tensor forward(const core::Tensor& input);
    void backward(const core::Tensor& grad_output, const core::Tensor& input);
    
    void zero_grad();
    void apply_gradients(float learning_rate);
    
    const core::Tensor& depthwise_weight() const { return depthwise_weight_; }
    const core::Tensor& pointwise_weight() const { return pointwise_weight_; }
    const core::Tensor& bias() const { return bias_; }
    
    // Optimized inference path with fused operations
    core::Tensor forward_optimized(const core::Tensor& input);

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    float multiplier_;
    
    // Depthwise weights: [in_channels, 1, kernel_size, kernel_size]
    core::Tensor depthwise_weight_;
    // Pointwise weights: [out_channels, in_channels, 1, 1]
    core::Tensor pointwise_weight_;
    core::Tensor bias_;
    
    // Gradients
    core::Tensor depthwise_grad_;
    core::Tensor pointwise_grad_;
    core::Tensor bias_grad_;
    
    // Intermediate tensor for depthwise output
    core::Tensor depthwise_output_;
    
    void initialize_weights();
    
    // Optimized kernels
    void depthwise_conv_simd(const float* input, float* output,
                            const float* weights, int batch, int channels,
                            int in_h, int in_w, int out_h, int out_w);
    
    void pointwise_conv_simd(const float* input, float* output,
                            const float* weights, const float* bias,
                            int batch, int in_channels, int out_channels,
                            int h, int w);
};

}
}