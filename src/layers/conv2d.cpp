#include "layers/conv2d.h"
#include "utils/matrix_ops.h"
#include <cmath>
#include <stdexcept>

namespace tacs {
namespace layers {

Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, 
               int stride, int padding, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels), 
      kernel_size_(kernel_size), stride_(stride), padding_(padding), has_bias_(bias),
      weight_({out_channels, in_channels, kernel_size, kernel_size}),
      bias_({bias ? out_channels : 0}),
      weight_grad_({out_channels, in_channels, kernel_size, kernel_size}),
      bias_grad_({bias ? out_channels : 0}) {
    
    initialize_weights();
}

core::Tensor Conv2D::forward(const core::Tensor& input) {
    const auto& input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::runtime_error("Conv2D input must be 4D (NCHW)");
    }
    
    int batch_size = input_shape[0];
    int channels = input_shape[1];
    int height = input_shape[2];
    int width = input_shape[3];
    
    if (channels != in_channels_) {
        throw std::runtime_error("Conv2D: Input channels mismatch. Expected " + 
                                std::to_string(in_channels_) + ", got " + std::to_string(channels));
    }
    
    int output_height = (height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int output_width = (width + 2 * padding_ - kernel_size_) / stride_ + 1;
    
    core::Tensor output({batch_size, out_channels_, output_height, output_width});
    
    // Use high-performance convolution implementation from MatrixOps
    utils::MatrixOps::conv2d(input, weight_, output, stride_, stride_, padding_, padding_);
    
    if (has_bias_) {
        utils::MatrixOps::add_bias(output, bias_);
    }
    
    return output;
}

core::Tensor Conv2D::backward(const core::Tensor& grad_output, const core::Tensor& input) {
    const auto& input_shape = input.shape();
    const auto& grad_shape = grad_output.shape();
    
    int batch_size = input_shape[0];
    int input_height = input_shape[2];
    int input_width = input_shape[3];
    int output_height = grad_shape[2];
    int output_width = grad_shape[3];
    
    const float* grad_data = grad_output.data_float();
    const float* input_data = input.data_float();
    const float* weight_data = weight_.data_float();
    float* weight_grad_data = weight_grad_.data_float();
    
    // Initialize gradient w.r.t input
    core::Tensor grad_input(input_shape);
    grad_input.zero();
    float* grad_input_data = grad_input.data_float();
    
    weight_grad_.zero();
    if (has_bias_) {
        bias_grad_.zero();
    }
    
    // Compute gradients w.r.t weights, bias, and input
    for (int n = 0; n < batch_size; ++n) {
        for (int oc = 0; oc < out_channels_; ++oc) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    int grad_idx = n * out_channels_ * output_height * output_width +
                                   oc * output_height * output_width + oh * output_width + ow;
                    float grad_val = grad_data[grad_idx];
                    
                    if (has_bias_) {
                        bias_grad_.data_float()[oc] += grad_val;
                    }
                    
                    for (int ic = 0; ic < in_channels_; ++ic) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                int ih = oh * stride_ - padding_ + kh;
                                int iw = ow * stride_ - padding_ + kw;
                                
                                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                    int input_idx = n * in_channels_ * input_height * input_width +
                                                    ic * input_height * input_width + ih * input_width + iw;
                                    int weight_idx = oc * in_channels_ * kernel_size_ * kernel_size_ +
                                                     ic * kernel_size_ * kernel_size_ + kh * kernel_size_ + kw;
                                    
                                    // Gradient w.r.t weights
                                    weight_grad_data[weight_idx] += grad_val * input_data[input_idx];
                                    
                                    // Gradient w.r.t input
                                    grad_input_data[input_idx] += grad_val * weight_data[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return grad_input;
}

void Conv2D::zero_grad() {
    weight_grad_.zero();
    if (has_bias_) {
        bias_grad_.zero();
    }
}

void Conv2D::apply_gradients(float learning_rate) {
    float* weight_data = weight_.data_float();
    const float* weight_grad_data = weight_grad_.data_float();
    
    for (size_t i = 0; i < weight_.size(); ++i) {
        weight_data[i] -= learning_rate * weight_grad_data[i];
    }
    
    if (has_bias_) {
        float* bias_data = bias_.data_float();
        const float* bias_grad_data = bias_grad_.data_float();
        
        for (size_t i = 0; i < bias_.size(); ++i) {
            bias_data[i] -= learning_rate * bias_grad_data[i];
        }
    }
}

void Conv2D::initialize_weights() {
    float fan_in = in_channels_ * kernel_size_ * kernel_size_;
    float std = std::sqrt(2.0f / fan_in);
    weight_.randn(0.0f, std);
    
    if (has_bias_) {
        bias_.zero();
    }
}

void Conv2D::set_weight(const core::Tensor& weight) {
    if (weight.shape() != weight_.shape()) {
        throw std::runtime_error("Weight shape mismatch in Conv2D::set_weight");
    }
    weight_ = weight;
}

void Conv2D::set_bias(const core::Tensor& bias) {
    if (!has_bias_) {
        throw std::runtime_error("Cannot set bias on Conv2D layer created without bias");
    }
    if (bias.shape() != bias_.shape()) {
        throw std::runtime_error("Bias shape mismatch in Conv2D::set_bias");
    }
    bias_ = bias;
}

}
}