// Conv2D FP16 - Half-precision convolution layer for TACS
// Production-ready FP16 convolution operations

#pragma once

#include "layers/conv2d.h"
#include "utils/fp16_ops.h"
#include <memory>

namespace tacs {
namespace layers {

// FP16-optimized Conv2D layer
class Conv2DFP16 : public Conv2D {
public:
    Conv2DFP16(int in_channels, int out_channels, int kernel_size, 
              int stride = 1, int padding = 0)
        : Conv2D(in_channels, out_channels, kernel_size, stride, padding),
          use_fp16_(true) {
        // Pre-convert weights to FP16
        convertWeightsToFP16();
    }
    
    // Override forward to use FP16 operations
    core::Tensor forward(const core::Tensor& input) override {
        if (!use_fp16_) {
            return Conv2D::forward(input);
        }
        
        // Convert input to FP16
        auto input_fp16 = utils::FP16Ops::tensor_to_fp16(input);
        
        // Prepare output tensor
        const auto& input_shape = input.shape();
        int batch = input_shape[0];
        int height = input_shape[2];
        int width = input_shape[3];
        
        int out_h = (height + 2 * padding_ - kernel_size_) / stride_ + 1;
        int out_w = (width + 2 * padding_ - kernel_size_) / stride_ + 1;
        
        core::Tensor output({batch, out_channels_, out_h, out_w});
        
        // Perform FP16 convolution
        utils::FP16Ops::conv2d_fp16(
            input_fp16.data(), weight_fp16_.data(),
            output.data_float(), bias_fp16_.data(),
            batch, in_channels_, out_channels_,
            height, width,
            kernel_size_, kernel_size_,
            stride_, stride_,
            padding_, padding_
        );
        
        return output;
    }
    
    // Enable/disable FP16 mode
    void setFP16Mode(bool enable) { 
        use_fp16_ = enable; 
        if (enable && weight_fp16_.empty()) {
            convertWeightsToFP16();
        }
    }
    
private:
    bool use_fp16_;
    std::vector<utils::fp16_t> weight_fp16_;
    std::vector<utils::fp16_t> bias_fp16_;
    
    void convertWeightsToFP16() {
        weight_fp16_ = utils::FP16Ops::tensor_to_fp16(weight_);
        bias_fp16_ = utils::FP16Ops::tensor_to_fp16(bias_);
    }
};

// Factory function to create FP16-enabled Conv2D layers
inline std::unique_ptr<Conv2D> createConv2D(int in_channels, int out_channels, 
                                           int kernel_size, int stride = 1, 
                                           int padding = 0, bool use_fp16 = true) {
    if (use_fp16) {
        return std::make_unique<Conv2DFP16>(in_channels, out_channels, 
                                           kernel_size, stride, padding);
    } else {
        return std::make_unique<Conv2D>(in_channels, out_channels, 
                                       kernel_size, stride, padding);
    }
}

} // namespace layers
} // namespace tacs