#include "layers/leaky_relu.h"
#include <algorithm>

namespace tacs {
namespace layers {

LeakyReLU::LeakyReLU(float negative_slope) : negative_slope_(negative_slope) {}

core::Tensor LeakyReLU::forward(const core::Tensor& input) {
    core::Tensor output = input;
    float* output_data = output.data_float();
    const size_t size = output.size();
    
    // Vectorized LeakyReLU with loop unrolling for optimal performance
    constexpr int UNROLL_FACTOR = 8;
    size_t remaining = size;
    size_t pos = 0;
    
    // Process in unrolled blocks for better vectorization
    while (remaining >= UNROLL_FACTOR) {
        // Branchless computation using min/max for better performance
        output_data[pos + 0] = std::max(output_data[pos + 0], output_data[pos + 0] * negative_slope_);
        output_data[pos + 1] = std::max(output_data[pos + 1], output_data[pos + 1] * negative_slope_);
        output_data[pos + 2] = std::max(output_data[pos + 2], output_data[pos + 2] * negative_slope_);
        output_data[pos + 3] = std::max(output_data[pos + 3], output_data[pos + 3] * negative_slope_);
        output_data[pos + 4] = std::max(output_data[pos + 4], output_data[pos + 4] * negative_slope_);
        output_data[pos + 5] = std::max(output_data[pos + 5], output_data[pos + 5] * negative_slope_);
        output_data[pos + 6] = std::max(output_data[pos + 6], output_data[pos + 6] * negative_slope_);
        output_data[pos + 7] = std::max(output_data[pos + 7], output_data[pos + 7] * negative_slope_);
        
        pos += UNROLL_FACTOR;
        remaining -= UNROLL_FACTOR;
    }
    
    // Handle remaining elements
    while (remaining > 0) {
        output_data[pos] = std::max(output_data[pos], output_data[pos] * negative_slope_);
        ++pos;
        --remaining;
    }
    
    return output;
}

core::Tensor LeakyReLU::backward(const core::Tensor& grad_output, const core::Tensor& input) {
    core::Tensor grad_input = grad_output;
    float* grad_data = grad_input.data_float();
    const float* input_data = input.data_float();
    
    for (size_t i = 0; i < grad_input.size(); ++i) {
        if (input_data[i] < 0.0f) {
            grad_data[i] *= negative_slope_;
        }
    }
    
    return grad_input;
}

}
}