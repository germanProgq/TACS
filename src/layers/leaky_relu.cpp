#include "layers/leaky_relu.h"
#include <algorithm>

#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace tacs {
namespace layers {

LeakyReLU::LeakyReLU(float negative_slope) : negative_slope_(negative_slope) {}

core::Tensor LeakyReLU::forward(const core::Tensor& input) {
    core::Tensor output = input;
    float* output_data = output.data_float();
    const size_t size = output.size();
    
#ifdef __AVX2__
    // SIMD-optimized LeakyReLU with AVX2
    constexpr int SIMD_WIDTH = 8;
    __m256 negative_slope_vec = _mm256_set1_ps(negative_slope_);
    __m256 zero_vec = _mm256_setzero_ps();
    
    size_t remaining = size;
    size_t pos = 0;
    
    // Process in SIMD blocks
    while (remaining >= SIMD_WIDTH) {
        __m256 x = _mm256_loadu_ps(&output_data[pos]);
        __m256 x_neg = _mm256_mul_ps(x, negative_slope_vec);
        __m256 mask = _mm256_cmp_ps(x, zero_vec, _CMP_GT_OQ);
        __m256 result = _mm256_blendv_ps(x_neg, x, mask);
        _mm256_storeu_ps(&output_data[pos], result);
        
        pos += SIMD_WIDTH;
        remaining -= SIMD_WIDTH;
    }
    
    // Handle remaining elements
    while (remaining > 0) {
        output_data[pos] = output_data[pos] > 0 ? output_data[pos] : output_data[pos] * negative_slope_;
        ++pos;
        --remaining;
    }
#elif defined(__ARM_NEON)
    // SIMD-optimized LeakyReLU with NEON
    constexpr int SIMD_WIDTH = 4;
    float32x4_t negative_slope_vec = vdupq_n_f32(negative_slope_);
    float32x4_t zero_vec = vdupq_n_f32(0.0f);
    
    size_t remaining = size;
    size_t pos = 0;
    
    // Process in SIMD blocks
    while (remaining >= SIMD_WIDTH) {
        float32x4_t x = vld1q_f32(&output_data[pos]);
        float32x4_t x_neg = vmulq_f32(x, negative_slope_vec);
        uint32x4_t mask = vcgtq_f32(x, zero_vec);
        float32x4_t result = vbslq_f32(mask, x, x_neg);
        vst1q_f32(&output_data[pos], result);
        
        pos += SIMD_WIDTH;
        remaining -= SIMD_WIDTH;
    }
    
    // Handle remaining elements
    while (remaining > 0) {
        output_data[pos] = output_data[pos] > 0 ? output_data[pos] : output_data[pos] * negative_slope_;
        ++pos;
        --remaining;
    }
#else
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
#endif
    
    return output;
}

core::Tensor LeakyReLU::backward(const core::Tensor& grad_output, const core::Tensor& input) {
    core::Tensor grad_input = grad_output;
    float* grad_data = grad_input.data_float();
    const float* input_data = input.data_float();
    const size_t size = grad_input.size();
    
#ifdef __AVX2__
    // SIMD-optimized backward pass with AVX2
    constexpr int SIMD_WIDTH = 8;
    __m256 negative_slope_vec = _mm256_set1_ps(negative_slope_);
    __m256 one_vec = _mm256_set1_ps(1.0f);
    __m256 zero_vec = _mm256_setzero_ps();
    
    size_t remaining = size;
    size_t pos = 0;
    
    // Process in SIMD blocks
    while (remaining >= SIMD_WIDTH) {
        __m256 input_vec = _mm256_loadu_ps(&input_data[pos]);
        __m256 grad_vec = _mm256_loadu_ps(&grad_data[pos]);
        __m256 mask = _mm256_cmp_ps(input_vec, zero_vec, _CMP_GE_OQ);
        __m256 multiplier = _mm256_blendv_ps(negative_slope_vec, one_vec, mask);
        __m256 result = _mm256_mul_ps(grad_vec, multiplier);
        _mm256_storeu_ps(&grad_data[pos], result);
        
        pos += SIMD_WIDTH;
        remaining -= SIMD_WIDTH;
    }
    
    // Handle remaining elements
    while (remaining > 0) {
        if (input_data[pos] < 0.0f) {
            grad_data[pos] *= negative_slope_;
        }
        ++pos;
        --remaining;
    }
#elif defined(__ARM_NEON)
    // SIMD-optimized backward pass with NEON
    constexpr int SIMD_WIDTH = 4;
    float32x4_t negative_slope_vec = vdupq_n_f32(negative_slope_);
    float32x4_t one_vec = vdupq_n_f32(1.0f);
    float32x4_t zero_vec = vdupq_n_f32(0.0f);
    
    size_t remaining = size;
    size_t pos = 0;
    
    // Process in SIMD blocks
    while (remaining >= SIMD_WIDTH) {
        float32x4_t input_vec = vld1q_f32(&input_data[pos]);
        float32x4_t grad_vec = vld1q_f32(&grad_data[pos]);
        uint32x4_t mask = vcgeq_f32(input_vec, zero_vec);
        float32x4_t multiplier = vbslq_f32(mask, one_vec, negative_slope_vec);
        float32x4_t result = vmulq_f32(grad_vec, multiplier);
        vst1q_f32(&grad_data[pos], result);
        
        pos += SIMD_WIDTH;
        remaining -= SIMD_WIDTH;
    }
    
    // Handle remaining elements
    while (remaining > 0) {
        if (input_data[pos] < 0.0f) {
            grad_data[pos] *= negative_slope_;
        }
        ++pos;
        --remaining;
    }
#else
    // Vectorized backward pass with loop unrolling
    constexpr int UNROLL_FACTOR = 8;
    size_t remaining = size;
    size_t pos = 0;
    
    // Process in unrolled blocks for better vectorization
    while (remaining >= UNROLL_FACTOR) {
        // Branchless computation using conditional multiplication
        grad_data[pos + 0] *= (input_data[pos + 0] >= 0.0f) ? 1.0f : negative_slope_;
        grad_data[pos + 1] *= (input_data[pos + 1] >= 0.0f) ? 1.0f : negative_slope_;
        grad_data[pos + 2] *= (input_data[pos + 2] >= 0.0f) ? 1.0f : negative_slope_;
        grad_data[pos + 3] *= (input_data[pos + 3] >= 0.0f) ? 1.0f : negative_slope_;
        grad_data[pos + 4] *= (input_data[pos + 4] >= 0.0f) ? 1.0f : negative_slope_;
        grad_data[pos + 5] *= (input_data[pos + 5] >= 0.0f) ? 1.0f : negative_slope_;
        grad_data[pos + 6] *= (input_data[pos + 6] >= 0.0f) ? 1.0f : negative_slope_;
        grad_data[pos + 7] *= (input_data[pos + 7] >= 0.0f) ? 1.0f : negative_slope_;
        
        pos += UNROLL_FACTOR;
        remaining -= UNROLL_FACTOR;
    }
    
    // Handle remaining elements
    while (remaining > 0) {
        grad_data[pos] *= (input_data[pos] >= 0.0f) ? 1.0f : negative_slope_;
        ++pos;
        --remaining;
    }
#endif
    
    return grad_input;
}

}
}