#include "layers/batch_norm.h"
#include <cmath>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace tacs {
namespace layers {

BatchNorm2D::BatchNorm2D(int num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum), training_(true),
      weight_({num_features}), bias_({num_features}), 
      running_mean_({num_features}), running_var_({num_features}),
      weight_grad_({num_features}), bias_grad_({num_features}) {
    initialize_parameters();
}

core::Tensor BatchNorm2D::forward(const core::Tensor& input, bool training) {
    const auto& input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::runtime_error("BatchNorm2D input must be 4D (NCHW)");
    }
    
    int batch_size = input_shape[0];
    int channels = input_shape[1];
    int height = input_shape[2];
    int width = input_shape[3];
    
    if (channels != num_features_) {
        throw std::runtime_error("Input channels mismatch");
    }
    
    core::Tensor output = input;
    const float* input_data = input.data_float();
    float* output_data = output.data_float();
    
    float* weight_data = weight_.data_float();
    float* bias_data = bias_.data_float();
    float* running_mean_data = running_mean_.data_float();
    float* running_var_data = running_var_.data_float();
    
    if (training) {
        core::Tensor batch_mean({channels});
        core::Tensor batch_var({channels});
        float* batch_mean_data = batch_mean.data_float();
        float* batch_var_data = batch_var.data_float();
        
        int spatial_size = height * width;
        int total_elements = batch_size * spatial_size;
        
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int n = 0; n < batch_size; ++n) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = n * channels * height * width + c * height * width + h * width + w;
                        sum += input_data[idx];
                    }
                }
            }
            batch_mean_data[c] = sum / total_elements;
        }
        
        for (int c = 0; c < channels; ++c) {
            float var_sum = 0.0f;
            float mean_val = batch_mean_data[c];
            
            for (int n = 0; n < batch_size; ++n) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = n * channels * height * width + c * height * width + h * width + w;
                        float diff = input_data[idx] - mean_val;
                        var_sum += diff * diff;
                    }
                }
            }
            batch_var_data[c] = var_sum / total_elements;
        }
        
        for (int c = 0; c < channels; ++c) {
            running_mean_data[c] = (1.0f - momentum_) * running_mean_data[c] + 
                                   momentum_ * batch_mean_data[c];
            running_var_data[c] = (1.0f - momentum_) * running_var_data[c] + 
                                  momentum_ * batch_var_data[c];
        }
        
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < channels; ++c) {
                float mean_val = batch_mean_data[c];
                float var_val = batch_var_data[c];
                float inv_std = 1.0f / std::sqrt(var_val + eps_);
                
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = n * channels * height * width + c * height * width + h * width + w;
                        float normalized = (input_data[idx] - mean_val) * inv_std;
                        output_data[idx] = weight_data[c] * normalized + bias_data[c];
                    }
                }
            }
        }
    } else {
        // Production-optimized inference path with SIMD vectorization
#ifdef __AVX2__
        constexpr int SIMD_WIDTH = 8;
        const int spatial_size = height * width;
        const int channel_stride = spatial_size;
        
        for (int n = 0; n < batch_size; ++n) {
            const float* input_batch = input_data + n * channels * spatial_size;
            float* output_batch = output_data + n * channels * spatial_size;
            
            for (int c = 0; c < channels; ++c) {
                const float mean_val = running_mean_data[c];
                const float var_val = running_var_data[c];
                const float inv_std = 1.0f / std::sqrt(var_val + eps_);
                const float scale = weight_data[c] * inv_std;
                const float shift = bias_data[c] - mean_val * scale;
                
                const float* input_channel = input_batch + c * channel_stride;
                float* output_channel = output_batch + c * channel_stride;
                
                // SIMD vectorized computation
                __m256 scale_vec = _mm256_set1_ps(scale);
                __m256 shift_vec = _mm256_set1_ps(shift);
                
                int remaining = spatial_size;
                int pos = 0;
                
                // Process in SIMD blocks
                while (remaining >= SIMD_WIDTH) {
                    __m256 input_vec = _mm256_loadu_ps(&input_channel[pos]);
                    __m256 result = _mm256_fmadd_ps(input_vec, scale_vec, shift_vec);
                    _mm256_storeu_ps(&output_channel[pos], result);
                    
                    pos += SIMD_WIDTH;
                    remaining -= SIMD_WIDTH;
                }
                
                // Handle remaining elements
                while (remaining > 0) {
                    output_channel[pos] = input_channel[pos] * scale + shift;
                    ++pos;
                    --remaining;
                }
            }
        }
#elif defined(__ARM_NEON)
        constexpr int SIMD_WIDTH = 4;
        const int spatial_size = height * width;
        const int channel_stride = spatial_size;
        
        for (int n = 0; n < batch_size; ++n) {
            const float* input_batch = input_data + n * channels * spatial_size;
            float* output_batch = output_data + n * channels * spatial_size;
            
            for (int c = 0; c < channels; ++c) {
                const float mean_val = running_mean_data[c];
                const float var_val = running_var_data[c];
                const float inv_std = 1.0f / std::sqrt(var_val + eps_);
                const float scale = weight_data[c] * inv_std;
                const float shift = bias_data[c] - mean_val * scale;
                
                const float* input_channel = input_batch + c * channel_stride;
                float* output_channel = output_batch + c * channel_stride;
                
                // NEON vectorized computation
                float32x4_t scale_vec = vdupq_n_f32(scale);
                float32x4_t shift_vec = vdupq_n_f32(shift);
                
                int remaining = spatial_size;
                int pos = 0;
                
                // Process in SIMD blocks
                while (remaining >= SIMD_WIDTH) {
                    float32x4_t input_vec = vld1q_f32(&input_channel[pos]);
                    float32x4_t result = vmlaq_f32(shift_vec, input_vec, scale_vec);
                    vst1q_f32(&output_channel[pos], result);
                    
                    pos += SIMD_WIDTH;
                    remaining -= SIMD_WIDTH;
                }
                
                // Handle remaining elements
                while (remaining > 0) {
                    output_channel[pos] = input_channel[pos] * scale + shift;
                    ++pos;
                    --remaining;
                }
            }
        }
#else
        // Fallback with loop unrolling
        constexpr int SPATIAL_UNROLL = 8;
        const int spatial_size = height * width;
        const int channel_stride = spatial_size;
        
        for (int n = 0; n < batch_size; ++n) {
            const float* input_batch = input_data + n * channels * spatial_size;
            float* output_batch = output_data + n * channels * spatial_size;
            
            for (int c = 0; c < channels; ++c) {
                const float mean_val = running_mean_data[c];
                const float var_val = running_var_data[c];
                const float inv_std = 1.0f / std::sqrt(var_val + eps_);
                const float scale = weight_data[c] * inv_std;
                const float shift = bias_data[c] - mean_val * scale;
                
                const float* input_channel = input_batch + c * channel_stride;
                float* output_channel = output_batch + c * channel_stride;
                
                // Vectorized computation with loop unrolling
                int remaining = spatial_size;
                int pos = 0;
                
                // Process in blocks of SPATIAL_UNROLL for better vectorization
                while (remaining >= SPATIAL_UNROLL) {
                    // Unrolled computation for better compiler optimization
                    output_channel[pos + 0] = input_channel[pos + 0] * scale + shift;
                    output_channel[pos + 1] = input_channel[pos + 1] * scale + shift;
                    output_channel[pos + 2] = input_channel[pos + 2] * scale + shift;
                    output_channel[pos + 3] = input_channel[pos + 3] * scale + shift;
                    output_channel[pos + 4] = input_channel[pos + 4] * scale + shift;
                    output_channel[pos + 5] = input_channel[pos + 5] * scale + shift;
                    output_channel[pos + 6] = input_channel[pos + 6] * scale + shift;
                    output_channel[pos + 7] = input_channel[pos + 7] * scale + shift;
                    
                    pos += SPATIAL_UNROLL;
                    remaining -= SPATIAL_UNROLL;
                }
                
                // Handle remaining elements
                while (remaining > 0) {
                    output_channel[pos] = input_channel[pos] * scale + shift;
                    ++pos;
                    --remaining;
                }
            }
        }
#endif
    }
    
    return output;
}

void BatchNorm2D::backward(const core::Tensor& grad_output, const core::Tensor& input,
                           const core::Tensor& normalized) {
    const auto& input_shape = input.shape();
    int batch_size = input_shape[0];
    int channels = input_shape[1];
    int height = input_shape[2];
    int width = input_shape[3];
    
    const float* grad_data = grad_output.data_float();
    const float* normalized_data = normalized.data_float();
    float* weight_grad_data = weight_grad_.data_float();
    float* bias_grad_data = bias_grad_.data_float();
    
    weight_grad_.zero();
    bias_grad_.zero();
    
    int spatial_size = height * width;
    int total_elements = batch_size * spatial_size;
    
    for (int c = 0; c < channels; ++c) {
        float grad_weight_sum = 0.0f;
        float grad_bias_sum = 0.0f;
        
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int idx = n * channels * height * width + c * height * width + h * width + w;
                    grad_weight_sum += grad_data[idx] * normalized_data[idx];
                    grad_bias_sum += grad_data[idx];
                }
            }
        }
        
        weight_grad_data[c] = grad_weight_sum;
        bias_grad_data[c] = grad_bias_sum;
    }
}

void BatchNorm2D::zero_grad() {
    weight_grad_.zero();
    bias_grad_.zero();
}

void BatchNorm2D::apply_gradients(float learning_rate) {
    float* weight_data = weight_.data_float();
    float* bias_data = bias_.data_float();
    const float* weight_grad_data = weight_grad_.data_float();
    const float* bias_grad_data = bias_grad_.data_float();
    
    for (int i = 0; i < num_features_; ++i) {
        weight_data[i] -= learning_rate * weight_grad_data[i];
        bias_data[i] -= learning_rate * bias_grad_data[i];
    }
}

void BatchNorm2D::initialize_parameters() {
    weight_.fill(1.0f);
    bias_.zero();
    running_mean_.zero();
    running_var_.fill(1.0f);
}

void BatchNorm2D::set_weight(const core::Tensor& weight) {
    if (weight.shape() != weight_.shape()) {
        throw std::runtime_error("Weight shape mismatch in BatchNorm2D::set_weight");
    }
    weight_ = weight;
}

void BatchNorm2D::set_bias(const core::Tensor& bias) {
    if (bias.shape() != bias_.shape()) {
        throw std::runtime_error("Bias shape mismatch in BatchNorm2D::set_bias");
    }
    bias_ = bias;
}

void BatchNorm2D::set_running_mean(const core::Tensor& running_mean) {
    if (running_mean.shape() != running_mean_.shape()) {
        throw std::runtime_error("Running mean shape mismatch in BatchNorm2D::set_running_mean");
    }
    running_mean_ = running_mean;
}

void BatchNorm2D::set_running_var(const core::Tensor& running_var) {
    if (running_var.shape() != running_var_.shape()) {
        throw std::runtime_error("Running variance shape mismatch in BatchNorm2D::set_running_var");
    }
    running_var_ = running_var;
}

}
}