#include "utils/onnx_simd_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef __x86_64__
#include <cpuid.h>
#endif

namespace tacs {
namespace utils {
namespace simd {

SimdCapability SimdDetector::detect_capability() {
    #ifdef __aarch64__
    return SimdCapability::NEON;
    #endif
    
    #ifdef __x86_64__
    if (has_avx512()) return SimdCapability::AVX512;
    if (has_avx2()) return SimdCapability::AVX2;
    if (has_avx()) return SimdCapability::AVX;
    if (has_sse4_1()) return SimdCapability::SSE4_1;
    if (has_sse2()) return SimdCapability::SSE2;
    #endif
    
    return SimdCapability::NONE;
}

bool SimdDetector::has_sse2() {
    #ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (edx & (1 << 26)) != 0;
    }
    #endif
    return false;
}

bool SimdDetector::has_sse4_1() {
    #ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (ecx & (1 << 19)) != 0;
    }
    #endif
    return false;
}

bool SimdDetector::has_avx() {
    #ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (ecx & (1 << 28)) != 0;
    }
    #endif
    return false;
}

bool SimdDetector::has_avx2() {
    #ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;
    }
    #endif
    return false;
}

bool SimdDetector::has_avx512() {
    #ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 16)) != 0;
    }
    #endif
    return false;
}

bool SimdDetector::has_neon() {
    #ifdef __aarch64__
    return true;
    #else
    return false;
    #endif
}

void SimdConv2D::compute(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch, int in_channels, int out_channels,
    int in_height, int in_width,
    int kernel_height, int kernel_width,
    int out_height, int out_width,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    SimdCapability simd_cap) {
    
    if (simd_cap == SimdCapability::NONE) {
        simd_cap = SimdDetector::detect_capability();
    }
    
    #ifdef __x86_64__
    if (simd_cap >= SimdCapability::AVX2) {
        compute_avx2(input, weight, bias, output,
                     batch, in_channels, out_channels,
                     in_height, in_width,
                     kernel_height, kernel_width,
                     out_height, out_width,
                     stride_h, stride_w,
                     pad_h, pad_w);
        return;
    }
    #endif
    
    #ifdef __aarch64__
    if (simd_cap == SimdCapability::NEON) {
        compute_neon(input, weight, bias, output,
                     batch, in_channels, out_channels,
                     in_height, in_width,
                     kernel_height, kernel_width,
                     out_height, out_width,
                     stride_h, stride_w,
                     pad_h, pad_w);
        return;
    }
    #endif
    
    compute_scalar(input, weight, bias, output,
                   batch, in_channels, out_channels,
                   in_height, in_width,
                   kernel_height, kernel_width,
                   out_height, out_width,
                   stride_h, stride_w,
                   pad_h, pad_w);
}

void SimdConv2D::compute_scalar(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch, int in_channels, int out_channels,
    int in_height, int in_width,
    int kernel_height, int kernel_width,
    int out_height, int out_width,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {
    
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = bias ? bias[oc] : 0.0f;
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_height; ++kh) {
                            for (int kw = 0; kw < kernel_width; ++kw) {
                                int ih = oh * stride_h - pad_h + kh;
                                int iw = ow * stride_w - pad_w + kw;
                                
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                                    int weight_idx = ((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw;
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                    
                    int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

#ifdef __x86_64__
void SimdConv2D::compute_avx2(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch, int in_channels, int out_channels,
    int in_height, int in_width,
    int kernel_height, int kernel_width,
    int out_height, int out_width,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {
    
    const int vec_size = 8;
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                int ow = 0;
                
                for (; ow + vec_size <= out_width; ow += vec_size) {
                    __m256 sum_vec = bias ? _mm256_broadcast_ss(&bias[oc]) : _mm256_setzero_ps();
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_height; ++kh) {
                            for (int kw = 0; kw < kernel_width; ++kw) {
                                int ih = oh * stride_h - pad_h + kh;
                                
                                if (ih >= 0 && ih < in_height) {
                                    __m256 input_vec = _mm256_setzero_ps();
                                    float input_vals[8];
                                    
                                    for (int v = 0; v < vec_size; ++v) {
                                        int iw = (ow + v) * stride_w - pad_w + kw;
                                        if (iw >= 0 && iw < in_width) {
                                            int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                                            input_vals[v] = input[input_idx];
                                        } else {
                                            input_vals[v] = 0.0f;
                                        }
                                    }
                                    
                                    input_vec = _mm256_loadu_ps(input_vals);
                                    
                                    int weight_idx = ((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw;
                                    __m256 weight_vec = _mm256_broadcast_ss(&weight[weight_idx]);
                                    
                                    sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
                                }
                            }
                        }
                    }
                    
                    float sum_arr[8];
                    _mm256_storeu_ps(sum_arr, sum_vec);
                    
                    for (int v = 0; v < vec_size; ++v) {
                        int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + (ow + v);
                        output[output_idx] = sum_arr[v];
                    }
                }
                
                for (; ow < out_width; ++ow) {
                    float sum = bias ? bias[oc] : 0.0f;
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_height; ++kh) {
                            for (int kw = 0; kw < kernel_width; ++kw) {
                                int ih = oh * stride_h - pad_h + kh;
                                int iw = ow * stride_w - pad_w + kw;
                                
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                                    int weight_idx = ((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw;
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                    
                    int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}
#endif

#ifdef __aarch64__
void SimdConv2D::compute_neon(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch, int in_channels, int out_channels,
    int in_height, int in_width,
    int kernel_height, int kernel_width,
    int out_height, int out_width,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {
    
    const int vec_size = 4;
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                int ow = 0;
                
                for (; ow + vec_size <= out_width; ow += vec_size) {
                    float32x4_t sum_vec = bias ? vdupq_n_f32(bias[oc]) : vdupq_n_f32(0.0f);
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_height; ++kh) {
                            for (int kw = 0; kw < kernel_width; ++kw) {
                                int ih = oh * stride_h - pad_h + kh;
                                
                                if (ih >= 0 && ih < in_height) {
                                    float input_vals[4];
                                    
                                    for (int v = 0; v < vec_size; ++v) {
                                        int iw = (ow + v) * stride_w - pad_w + kw;
                                        if (iw >= 0 && iw < in_width) {
                                            int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                                            input_vals[v] = input[input_idx];
                                        } else {
                                            input_vals[v] = 0.0f;
                                        }
                                    }
                                    
                                    float32x4_t input_vec = vld1q_f32(input_vals);
                                    
                                    int weight_idx = ((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw;
                                    float32x4_t weight_vec = vdupq_n_f32(weight[weight_idx]);
                                    
                                    sum_vec = vmlaq_f32(sum_vec, input_vec, weight_vec);
                                }
                            }
                        }
                    }
                    
                    float sum_arr[4];
                    vst1q_f32(sum_arr, sum_vec);
                    
                    for (int v = 0; v < vec_size; ++v) {
                        int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + (ow + v);
                        output[output_idx] = sum_arr[v];
                    }
                }
                
                for (; ow < out_width; ++ow) {
                    float sum = bias ? bias[oc] : 0.0f;
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_height; ++kh) {
                            for (int kw = 0; kw < kernel_width; ++kw) {
                                int ih = oh * stride_h - pad_h + kh;
                                int iw = ow * stride_w - pad_w + kw;
                                
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                                    int weight_idx = ((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw;
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                    
                    int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}
#endif

void SimdBatchNorm::compute(
    const float* input, const float* scale, const float* bias,
    const float* mean, const float* var,
    float* output,
    int batch, int channels, int spatial_size,
    float epsilon,
    SimdCapability simd_cap) {
    
    if (simd_cap == SimdCapability::NONE) {
        simd_cap = SimdDetector::detect_capability();
    }
    
    #ifdef __x86_64__
    if (simd_cap >= SimdCapability::AVX2) {
        compute_avx2(input, scale, bias, mean, var, output,
                     batch, channels, spatial_size, epsilon);
        return;
    }
    #endif
    
    #ifdef __aarch64__
    if (simd_cap == SimdCapability::NEON) {
        compute_neon(input, scale, bias, mean, var, output,
                     batch, channels, spatial_size, epsilon);
        return;
    }
    #endif
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            float inv_std = 1.0f / std::sqrt(var[c] + epsilon);
            float scale_val = scale[c];
            float bias_val = bias[c];
            float mean_val = mean[c];
            
            for (int i = 0; i < spatial_size; ++i) {
                int idx = (b * channels + c) * spatial_size + i;
                output[idx] = (input[idx] - mean_val) * inv_std * scale_val + bias_val;
            }
        }
    }
}

#ifdef __x86_64__
void SimdBatchNorm::compute_avx2(
    const float* input, const float* scale, const float* bias,
    const float* mean, const float* var,
    float* output,
    int batch, int channels, int spatial_size,
    float epsilon) {
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            float inv_std = 1.0f / std::sqrt(var[c] + epsilon);
            __m256 inv_std_vec = _mm256_broadcast_ss(&inv_std);
            __m256 scale_vec = _mm256_broadcast_ss(&scale[c]);
            __m256 bias_vec = _mm256_broadcast_ss(&bias[c]);
            __m256 mean_vec = _mm256_broadcast_ss(&mean[c]);
            
            int i = 0;
            for (; i + 8 <= spatial_size; i += 8) {
                int idx = (b * channels + c) * spatial_size + i;
                __m256 input_vec = _mm256_loadu_ps(&input[idx]);
                __m256 normalized = _mm256_sub_ps(input_vec, mean_vec);
                normalized = _mm256_mul_ps(normalized, inv_std_vec);
                normalized = _mm256_mul_ps(normalized, scale_vec);
                normalized = _mm256_add_ps(normalized, bias_vec);
                _mm256_storeu_ps(&output[idx], normalized);
            }
            
            for (; i < spatial_size; ++i) {
                int idx = (b * channels + c) * spatial_size + i;
                output[idx] = (input[idx] - mean[c]) * inv_std * scale[c] + bias[c];
            }
        }
    }
}
#endif

#ifdef __aarch64__
void SimdBatchNorm::compute_neon(
    const float* input, const float* scale, const float* bias,
    const float* mean, const float* var,
    float* output,
    int batch, int channels, int spatial_size,
    float epsilon) {
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            float inv_std = 1.0f / std::sqrt(var[c] + epsilon);
            float32x4_t inv_std_vec = vdupq_n_f32(inv_std);
            float32x4_t scale_vec = vdupq_n_f32(scale[c]);
            float32x4_t bias_vec = vdupq_n_f32(bias[c]);
            float32x4_t mean_vec = vdupq_n_f32(mean[c]);
            
            int i = 0;
            for (; i + 4 <= spatial_size; i += 4) {
                int idx = (b * channels + c) * spatial_size + i;
                float32x4_t input_vec = vld1q_f32(&input[idx]);
                float32x4_t normalized = vsubq_f32(input_vec, mean_vec);
                normalized = vmulq_f32(normalized, inv_std_vec);
                normalized = vmulq_f32(normalized, scale_vec);
                normalized = vaddq_f32(normalized, bias_vec);
                vst1q_f32(&output[idx], normalized);
            }
            
            for (; i < spatial_size; ++i) {
                int idx = (b * channels + c) * spatial_size + i;
                output[idx] = (input[idx] - mean[c]) * inv_std * scale[c] + bias[c];
            }
        }
    }
}
#endif

void SimdActivations::relu(const float* input, float* output, size_t size,
                            SimdCapability simd_cap) {
    if (simd_cap == SimdCapability::NONE) {
        simd_cap = SimdDetector::detect_capability();
    }
    
    #ifdef __x86_64__
    if (simd_cap >= SimdCapability::AVX2) {
        relu_avx2(input, output, size);
        return;
    }
    #endif
    
    #ifdef __aarch64__
    if (simd_cap == SimdCapability::NEON) {
        relu_neon(input, output, size);
        return;
    }
    #endif
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

#ifdef __x86_64__
void SimdActivations::relu_avx2(const float* input, float* output, size_t size) {
    __m256 zero = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 in_vec = _mm256_loadu_ps(&input[i]);
        __m256 out_vec = _mm256_max_ps(in_vec, zero);
        _mm256_storeu_ps(&output[i], out_vec);
    }
    
    for (; i < size; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}
#endif

#ifdef __aarch64__
void SimdActivations::relu_neon(const float* input, float* output, size_t size) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t in_vec = vld1q_f32(&input[i]);
        float32x4_t out_vec = vmaxq_f32(in_vec, zero);
        vst1q_f32(&output[i], out_vec);
    }
    
    for (; i < size; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}
#endif

void SimdActivations::leaky_relu(const float* input, float* output, size_t size,
                                 float alpha, SimdCapability simd_cap) {
    if (simd_cap == SimdCapability::NONE) {
        simd_cap = SimdDetector::detect_capability();
    }
    
    #ifdef __x86_64__
    if (simd_cap >= SimdCapability::AVX2) {
        leaky_relu_avx2(input, output, size, alpha);
        return;
    }
    #endif
    
    #ifdef __aarch64__
    if (simd_cap == SimdCapability::NEON) {
        leaky_relu_neon(input, output, size, alpha);
        return;
    }
    #endif
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        output[i] = input[i] > 0 ? input[i] : alpha * input[i];
    }
}

#ifdef __x86_64__
void SimdActivations::leaky_relu_avx2(const float* input, float* output, size_t size, float alpha) {
    __m256 zero = _mm256_setzero_ps();
    __m256 alpha_vec = _mm256_broadcast_ss(&alpha);
    
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 in_vec = _mm256_loadu_ps(&input[i]);
        __m256 mask = _mm256_cmp_ps(in_vec, zero, _CMP_GT_OQ);
        __m256 neg_part = _mm256_mul_ps(in_vec, alpha_vec);
        __m256 out_vec = _mm256_blendv_ps(neg_part, in_vec, mask);
        _mm256_storeu_ps(&output[i], out_vec);
    }
    
    for (; i < size; ++i) {
        output[i] = input[i] > 0 ? input[i] : alpha * input[i];
    }
}
#endif

#ifdef __aarch64__
void SimdActivations::leaky_relu_neon(const float* input, float* output, size_t size, float alpha) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t alpha_vec = vdupq_n_f32(alpha);
    
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t in_vec = vld1q_f32(&input[i]);
        uint32x4_t mask = vcgtq_f32(in_vec, zero);
        float32x4_t neg_part = vmulq_f32(in_vec, alpha_vec);
        float32x4_t out_vec = vbslq_f32(mask, in_vec, neg_part);
        vst1q_f32(&output[i], out_vec);
    }
    
    for (; i < size; ++i) {
        output[i] = input[i] > 0 ? input[i] : alpha * input[i];
    }
}
#endif

void SimdPooling::max_pool_2d(
    const float* input, float* output,
    int batch, int channels,
    int in_height, int in_width,
    int out_height, int out_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    SimdCapability simd_cap) {
    
    if (simd_cap == SimdCapability::NONE) {
        simd_cap = SimdDetector::detect_capability();
    }
    
    #ifdef __x86_64__
    if (simd_cap >= SimdCapability::AVX2) {
        max_pool_2d_avx2(input, output, batch, channels,
                         in_height, in_width, out_height, out_width,
                         kernel_h, kernel_w, stride_h, stride_w);
        return;
    }
    #endif
    
    #ifdef __aarch64__
    if (simd_cap == SimdCapability::NEON) {
        max_pool_2d_neon(input, output, batch, channels,
                         in_height, in_width, out_height, out_width,
                         kernel_h, kernel_w, stride_h, stride_w);
        return;
    }
    #endif
    
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride_h + kh;
                            int iw = ow * stride_w + kw;
                            
                            if (ih < in_height && iw < in_width) {
                                int idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                                max_val = std::max(max_val, input[idx]);
                            }
                        }
                    }
                    
                    int out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }
}

#ifdef __x86_64__
void SimdPooling::max_pool_2d_avx2(
    const float* input, float* output,
    int batch, int channels,
    int in_height, int in_width,
    int out_height, int out_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w) {
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                int ow = 0;
                
                for (; ow + 8 <= out_width; ow += 8) {
                    __m256 max_vals[8];
                    for (int i = 0; i < 8; ++i) {
                        max_vals[i] = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
                    }
                    
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride_h + kh;
                            
                            if (ih < in_height) {
                                for (int v = 0; v < 8; ++v) {
                                    int iw = (ow + v) * stride_w + kw;
                                    if (iw < in_width) {
                                        int idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                                        __m256 val = _mm256_broadcast_ss(&input[idx]);
                                        max_vals[v] = _mm256_max_ps(max_vals[v], val);
                                    }
                                }
                            }
                        }
                    }
                    
                    for (int v = 0; v < 8; ++v) {
                        float max_val = _mm256_cvtss_f32(max_vals[v]);
                        int out_idx = ((b * channels + c) * out_height + oh) * out_width + (ow + v);
                        output[out_idx] = max_val;
                    }
                }
                
                for (; ow < out_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride_h + kh;
                            int iw = ow * stride_w + kw;
                            
                            if (ih < in_height && iw < in_width) {
                                int idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                                max_val = std::max(max_val, input[idx]);
                            }
                        }
                    }
                    
                    int out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }
}
#endif

#ifdef __aarch64__
void SimdPooling::max_pool_2d_neon(
    const float* input, float* output,
    int batch, int channels,
    int in_height, int in_width,
    int out_height, int out_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w) {
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                int ow = 0;
                
                for (; ow + 4 <= out_width; ow += 4) {
                    float32x4_t max_vals[4];
                    for (int i = 0; i < 4; ++i) {
                        max_vals[i] = vdupq_n_f32(-std::numeric_limits<float>::infinity());
                    }
                    
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride_h + kh;
                            
                            if (ih < in_height) {
                                for (int v = 0; v < 4; ++v) {
                                    int iw = (ow + v) * stride_w + kw;
                                    if (iw < in_width) {
                                        int idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                                        float32x4_t val = vdupq_n_f32(input[idx]);
                                        max_vals[v] = vmaxq_f32(max_vals[v], val);
                                    }
                                }
                            }
                        }
                    }
                    
                    for (int v = 0; v < 4; ++v) {
                        float max_val = vgetq_lane_f32(max_vals[v], 0);
                        int out_idx = ((b * channels + c) * out_height + oh) * out_width + (ow + v);
                        output[out_idx] = max_val;
                    }
                }
                
                for (; ow < out_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride_h + kh;
                            int iw = ow * stride_w + kw;
                            
                            if (ih < in_height && iw < in_width) {
                                int idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                                max_val = std::max(max_val, input[idx]);
                            }
                        }
                    }
                    
                    int out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }
}
#endif

void SimdElementwise::add(const float* a, const float* b, float* output, size_t size,
                          SimdCapability simd_cap) {
    if (simd_cap == SimdCapability::NONE) {
        simd_cap = SimdDetector::detect_capability();
    }
    
    #ifdef __x86_64__
    if (simd_cap >= SimdCapability::AVX2) {
        add_avx2(a, b, output, size);
        return;
    }
    #endif
    
    #ifdef __aarch64__
    if (simd_cap == SimdCapability::NEON) {
        add_neon(a, b, output, size);
        return;
    }
    #endif
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        output[i] = a[i] + b[i];
    }
}

#ifdef __x86_64__
void SimdElementwise::add_avx2(const float* a, const float* b, float* output, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 out_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(&output[i], out_vec);
    }
    
    for (; i < size; ++i) {
        output[i] = a[i] + b[i];
    }
}
#endif

#ifdef __aarch64__
void SimdElementwise::add_neon(const float* a, const float* b, float* output, size_t size) {
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t a_vec = vld1q_f32(&a[i]);
        float32x4_t b_vec = vld1q_f32(&b[i]);
        float32x4_t out_vec = vaddq_f32(a_vec, b_vec);
        vst1q_f32(&output[i], out_vec);
    }
    
    for (; i < size; ++i) {
        output[i] = a[i] + b[i];
    }
}
#endif

void SimdElementwise::multiply(const float* a, const float* b, float* output, size_t size,
                               SimdCapability simd_cap) {
    if (simd_cap == SimdCapability::NONE) {
        simd_cap = SimdDetector::detect_capability();
    }
    
    #ifdef __x86_64__
    if (simd_cap >= SimdCapability::AVX2) {
        multiply_avx2(a, b, output, size);
        return;
    }
    #endif
    
    #ifdef __aarch64__
    if (simd_cap == SimdCapability::NEON) {
        multiply_neon(a, b, output, size);
        return;
    }
    #endif
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        output[i] = a[i] * b[i];
    }
}

#ifdef __x86_64__
void SimdElementwise::multiply_avx2(const float* a, const float* b, float* output, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 out_vec = _mm256_mul_ps(a_vec, b_vec);
        _mm256_storeu_ps(&output[i], out_vec);
    }
    
    for (; i < size; ++i) {
        output[i] = a[i] * b[i];
    }
}
#endif

#ifdef __aarch64__
void SimdElementwise::multiply_neon(const float* a, const float* b, float* output, size_t size) {
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t a_vec = vld1q_f32(&a[i]);
        float32x4_t b_vec = vld1q_f32(&b[i]);
        float32x4_t out_vec = vmulq_f32(a_vec, b_vec);
        vst1q_f32(&output[i], out_vec);
    }
    
    for (; i < size; ++i) {
        output[i] = a[i] * b[i];
    }
}
#endif

void SimdActivations::sigmoid(const float* input, float* output, size_t size,
                              SimdCapability simd_cap) {
    if (simd_cap == SimdCapability::NONE) {
        simd_cap = SimdDetector::detect_capability();
    }
    
    #ifdef __x86_64__
    if (simd_cap >= SimdCapability::AVX2) {
        sigmoid_avx2(input, output, size);
        return;
    }
    #endif
    
    #ifdef __aarch64__
    if (simd_cap == SimdCapability::NEON) {
        sigmoid_neon(input, output, size);
        return;
    }
    #endif
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

#ifdef __x86_64__
void SimdActivations::sigmoid_avx2(const float* input, float* output, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 in_vec = _mm256_loadu_ps(&input[i]);
        __m256 neg_vec = _mm256_sub_ps(_mm256_setzero_ps(), in_vec);
        
        float vals[8];
        _mm256_storeu_ps(vals, neg_vec);
        for (int j = 0; j < 8; ++j) {
            vals[j] = 1.0f / (1.0f + std::exp(vals[j]));
        }
        _mm256_storeu_ps(&output[i], _mm256_loadu_ps(vals));
    }
    
    for (; i < size; ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}
#endif

#ifdef __aarch64__
void SimdActivations::sigmoid_neon(const float* input, float* output, size_t size) {
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t in_vec = vld1q_f32(&input[i]);
        float32x4_t neg_vec = vnegq_f32(in_vec);
        
        float vals[4];
        vst1q_f32(vals, neg_vec);
        for (int j = 0; j < 4; ++j) {
            vals[j] = 1.0f / (1.0f + std::exp(vals[j]));
        }
        vst1q_f32(&output[i], vld1q_f32(vals));
    }
    
    for (; i < size; ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}
#endif

void SimdActivations::tanh(const float* input, float* output, size_t size,
                           SimdCapability simd_cap) {
    if (simd_cap == SimdCapability::NONE) {
        simd_cap = SimdDetector::detect_capability();
    }
    
    #ifdef __x86_64__
    if (simd_cap >= SimdCapability::AVX2) {
        tanh_avx2(input, output, size);
        return;
    }
    #endif
    
    #ifdef __aarch64__
    if (simd_cap == SimdCapability::NEON) {
        tanh_neon(input, output, size);
        return;
    }
    #endif
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::tanh(input[i]);
    }
}

#ifdef __x86_64__
void SimdActivations::tanh_avx2(const float* input, float* output, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        float vals[8];
        _mm256_storeu_ps(vals, _mm256_loadu_ps(&input[i]));
        for (int j = 0; j < 8; ++j) {
            vals[j] = std::tanh(vals[j]);
        }
        _mm256_storeu_ps(&output[i], _mm256_loadu_ps(vals));
    }
    
    for (; i < size; ++i) {
        output[i] = std::tanh(input[i]);
    }
}
#endif

#ifdef __aarch64__
void SimdActivations::tanh_neon(const float* input, float* output, size_t size) {
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        float vals[4];
        vst1q_f32(vals, vld1q_f32(&input[i]));
        for (int j = 0; j < 4; ++j) {
            vals[j] = std::tanh(vals[j]);
        }
        vst1q_f32(&output[i], vld1q_f32(vals));
    }
    
    for (; i < size; ++i) {
        output[i] = std::tanh(input[i]);
    }
}
#endif

}
}
}