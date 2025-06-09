#include "utils/matrix_ops.h"
#include <cstring>
#include <algorithm>
#include <thread>
#include <vector>

namespace tacs {
namespace utils {

void MatrixOps::gemm(const core::Tensor& a, const core::Tensor& b, core::Tensor& c,
                     float alpha, float beta, bool transpose_a, bool transpose_b) {
    if (a.dtype() != core::DataType::FLOAT32 || b.dtype() != core::DataType::FLOAT32 ||
        c.dtype() != core::DataType::FLOAT32) {
        throw std::runtime_error("GEMM only supports FLOAT32 tensors");
    }
    
    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();
    const auto& c_shape = c.shape();
    
    if (a_shape.size() != 2 || b_shape.size() != 2 || c_shape.size() != 2) {
        throw std::runtime_error("GEMM requires 2D tensors");
    }
    
    int m = transpose_a ? a_shape[1] : a_shape[0];
    int k = transpose_a ? a_shape[0] : a_shape[1];
    int n = transpose_b ? b_shape[0] : b_shape[1];
    
    if (k != (transpose_b ? b_shape[1] : b_shape[0])) {
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");
    }
    
    if (c_shape[0] != m || c_shape[1] != n) {
        throw std::runtime_error("Output tensor dimensions incorrect");
    }
    
    constexpr int BLOCK_SIZE = 64;
    if (m >= BLOCK_SIZE && n >= BLOCK_SIZE && k >= BLOCK_SIZE) {
        gemm_blocked(a.data_float(), b.data_float(), c.data_float(),
                     m, n, k, alpha, beta, transpose_a, transpose_b);
    } else {
        gemm_naive(a.data_float(), b.data_float(), c.data_float(),
                   m, n, k, alpha, beta, transpose_a, transpose_b);
    }
}

/**
 * @brief Performs 2D convolution operation with configurable stride and padding
 * 
 * Implements standard convolution using nested loops optimized for cache efficiency.
 * Handles boundary conditions with zero-padding and supports arbitrary kernel sizes.
 * Critical path for neural network forward pass - requires <10ms execution time.
 */
/**
 * @brief Ultra-optimized convolution for sub-50ms inference
 * 
 * Implements fastest possible convolution with aggressive optimizations:
 * - Direct convolution without im2col for small kernels
 * - SIMD vectorization with manual loop unrolling
 * - Cache-optimized memory access patterns
 * - Fast paths for common kernel sizes
 */
void MatrixOps::conv2d(const core::Tensor& input, const core::Tensor& weight, core::Tensor& output,
                       int stride_h, int stride_w, int pad_h, int pad_w) {
    const auto& input_shape = input.shape();
    const auto& weight_shape = weight.shape();
    const auto& output_shape = output.shape();
    
    if (input_shape.size() != 4 || weight_shape.size() != 4 || output_shape.size() != 4) {
        throw std::runtime_error("Conv2D requires 4D tensors (NCHW format)");
    }
    
    int batch_size = input_shape[0];
    int in_channels = input_shape[1];
    int input_h = input_shape[2];
    int input_w = input_shape[3];
    
    int out_channels = weight_shape[0];
    int kernel_h = weight_shape[2];
    int kernel_w = weight_shape[3];
    
    int output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    
    const float* input_data = input.data_float();
    const float* weight_data = weight.data_float();
    float* output_data = output.data_float();
    
    std::fill(output_data, output_data + output.size(), 0.0f);
    
    // ULTRA-FAST MODE: Use most aggressive optimizations for sub-50ms inference
    if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
        // Force use of optimized 3x3 implementation regardless of SIMD availability
#if defined(__AVX2__) || defined(__ARM_NEON)
        conv2d_3x3_simd(input_data, weight_data, output_data, 
                        batch_size, in_channels, out_channels,
                        input_h, input_w, output_h, output_w, pad_h, pad_w);
#else
        conv2d_3x3_optimized(input_data, weight_data, output_data, 
                             batch_size, in_channels, out_channels,
                             input_h, input_w, output_h, output_w, pad_h, pad_w);
#endif
    } else if (kernel_h == 1 && kernel_w == 1) {
#if defined(__AVX2__) || defined(__ARM_NEON)
        conv2d_1x1_simd(input_data, weight_data, output_data,
                        batch_size, in_channels, out_channels,
                        input_h, input_w, output_h, output_w);
#else
        conv2d_1x1_optimized(input_data, weight_data, output_data,
                             batch_size, in_channels, out_channels,
                             input_h, input_w, output_h, output_w);
#endif
    } else {
        // Use parallel processing for large tensors
        int total_ops = batch_size * out_channels * output_h * output_w;
        if (total_ops > 50000) {
            conv2d_parallel(input_data, weight_data, output_data,
                           batch_size, in_channels, out_channels,
                           input_h, input_w, output_h, output_w,
                           kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
        } else {
            conv2d_generic_optimized(input_data, weight_data, output_data,
                                    batch_size, in_channels, out_channels,
                                    input_h, input_w, output_h, output_w,
                                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
        }
    }
}

void MatrixOps::add_bias(core::Tensor& output, const core::Tensor& bias) {
    const auto& output_shape = output.shape();
    const auto& bias_shape = bias.shape();
    
    if (output_shape.size() != 4 || bias_shape.size() != 1) {
        throw std::runtime_error("add_bias expects 4D output and 1D bias");
    }
    
    int batch_size = output_shape[0];
    int channels = output_shape[1];
    int height = output_shape[2];
    int width = output_shape[3];
    
    if (bias_shape[0] != channels) {
        throw std::runtime_error("Bias size must match number of channels");
    }
    
    float* output_data = output.data_float();
    const float* bias_data = bias.data_float();
    
    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channels; ++c) {
            float bias_val = bias_data[c];
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int idx = n * channels * height * width + c * height * width + h * width + w;
                    output_data[idx] += bias_val;
                }
            }
        }
    }
}

void MatrixOps::quantize_fp32_to_int8(const core::Tensor& input, core::Tensor& output,
                                      float scale, int zero_point) {
    if (input.dtype() != core::DataType::FLOAT32 || output.dtype() != core::DataType::INT8) {
        throw std::runtime_error("quantize_fp32_to_int8 requires FLOAT32 input and INT8 output");
    }
    
    if (input.size() != output.size()) {
        throw std::runtime_error("Input and output tensors must have same size");
    }
    
    const float* input_data = input.data_float();
    int8_t* output_data = output.data_int8();
    const size_t size = input.size();
    
#ifdef __AVX2__
    // SIMD-optimized INT8 quantization with AVX2
    constexpr int SIMD_WIDTH = 8;
    __m256 scale_vec = _mm256_set1_ps(1.0f / scale);
    __m256 zero_point_vec = _mm256_set1_ps(static_cast<float>(zero_point));
    __m256i min_val = _mm256_set1_epi32(-128);
    __m256i max_val = _mm256_set1_epi32(127);
    
    size_t remaining = size;
    size_t pos = 0;
    
    // Process in SIMD blocks
    while (remaining >= SIMD_WIDTH) {
        __m256 input_vec = _mm256_loadu_ps(&input_data[pos]);
        __m256 scaled = _mm256_mul_ps(input_vec, scale_vec);
        __m256 shifted = _mm256_add_ps(scaled, zero_point_vec);
        __m256i rounded = _mm256_cvtps_epi32(shifted);
        
        // Clamp to INT8 range
        rounded = _mm256_max_epi32(rounded, min_val);
        rounded = _mm256_min_epi32(rounded, max_val);
        
        // Pack to INT8
        __m128i lower = _mm256_extracti128_si256(rounded, 0);
        __m128i upper = _mm256_extracti128_si256(rounded, 1);
        __m128i packed = _mm_packs_epi32(lower, upper);
        __m128i packed_8 = _mm_packs_epi16(packed, packed);
        
        // Store 8 INT8 values
        _mm_storel_epi64(reinterpret_cast<__m128i*>(&output_data[pos]), packed_8);
        
        pos += SIMD_WIDTH;
        remaining -= SIMD_WIDTH;
    }
    
    // Handle remaining elements
    while (remaining > 0) {
        int quantized = static_cast<int>(std::round(input_data[pos] / scale + zero_point));
        output_data[pos] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
        ++pos;
        --remaining;
    }
#elif defined(__ARM_NEON)
    // SIMD-optimized INT8 quantization with NEON
    constexpr int SIMD_WIDTH = 4;
    float32x4_t scale_vec = vdupq_n_f32(1.0f / scale);
    float32x4_t zero_point_vec = vdupq_n_f32(static_cast<float>(zero_point));
    
    size_t remaining = size;
    size_t pos = 0;
    
    // Process in SIMD blocks
    while (remaining >= SIMD_WIDTH) {
        float32x4_t input_vec = vld1q_f32(&input_data[pos]);
        float32x4_t scaled = vmulq_f32(input_vec, scale_vec);
        float32x4_t shifted = vaddq_f32(scaled, zero_point_vec);
        int32x4_t rounded = vcvtnq_s32_f32(shifted);
        
        // Clamp to INT8 range
        int32x4_t clamped = vmaxq_s32(rounded, vdupq_n_s32(-128));
        clamped = vminq_s32(clamped, vdupq_n_s32(127));
        
        // Extract and store INT8 values
        output_data[pos + 0] = static_cast<int8_t>(vgetq_lane_s32(clamped, 0));
        output_data[pos + 1] = static_cast<int8_t>(vgetq_lane_s32(clamped, 1));
        output_data[pos + 2] = static_cast<int8_t>(vgetq_lane_s32(clamped, 2));
        output_data[pos + 3] = static_cast<int8_t>(vgetq_lane_s32(clamped, 3));
        
        pos += SIMD_WIDTH;
        remaining -= SIMD_WIDTH;
    }
    
    // Handle remaining elements
    while (remaining > 0) {
        int quantized = static_cast<int>(std::round(input_data[pos] / scale + zero_point));
        output_data[pos] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
        ++pos;
        --remaining;
    }
#else
    // Optimized scalar implementation with loop unrolling
    constexpr int UNROLL_FACTOR = 8;
    const float inv_scale = 1.0f / scale;
    const float fp_zero_point = static_cast<float>(zero_point);
    
    size_t remaining = size;
    size_t pos = 0;
    
    // Process in unrolled blocks
    while (remaining >= UNROLL_FACTOR) {
        for (int i = 0; i < UNROLL_FACTOR; ++i) {
            float scaled = input_data[pos + i] * inv_scale + fp_zero_point;
            int quantized = static_cast<int>(std::round(scaled));
            output_data[pos + i] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
        }
        
        pos += UNROLL_FACTOR;
        remaining -= UNROLL_FACTOR;
    }
    
    // Handle remaining elements
    while (remaining > 0) {
        int quantized = static_cast<int>(std::round(input_data[pos] / scale + zero_point));
        output_data[pos] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
        ++pos;
        --remaining;
    }
#endif
}

void MatrixOps::quantize_fp32_to_fp16(const core::Tensor& input, core::Tensor& output) {
    if (input.dtype() != core::DataType::FLOAT32 || output.dtype() != core::DataType::FLOAT16) {
        throw std::runtime_error("quantize_fp32_to_fp16 requires FLOAT32 input and FLOAT16 output");
    }
    
    if (input.size() != output.size()) {
        throw std::runtime_error("Input and output tensors must have same size");
    }
    
    const float* input_data = input.data_float();
    uint16_t* output_data = static_cast<uint16_t*>(output.data());
    const size_t size = input.size();
    
#ifdef __AVX2__
    // SIMD-optimized FP16 conversion with AVX2 + F16C
    #ifdef __F16C__
    constexpr int SIMD_WIDTH = 8;
    size_t remaining = size;
    size_t pos = 0;
    
    // Process in SIMD blocks using F16C intrinsics
    while (remaining >= SIMD_WIDTH) {
        __m256 input_vec = _mm256_loadu_ps(&input_data[pos]);
        __m128i fp16_vec = _mm256_cvtps_ph(input_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&output_data[pos]), fp16_vec);
        
        pos += SIMD_WIDTH;
        remaining -= SIMD_WIDTH;
    }
    
    // Handle remaining elements
    while (remaining > 0) {
        __m128 single = _mm_load_ss(&input_data[pos]);
        __m128i fp16_single = _mm_cvtps_ph(single, _MM_FROUND_TO_NEAREST_INT);
        output_data[pos] = static_cast<uint16_t>(_mm_extract_epi16(fp16_single, 0));
        ++pos;
        --remaining;
    }
    #else
    // Manual conversion when F16C is not available
    goto manual_conversion;
    #endif
#elif defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
    // SIMD-optimized FP16 conversion with NEON
    constexpr int SIMD_WIDTH = 4;
    size_t remaining = size;
    size_t pos = 0;
    
    // Process in SIMD blocks
    while (remaining >= SIMD_WIDTH) {
        float32x4_t input_vec = vld1q_f32(&input_data[pos]);
        float16x4_t fp16_vec = vcvt_f16_f32(input_vec);
        vst1_f16(reinterpret_cast<__fp16*>(&output_data[pos]), fp16_vec);
        
        pos += SIMD_WIDTH;
        remaining -= SIMD_WIDTH;
    }
    
    // Handle remaining elements
    while (remaining > 0) {
        // Manual conversion for remaining elements
        uint32_t f32_bits = *reinterpret_cast<const uint32_t*>(&input_data[pos]);
        uint16_t sign = (f32_bits >> 16) & 0x8000;
        uint32_t exp = (f32_bits >> 23) & 0xff;
        uint32_t mantissa = f32_bits & 0x7fffff;
        
        if (exp == 0) {
            output_data[pos] = sign;
        } else if (exp == 0xff) {
            output_data[pos] = sign | 0x7c00 | (mantissa ? 0x0200 : 0);
        } else {
            int new_exp = static_cast<int>(exp) - 127 + 15;
            if (new_exp >= 31) {
                output_data[pos] = sign | 0x7c00;
            } else if (new_exp <= 0) {
                output_data[pos] = sign;
            } else {
                output_data[pos] = sign | (new_exp << 10) | (mantissa >> 13);
            }
        }
        ++pos;
        --remaining;
    }
#else
manual_conversion:
    // Optimized manual conversion with loop unrolling
    constexpr int UNROLL_FACTOR = 8;
    size_t remaining = size;
    size_t pos = 0;
    
    // Process in unrolled blocks
    while (remaining >= UNROLL_FACTOR) {
        for (int i = 0; i < UNROLL_FACTOR; ++i) {
            uint32_t f32_bits = *reinterpret_cast<const uint32_t*>(&input_data[pos + i]);
            uint16_t sign = (f32_bits >> 16) & 0x8000;
            uint32_t exp = (f32_bits >> 23) & 0xff;
            uint32_t mantissa = f32_bits & 0x7fffff;
            
            if (exp == 0) {
                output_data[pos + i] = sign;
            } else if (exp == 0xff) {
                output_data[pos + i] = sign | 0x7c00 | (mantissa ? 0x0200 : 0);
            } else {
                int new_exp = static_cast<int>(exp) - 127 + 15;
                if (new_exp >= 31) {
                    output_data[pos + i] = sign | 0x7c00;
                } else if (new_exp <= 0) {
                    output_data[pos + i] = sign;
                } else {
                    output_data[pos + i] = sign | (new_exp << 10) | (mantissa >> 13);
                }
            }
        }
        
        pos += UNROLL_FACTOR;
        remaining -= UNROLL_FACTOR;
    }
    
    // Handle remaining elements
    while (remaining > 0) {
        uint32_t f32_bits = *reinterpret_cast<const uint32_t*>(&input_data[pos]);
        uint16_t sign = (f32_bits >> 16) & 0x8000;
        uint32_t exp = (f32_bits >> 23) & 0xff;
        uint32_t mantissa = f32_bits & 0x7fffff;
        
        if (exp == 0) {
            output_data[pos] = sign;
        } else if (exp == 0xff) {
            output_data[pos] = sign | 0x7c00 | (mantissa ? 0x0200 : 0);
        } else {
            int new_exp = static_cast<int>(exp) - 127 + 15;
            if (new_exp >= 31) {
                output_data[pos] = sign | 0x7c00;
            } else if (new_exp <= 0) {
                output_data[pos] = sign;
            } else {
                output_data[pos] = sign | (new_exp << 10) | (mantissa >> 13);
            }
        }
        ++pos;
        --remaining;
    }
#endif
}

void MatrixOps::gemm_naive(const float* a, const float* b, float* c,
                           int m, int n, int k, float alpha, float beta,
                           bool transpose_a, bool transpose_b) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                float a_val = transpose_a ? a[l * m + i] : a[i * k + l];
                float b_val = transpose_b ? b[j * k + l] : b[l * n + j];
                sum += a_val * b_val;
            }
            c[i * n + j] = alpha * sum + beta * c[i * n + j];
        }
    }
}

/**
 * @brief Cache-optimized blocked matrix multiplication for large matrices
 * 
 * Implements tiled matrix multiplication to improve cache locality and reduce
 * memory bandwidth requirements. Block size tuned for L1 cache efficiency.
 * Essential for high-performance neural network operations.
 */
void MatrixOps::gemm_blocked(const float* a, const float* b, float* c,
                             int m, int n, int k, float alpha, float beta,
                             bool transpose_a, bool transpose_b) {
    constexpr int BLOCK_SIZE_M = 96;
    constexpr int BLOCK_SIZE_N = 64;
    constexpr int BLOCK_SIZE_K = 256;
    constexpr int MICRO_M = 6;
    constexpr int MICRO_N = 8;
    
    // Optimized cache-blocking with micro-kernels
    for (int i = 0; i < m; i += BLOCK_SIZE_M) {
        int end_i = std::min(i + BLOCK_SIZE_M, m);
        
        for (int j = 0; j < n; j += BLOCK_SIZE_N) {
            int end_j = std::min(j + BLOCK_SIZE_N, n);
            
            // Initialize C block for first k-block
            for (int ii = i; ii < end_i; ++ii) {
                for (int jj = j; jj < end_j; ++jj) {
                    c[ii * n + jj] *= beta;
                }
            }
            
            for (int l = 0; l < k; l += BLOCK_SIZE_K) {
                int end_l = std::min(l + BLOCK_SIZE_K, k);
                
                // Micro-kernel optimization for inner blocks
                for (int ii = i; ii < end_i; ii += MICRO_M) {
                    int micro_end_i = std::min(ii + MICRO_M, end_i);
                    
                    for (int jj = j; jj < end_j; jj += MICRO_N) {
                        int micro_end_j = std::min(jj + MICRO_N, end_j);
                        
                        // High-performance micro-kernel with loop unrolling
                        float local_sum[MICRO_M * MICRO_N] = {0.0f};
                        
                        for (int ll = l; ll < end_l; ++ll) {
                            // Load A values once for all j positions
                            float a_vals[MICRO_M];
                            for (int mi = 0; mi < micro_end_i - ii; ++mi) {
                                a_vals[mi] = transpose_a ? a[ll * m + (ii + mi)] : a[(ii + mi) * k + ll];
                            }
                            
                            // Compute outer products with loop unrolling
                            for (int mj = 0; mj < micro_end_j - jj; ++mj) {
                                float b_val = transpose_b ? b[(jj + mj) * k + ll] : b[ll * n + (jj + mj)];
                                
                                // Unrolled computation for multiple rows
                                for (int mi = 0; mi < micro_end_i - ii; ++mi) {
                                    local_sum[mi * MICRO_N + mj] += a_vals[mi] * b_val;
                                }
                            }
                        }
                        
                        // Store accumulated results
                        for (int mi = 0; mi < micro_end_i - ii; ++mi) {
                            for (int mj = 0; mj < micro_end_j - jj; ++mj) {
                                c[(ii + mi) * n + (jj + mj)] += alpha * local_sum[mi * MICRO_N + mj];
                            }
                        }
                    }
                }
            }
        }
    }
}

void MatrixOps::conv2d_3x3_optimized(const float* input, const float* weight, float* output,
                                     int batch_size, int in_channels, int out_channels,
                                     int input_h, int input_w, int output_h, int output_w,
                                     int pad_h, int pad_w) {
    constexpr int SPATIAL_UNROLL = 8;
    constexpr int CHANNEL_BLOCK = 16;
    
    // Pre-compute strides for cache-friendly access
    const int input_batch_stride = in_channels * input_h * input_w;
    const int input_channel_stride = input_h * input_w;
    const int output_batch_stride = out_channels * output_h * output_w;
    const int output_channel_stride = output_h * output_w;
    const int weight_filter_stride = in_channels * 9;
    
    for (int n = 0; n < batch_size; ++n) {
        const float* input_batch = input + n * input_batch_stride;
        float* output_batch = output + n * output_batch_stride;
        
        // Process output channels in blocks for better cache utilization
        for (int oc_block = 0; oc_block < out_channels; oc_block += CHANNEL_BLOCK) {
            int oc_end = std::min(oc_block + CHANNEL_BLOCK, out_channels);
            
            for (int oc = oc_block; oc < oc_end; ++oc) {
                const float* weight_filter = weight + oc * weight_filter_stride;
                float* output_channel = output_batch + oc * output_channel_stride;
                
                // Initialize output channel to zero for accumulation
                std::fill(output_channel, output_channel + output_channel_stride, 0.0f);
                
                // Accumulate over input channels
                for (int ic = 0; ic < in_channels; ++ic) {
                    const float* input_channel = input_batch + ic * input_channel_stride;
                    const float* weight_channel = weight_filter + ic * 9;
                    
                    // Spatial convolution with aggressive loop unrolling
                    for (int oh = 0; oh < output_h; ++oh) {
                        const int ih_center = oh - pad_h;
                        
                        // Process multiple output width positions simultaneously
                        for (int ow = 0; ow < output_w; ow += SPATIAL_UNROLL) {
                            int ow_end = std::min(ow + SPATIAL_UNROLL, output_w);
                            float accumulator[SPATIAL_UNROLL] = {0.0f};
                            
                            // Unrolled 3x3 kernel computation
                            for (int kh = 0; kh < 3; ++kh) {
                                const int ih = ih_center + kh;
                                if (ih >= 0 && ih < input_h) {
                                    const float* input_row = input_channel + ih * input_w;
                                    const float w0 = weight_channel[kh * 3 + 0];
                                    const float w1 = weight_channel[kh * 3 + 1];
                                    const float w2 = weight_channel[kh * 3 + 2];
                                    
                                    // Vectorized computation across width
                                    for (int idx = 0; idx < ow_end - ow; ++idx) {
                                        const int ow_pos = ow + idx;
                                        const int iw_center = ow_pos - pad_w;
                                        
                                        // Manual 3x3 kernel unroll for maximum performance
                                        if (iw_center >= 0 && iw_center + 2 < input_w) {
                                            // Fast path: no boundary checking needed
                                            const float* input_pos = input_row + iw_center;
                                            accumulator[idx] += input_pos[0] * w0 + 
                                                              input_pos[1] * w1 + 
                                                              input_pos[2] * w2;
                                        } else {
                                            // Boundary-safe path
                                            for (int kw = 0; kw < 3; ++kw) {
                                                const int iw = iw_center + kw;
                                                if (iw >= 0 && iw < input_w) {
                                                    accumulator[idx] += input_row[iw] * weight_channel[kh * 3 + kw];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // Accumulate results into output
                            for (int idx = 0; idx < ow_end - ow; ++idx) {
                                output_channel[oh * output_w + ow + idx] += accumulator[idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

void MatrixOps::conv2d_1x1_optimized(const float* input, const float* weight, float* output,
                                     int batch_size, int in_channels, int out_channels,
                                     int input_h, int input_w, int output_h, int output_w) {
    int spatial_size = output_h * output_w;
    
    for (int n = 0; n < batch_size; ++n) {
        const float* input_batch = input + n * in_channels * spatial_size;
        float* output_batch = output + n * out_channels * spatial_size;
        
        gemm_naive(weight, input_batch, output_batch, out_channels, spatial_size, in_channels, 1.0f, 0.0f, false, false);
    }
}

void MatrixOps::conv2d_generic_optimized(const float* input, const float* weight, float* output,
                                         int batch_size, int in_channels, int out_channels,
                                         int input_h, int input_w, int output_h, int output_w,
                                         int kernel_h, int kernel_w, int stride_h, int stride_w,
                                         int pad_h, int pad_w) {
    constexpr int CACHE_BLOCK_SIZE = 32;
    constexpr int UNROLL_FACTOR = 4;
    
    // Pre-compute stride factors for improved cache efficiency
    const int input_channel_stride = input_h * input_w;
    const int output_channel_stride = output_h * output_w;
    const int weight_channel_stride = kernel_h * kernel_w;
    const int weight_filter_stride = in_channels * weight_channel_stride;
    
    for (int n = 0; n < batch_size; ++n) {
        const float* input_batch = input + n * in_channels * input_channel_stride;
        float* output_batch = output + n * out_channels * output_channel_stride;
        
        // Cache-blocking for output channels
        for (int oc_block = 0; oc_block < out_channels; oc_block += CACHE_BLOCK_SIZE) {
            int oc_end = std::min(oc_block + CACHE_BLOCK_SIZE, out_channels);
            
            // Cache-blocking for spatial dimensions
            for (int oh_block = 0; oh_block < output_h; oh_block += CACHE_BLOCK_SIZE) {
                int oh_end = std::min(oh_block + CACHE_BLOCK_SIZE, output_h);
                
                for (int ow_block = 0; ow_block < output_w; ow_block += CACHE_BLOCK_SIZE) {
                    int ow_end = std::min(ow_block + CACHE_BLOCK_SIZE, output_w);
                    
                    // Process cache block with optimized inner loops
                    for (int oc = oc_block; oc < oc_end; ++oc) {
                        const float* weight_filter = weight + oc * weight_filter_stride;
                        float* output_channel = output_batch + oc * output_channel_stride;
                        
                        // Vectorized spatial computation with loop unrolling
                        for (int oh = oh_block; oh < oh_end; ++oh) {
                            const int ih_base = oh * stride_h - pad_h;
                            
                            for (int ow = ow_block; ow < ow_end; ow += UNROLL_FACTOR) {
                                int ow_actual_end = std::min(ow + UNROLL_FACTOR, ow_end);
                                float accumulator[UNROLL_FACTOR] = {0.0f};
                                
                                // Compute multiple output pixels simultaneously
                                for (int ic = 0; ic < in_channels; ++ic) {
                                    const float* input_channel = input_batch + ic * input_channel_stride;
                                    const float* weight_channel = weight_filter + ic * weight_channel_stride;
                                    
                                    for (int kh = 0; kh < kernel_h; ++kh) {
                                        const int ih = ih_base + kh;
                                        if (ih >= 0 && ih < input_h) {
                                            const float* input_row = input_channel + ih * input_w;
                                            const float* weight_row = weight_channel + kh * kernel_w;
                                            
                                            for (int kw = 0; kw < kernel_w; ++kw) {
                                                const float weight_val = weight_row[kw];
                                                
                                                // Unrolled computation for multiple output positions
                                                for (int unroll_idx = 0; unroll_idx < ow_actual_end - ow; ++unroll_idx) {
                                                    const int iw = (ow + unroll_idx) * stride_w - pad_w + kw;
                                                    if (iw >= 0 && iw < input_w) {
                                                        accumulator[unroll_idx] += input_row[iw] * weight_val;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                // Store accumulated results
                                for (int unroll_idx = 0; unroll_idx < ow_actual_end - ow; ++unroll_idx) {
                                    output_channel[oh * output_w + ow + unroll_idx] = accumulator[unroll_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief SIMD-optimized 3x3 convolution using platform-specific intrinsics
 * 
 * Implements vectorized 3x3 convolution with aggressive loop unrolling and
 * SIMD instructions for maximum throughput. Critical for meeting 50ms latency.
 */
void MatrixOps::conv2d_3x3_simd(const float* input, const float* weight, float* output,
                                int batch_size, int in_channels, int out_channels,
                                int input_h, int input_w, int output_h, int output_w,
                                int pad_h, int pad_w) {
#ifdef __AVX2__
    constexpr int SIMD_WIDTH = 8;
    const int input_batch_stride = in_channels * input_h * input_w;
    const int output_batch_stride = out_channels * output_h * output_w;
    const int input_channel_stride = input_h * input_w;
    const int output_channel_stride = output_h * output_w;
    const int weight_filter_stride = in_channels * 9;
    
    for (int n = 0; n < batch_size; ++n) {
        const float* input_batch = input + n * input_batch_stride;
        float* output_batch = output + n * output_batch_stride;
        
        for (int oc = 0; oc < out_channels; ++oc) {
            const float* weight_filter = weight + oc * weight_filter_stride;
            float* output_channel = output_batch + oc * output_channel_stride;
            
            // Zero out output channel
            std::fill(output_channel, output_channel + output_channel_stride, 0.0f);
            
            for (int ic = 0; ic < in_channels; ++ic) {
                const float* input_channel = input_batch + ic * input_channel_stride;
                const float* weight_channel = weight_filter + ic * 9;
                
                // Load 3x3 weights into SIMD registers
                __m256 w0 = _mm256_broadcast_ss(&weight_channel[0]);
                __m256 w1 = _mm256_broadcast_ss(&weight_channel[1]);
                __m256 w2 = _mm256_broadcast_ss(&weight_channel[2]);
                __m256 w3 = _mm256_broadcast_ss(&weight_channel[3]);
                __m256 w4 = _mm256_broadcast_ss(&weight_channel[4]);
                __m256 w5 = _mm256_broadcast_ss(&weight_channel[5]);
                __m256 w6 = _mm256_broadcast_ss(&weight_channel[6]);
                __m256 w7 = _mm256_broadcast_ss(&weight_channel[7]);
                __m256 w8 = _mm256_broadcast_ss(&weight_channel[8]);
                
                for (int oh = 0; oh < output_h; ++oh) {
                    const int ih_center = oh - pad_h;
                    
                    for (int ow = 0; ow < output_w; ow += SIMD_WIDTH) {
                        int width_remaining = std::min(SIMD_WIDTH, output_w - ow);
                        
                        if (width_remaining == SIMD_WIDTH && 
                            ih_center >= 0 && ih_center + 2 < input_h) {
                            // Fast SIMD path for interior pixels
                            __m256 acc = _mm256_setzero_ps();
                            
                            // Unrolled 3x3 convolution with SIMD
                            for (int kh = 0; kh < 3; ++kh) {
                                const int ih = ih_center + kh;
                                const float* input_row = input_channel + ih * input_w;
                                
                                for (int idx = 0; idx < width_remaining; ++idx) {
                                    const int iw_center = (ow + idx) - pad_w;
                                    if (iw_center >= 0 && iw_center + 2 < input_w) {
                                        __m256 i0 = _mm256_broadcast_ss(&input_row[iw_center + 0]);
                                        __m256 i1 = _mm256_broadcast_ss(&input_row[iw_center + 1]);
                                        __m256 i2 = _mm256_broadcast_ss(&input_row[iw_center + 2]);
                                        
                                        __m256 kernel_weight;
                                        switch(kh) {
                                            case 0: kernel_weight = (idx == 0) ? w0 : ((idx == 1) ? w1 : w2); break;
                                            case 1: kernel_weight = (idx == 0) ? w3 : ((idx == 1) ? w4 : w5); break;
                                            case 2: kernel_weight = (idx == 0) ? w6 : ((idx == 1) ? w7 : w8); break;
                                        }
                                        
                                        acc = _mm256_fmadd_ps(i0, kernel_weight, acc);
                                    }
                                }
                            }
                            
                            // Add to existing output
                            __m256 current = _mm256_loadu_ps(&output_channel[oh * output_w + ow]);
                            __m256 result = _mm256_add_ps(current, acc);
                            _mm256_storeu_ps(&output_channel[oh * output_w + ow], result);
                        } else {
                            // Fallback to scalar for boundary cases
                            for (int idx = 0; idx < width_remaining; ++idx) {
                                float acc = 0.0f;
                                const int ow_pos = ow + idx;
                                const int iw_center = ow_pos - pad_w;
                                
                                for (int kh = 0; kh < 3; ++kh) {
                                    const int ih = ih_center + kh;
                                    if (ih >= 0 && ih < input_h) {
                                        for (int kw = 0; kw < 3; ++kw) {
                                            const int iw = iw_center + kw;
                                            if (iw >= 0 && iw < input_w) {
                                                acc += input_channel[ih * input_w + iw] * 
                                                       weight_channel[kh * 3 + kw];
                                            }
                                        }
                                    }
                                }
                                output_channel[oh * output_w + ow_pos] += acc;
                            }
                        }
                    }
                }
            }
        }
    }
#elif defined(__ARM_NEON)
    // ARM NEON implementation for mobile/embedded devices
    constexpr int SIMD_WIDTH = 4;
    const int input_batch_stride = in_channels * input_h * input_w;
    const int output_batch_stride = out_channels * output_h * output_w;
    const int input_channel_stride = input_h * input_w;
    const int output_channel_stride = output_h * output_w;
    const int weight_filter_stride = in_channels * 9;
    
    for (int n = 0; n < batch_size; ++n) {
        const float* input_batch = input + n * input_batch_stride;
        float* output_batch = output + n * output_batch_stride;
        
        for (int oc = 0; oc < out_channels; ++oc) {
            const float* weight_filter = weight + oc * weight_filter_stride;
            float* output_channel = output_batch + oc * output_channel_stride;
            
            std::fill(output_channel, output_channel + output_channel_stride, 0.0f);
            
            for (int ic = 0; ic < in_channels; ++ic) {
                const float* input_channel = input_batch + ic * input_channel_stride;
                const float* weight_channel = weight_filter + ic * 9;
                
                // Load weights into NEON registers
                float32x4_t w0123 = vld1q_f32(&weight_channel[0]);
                float32x4_t w4567 = vld1q_f32(&weight_channel[4]);
                float32x4_t w8xxx = vld1q_dup_f32(&weight_channel[8]);
                
                for (int oh = 0; oh < output_h; ++oh) {
                    const int ih_center = oh - pad_h;
                    
                    for (int ow = 0; ow < output_w; ow += SIMD_WIDTH) {
                        int width_remaining = std::min(SIMD_WIDTH, output_w - ow);
                        
                        if (width_remaining == SIMD_WIDTH && 
                            ih_center >= 0 && ih_center + 2 < input_h) {
                            float32x4_t acc = vdupq_n_f32(0.0f);
                            
                            // NEON optimized 3x3 convolution
                            for (int kh = 0; kh < 3; ++kh) {
                                const int ih = ih_center + kh;
                                const float* input_row = input_channel + ih * input_w;
                                
                                for (int kw = 0; kw < 3; ++kw) {
                                    // Load 4 input values efficiently for NEON
                                    float input_vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                                    
                                    for (int idx = 0; idx < SIMD_WIDTH; ++idx) {
                                        const int iw = (ow + idx) - pad_w + kw;
                                        if (iw >= 0 && iw < input_w) {
                                            input_vals[idx] = input_row[iw];
                                        }
                                    }
                                    
                                    float32x4_t inputs = vld1q_f32(input_vals);
                                    float32x4_t weight_vec = vdupq_n_f32(weight_channel[kh * 3 + kw]);
                                    acc = vmlaq_f32(acc, inputs, weight_vec);
                                }
                            }
                            
                            // Add to existing output
                            float32x4_t current = vld1q_f32(&output_channel[oh * output_w + ow]);
                            float32x4_t result = vaddq_f32(current, acc);
                            vst1q_f32(&output_channel[oh * output_w + ow], result);
                        } else {
                            // Scalar fallback for boundaries
                            for (int idx = 0; idx < width_remaining; ++idx) {
                                float acc = 0.0f;
                                const int ow_pos = ow + idx;
                                const int iw_center = ow_pos - pad_w;
                                
                                for (int kh = 0; kh < 3; ++kh) {
                                    const int ih = ih_center + kh;
                                    if (ih >= 0 && ih < input_h) {
                                        for (int kw = 0; kw < 3; ++kw) {
                                            const int iw = iw_center + kw;
                                            if (iw >= 0 && iw < input_w) {
                                                acc += input_channel[ih * input_w + iw] * 
                                                       weight_channel[kh * 3 + kw];
                                            }
                                        }
                                    }
                                }
                                output_channel[oh * output_w + ow_pos] += acc;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    // Fallback to optimized scalar implementation
    conv2d_3x3_optimized(input, weight, output, batch_size, in_channels, out_channels,
                         input_h, input_w, output_h, output_w, pad_h, pad_w);
#endif
}

/**
 * @brief SIMD-optimized 1x1 convolution for pointwise operations
 */
void MatrixOps::conv2d_1x1_simd(const float* input, const float* weight, float* output,
                               int batch_size, int in_channels, int out_channels,
                               int input_h, int input_w, int output_h, int output_w) {
#ifdef __AVX2__
    constexpr int SIMD_WIDTH = 8;
    const int spatial_size = output_h * output_w;
    
    for (int n = 0; n < batch_size; ++n) {
        const float* input_batch = input + n * in_channels * spatial_size;
        float* output_batch = output + n * out_channels * spatial_size;
        
        // Use SIMD-optimized GEMM for 1x1 convolution
        gemm_vectorized(weight, input_batch, output_batch, 
                       out_channels, spatial_size, in_channels, 1.0f, 0.0f, false, false);
    }
#elif defined(__ARM_NEON)
    constexpr int SIMD_WIDTH = 4;
    const int spatial_size = output_h * output_w;
    
    for (int n = 0; n < batch_size; ++n) {
        const float* input_batch = input + n * in_channels * spatial_size;
        float* output_batch = output + n * out_channels * spatial_size;
        
        // NEON-optimized matrix multiplication
        for (int oc = 0; oc < out_channels; ++oc) {
            const float* weight_row = weight + oc * in_channels;
            float* output_row = output_batch + oc * spatial_size;
            
            std::fill(output_row, output_row + spatial_size, 0.0f);
            
            for (int ic = 0; ic < in_channels; ++ic) {
                const float* input_channel = input_batch + ic * spatial_size;
                float32x4_t weight_vec = vdupq_n_f32(weight_row[ic]);
                
                for (int pos = 0; pos < spatial_size; pos += SIMD_WIDTH) {
                    int remaining = std::min(SIMD_WIDTH, spatial_size - pos);
                    
                    if (remaining == SIMD_WIDTH) {
                        float32x4_t input_vec = vld1q_f32(&input_channel[pos]);
                        float32x4_t output_vec = vld1q_f32(&output_row[pos]);
                        output_vec = vmlaq_f32(output_vec, input_vec, weight_vec);
                        vst1q_f32(&output_row[pos], output_vec);
                    } else {
                        // Scalar fallback for remaining elements
                        for (int i = 0; i < remaining; ++i) {
                            output_row[pos + i] += input_channel[pos + i] * weight_row[ic];
                        }
                    }
                }
            }
        }
    }
#else
    // Fallback to optimized scalar implementation
    conv2d_1x1_optimized(input, weight, output, batch_size, in_channels, out_channels,
                         input_h, input_w, output_h, output_w);
#endif
}

/**
 * @brief SIMD-optimized GEMM implementation using platform-specific intrinsics
 */
void MatrixOps::gemm_vectorized(const float* a, const float* b, float* c,
                               int m, int n, int k, float alpha, float beta,
                               bool transpose_a, bool transpose_b) {
#ifdef __AVX2__
    constexpr int SIMD_WIDTH = 8;
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 256;
    
    for (int i = 0; i < m; i += BLOCK_M) {
        int end_i = std::min(i + BLOCK_M, m);
        
        for (int j = 0; j < n; j += BLOCK_N) {
            int end_j = std::min(j + BLOCK_N, n);
            
            // Initialize C block
            for (int ii = i; ii < end_i; ++ii) {
                for (int jj = j; jj < end_j; jj += SIMD_WIDTH) {
                    int width = std::min(SIMD_WIDTH, end_j - jj);
                    if (width == SIMD_WIDTH) {
                        __m256 c_vec = _mm256_loadu_ps(&c[ii * n + jj]);
                        c_vec = _mm256_mul_ps(c_vec, _mm256_set1_ps(beta));
                        _mm256_storeu_ps(&c[ii * n + jj], c_vec);
                    } else {
                        for (int w = 0; w < width; ++w) {
                            c[ii * n + jj + w] *= beta;
                        }
                    }
                }
            }
            
            for (int l = 0; l < k; l += BLOCK_K) {
                int end_l = std::min(l + BLOCK_K, k);
                
                for (int ii = i; ii < end_i; ++ii) {
                    for (int jj = j; jj < end_j; jj += SIMD_WIDTH) {
                        int width = std::min(SIMD_WIDTH, end_j - jj);
                        
                        if (width == SIMD_WIDTH) {
                            __m256 acc = _mm256_setzero_ps();
                            
                            for (int ll = l; ll < end_l; ++ll) {
                                float a_val = transpose_a ? a[ll * m + ii] : a[ii * k + ll];
                                __m256 a_vec = _mm256_set1_ps(a_val);
                                
                                __m256 b_vec;
                                if (transpose_b) {
                                    // Load non-contiguous elements for transposed B
                                    float b_vals[8];
                                    for (int w = 0; w < 8; ++w) {
                                        b_vals[w] = b[(jj + w) * k + ll];
                                    }
                                    b_vec = _mm256_loadu_ps(b_vals);
                                } else {
                                    b_vec = _mm256_loadu_ps(&b[ll * n + jj]);
                                }
                                
                                acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
                            }
                            
                            __m256 c_vec = _mm256_loadu_ps(&c[ii * n + jj]);
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(alpha), acc, c_vec);
                            _mm256_storeu_ps(&c[ii * n + jj], acc);
                        } else {
                            // Scalar fallback for partial blocks
                            for (int w = 0; w < width; ++w) {
                                float acc = 0.0f;
                                for (int ll = l; ll < end_l; ++ll) {
                                    float a_val = transpose_a ? a[ll * m + ii] : a[ii * k + ll];
                                    float b_val = transpose_b ? b[(jj + w) * k + ll] : b[ll * n + (jj + w)];
                                    acc += a_val * b_val;
                                }
                                c[ii * n + jj + w] += alpha * acc;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    // Fallback to blocked implementation
    gemm_blocked(a, b, c, m, n, k, alpha, beta, transpose_a, transpose_b);
#endif
}

/**
 * @brief Multi-threaded convolution for large tensors
 */
void MatrixOps::conv2d_parallel(const float* input, const float* weight, float* output,
                               int batch_size, int in_channels, int out_channels,
                               int input_h, int input_w, int output_h, int output_w,
                               int kernel_h, int kernel_w, int stride_h, int stride_w,
                               int pad_h, int pad_w) {
    // Get number of available threads
    const int num_threads = std::thread::hardware_concurrency();
    if (num_threads <= 1) {
        // Fallback to single-threaded version
        conv2d_generic_optimized(input, weight, output, batch_size, in_channels, out_channels,
                                input_h, input_w, output_h, output_w,
                                kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
        return;
    }
    
    // Clear output tensor
    std::fill(output, output + batch_size * out_channels * output_h * output_w, 0.0f);
    
    // Parallel processing across output channels
    std::vector<std::thread> threads;
    const int channels_per_thread = (out_channels + num_threads - 1) / num_threads;
    
    for (int t = 0; t < num_threads; ++t) {
        const int start_channel = t * channels_per_thread;
        const int end_channel = std::min(start_channel + channels_per_thread, out_channels);
        
        if (start_channel >= out_channels) break;
        
        threads.emplace_back([=]() {
            // Process assigned output channels
            for (int n = 0; n < batch_size; ++n) {
                for (int oc = start_channel; oc < end_channel; ++oc) {
                    for (int oh = 0; oh < output_h; ++oh) {
                        for (int ow = 0; ow < output_w; ++ow) {
                            float sum = 0.0f;
                            
                            for (int ic = 0; ic < in_channels; ++ic) {
                                for (int kh = 0; kh < kernel_h; ++kh) {
                                    for (int kw = 0; kw < kernel_w; ++kw) {
                                        int ih = oh * stride_h - pad_h + kh;
                                        int iw = ow * stride_w - pad_w + kw;
                                        
                                        if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                                            int input_idx = n * in_channels * input_h * input_w +
                                                           ic * input_h * input_w + ih * input_w + iw;
                                            int weight_idx = oc * in_channels * kernel_h * kernel_w +
                                                            ic * kernel_h * kernel_w + kh * kernel_w + kw;
                                            sum += input[input_idx] * weight[weight_idx];
                                        }
                                    }
                                }
                            }
                            
                            int output_idx = n * out_channels * output_h * output_w +
                                            oc * output_h * output_w + oh * output_w + ow;
                            output[output_idx] = sum;
                        }
                    }
                }
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}

// Memory prefetching utilities for cache optimization
inline void MatrixOps::prefetch_read(const void* addr) {
#ifdef __builtin_prefetch
    __builtin_prefetch(addr, 0, 3);
#endif
}

inline void MatrixOps::prefetch_write(void* addr) {
#ifdef __builtin_prefetch
    __builtin_prefetch(addr, 1, 3);
#endif
}

/**
 * @brief Optimized im2col transformation for fast convolution computation
 * 
 * Transforms input tensor into matrix format suitable for GEMM-based convolution.
 * Critical optimization for achieving sub-50ms inference times with large kernels.
 * Uses cache-friendly memory access patterns and vectorized operations.
 */
void MatrixOps::im2col(const core::Tensor& input, core::Tensor& output,
                       int kernel_h, int kernel_w, int stride_h, int stride_w, 
                       int pad_h, int pad_w) {
    const auto& input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::runtime_error("im2col requires 4D input tensor (NCHW format)");
    }
    
    int batch_size = input_shape[0];
    int in_channels = input_shape[1];
    int input_h = input_shape[2];
    int input_w = input_shape[3];
    
    int output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    
    const auto& output_shape = output.shape();
    if (output_shape.size() != 2) {
        throw std::runtime_error("im2col output must be 2D matrix");
    }
    
    int expected_cols = output_h * output_w * batch_size;
    int expected_rows = in_channels * kernel_h * kernel_w;
    
    if (output_shape[0] != expected_rows || output_shape[1] != expected_cols) {
        throw std::runtime_error("im2col output dimensions incorrect");
    }
    
    const float* input_data = input.data_float();
    float* output_data = output.data_float();
    
    // Pre-compute strides for cache efficiency
    const int input_channel_stride = input_h * input_w;
    const int input_batch_stride = in_channels * input_channel_stride;
    const int spatial_size = output_h * output_w;
    
    // Parallelize over batches and output spatial positions
    for (int n = 0; n < batch_size; ++n) {
        const float* input_batch = input_data + n * input_batch_stride;
        
        for (int oh = 0; oh < output_h; ++oh) {
            for (int ow = 0; ow < output_w; ++ow) {
                int col_idx = n * spatial_size + oh * output_w + ow;
                
                for (int ic = 0; ic < in_channels; ++ic) {
                    const float* input_channel = input_batch + ic * input_channel_stride;
                    
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride_h - pad_h + kh;
                            int iw = ow * stride_w - pad_w + kw;
                            
                            int row_idx = ic * kernel_h * kernel_w + kh * kernel_w + kw;
                            
                            if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                                output_data[row_idx * expected_cols + col_idx] = 
                                    input_channel[ih * input_w + iw];
                            } else {
                                output_data[row_idx * expected_cols + col_idx] = 0.0f;
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Vectorized element-wise addition using SIMD instructions
 */
void MatrixOps::vectorized_add(const float* a, const float* b, float* c, int size) {
#ifdef __AVX2__
    constexpr int SIMD_WIDTH = 8;
    int vectorized_size = (size / SIMD_WIDTH) * SIMD_WIDTH;
    
    for (int i = 0; i < vectorized_size; i += SIMD_WIDTH) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&c[i], result);
    }
    
    // Handle remaining elements
    for (int i = vectorized_size; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
#elif defined(__ARM_NEON)
    constexpr int SIMD_WIDTH = 4;
    int vectorized_size = (size / SIMD_WIDTH) * SIMD_WIDTH;
    
    for (int i = 0; i < vectorized_size; i += SIMD_WIDTH) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t result = vaddq_f32(va, vb);
        vst1q_f32(&c[i], result);
    }
    
    for (int i = vectorized_size; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
#else
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
#endif
}

/**
 * @brief Vectorized element-wise multiplication using SIMD instructions
 */
void MatrixOps::vectorized_multiply(const float* a, const float* b, float* c, int size) {
#ifdef __AVX2__
    constexpr int SIMD_WIDTH = 8;
    int vectorized_size = (size / SIMD_WIDTH) * SIMD_WIDTH;
    
    for (int i = 0; i < vectorized_size; i += SIMD_WIDTH) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&c[i], result);
    }
    
    for (int i = vectorized_size; i < size; ++i) {
        c[i] = a[i] * b[i];
    }
#elif defined(__ARM_NEON)
    constexpr int SIMD_WIDTH = 4;
    int vectorized_size = (size / SIMD_WIDTH) * SIMD_WIDTH;
    
    for (int i = 0; i < vectorized_size; i += SIMD_WIDTH) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t result = vmulq_f32(va, vb);
        vst1q_f32(&c[i], result);
    }
    
    for (int i = vectorized_size; i < size; ++i) {
        c[i] = a[i] * b[i];
    }
#else
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] * b[i];
    }
#endif
}

/**
 * @brief Vectorized fused multiply-add using SIMD instructions
 */
void MatrixOps::vectorized_fma(const float* a, const float* b, float* c, int size) {
#ifdef __AVX2__
    constexpr int SIMD_WIDTH = 8;
    int vectorized_size = (size / SIMD_WIDTH) * SIMD_WIDTH;
    
    for (int i = 0; i < vectorized_size; i += SIMD_WIDTH) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_loadu_ps(&c[i]);
        __m256 result = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(&c[i], result);
    }
    
    for (int i = vectorized_size; i < size; ++i) {
        c[i] += a[i] * b[i];
    }
#elif defined(__ARM_NEON)
    constexpr int SIMD_WIDTH = 4;
    int vectorized_size = (size / SIMD_WIDTH) * SIMD_WIDTH;
    
    for (int i = 0; i < vectorized_size; i += SIMD_WIDTH) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vc = vld1q_f32(&c[i]);
        float32x4_t result = vmlaq_f32(vc, va, vb);
        vst1q_f32(&c[i], result);
    }
    
    for (int i = vectorized_size; i < size; ++i) {
        c[i] += a[i] * b[i];
    }
#else
    for (int i = 0; i < size; ++i) {
        c[i] += a[i] * b[i];
    }
#endif
}

}
}