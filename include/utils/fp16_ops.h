// FP16 Operations - Manual half-precision implementation for TACS
// Production-ready FP16 quantization without external dependencies

#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include "core/tensor.h"

namespace tacs {
namespace utils {

// Manual FP16 representation using uint16_t
using fp16_t = uint16_t;

// FP16 conversion utilities
class FP16Ops {
public:
    // Convert float32 to fp16
    static fp16_t float_to_fp16(float value) {
        uint32_t f32_bits = *reinterpret_cast<uint32_t*>(&value);
        
        // Extract components
        uint32_t sign = (f32_bits >> 31) & 0x1;
        int32_t exponent = ((f32_bits >> 23) & 0xFF) - 127;
        uint32_t mantissa = f32_bits & 0x7FFFFF;
        
        // Handle special cases
        if (exponent == 128) {  // Infinity or NaN
            if (mantissa == 0) {
                // Infinity
                return (sign << 15) | 0x7C00;
            } else {
                // NaN
                return (sign << 15) | 0x7C00 | (mantissa >> 13);
            }
        }
        
        // Clamp to FP16 range
        if (exponent < -14) {
            // Too small for normalized FP16, use denormalized
            if (exponent >= -24) {
                mantissa |= 0x800000;  // Add implicit 1
                int shift = -exponent - 14 + 23;
                mantissa >>= shift;
                return (sign << 15) | mantissa;
            } else {
                // Too small even for denormalized, return zero
                return sign << 15;
            }
        } else if (exponent > 15) {
            // Too large, return infinity
            return (sign << 15) | 0x7C00;
        }
        
        // Normal case
        uint16_t fp16_exp = (exponent + 15) & 0x1F;
        uint16_t fp16_mantissa = (mantissa >> 13) & 0x3FF;
        
        return (sign << 15) | (fp16_exp << 10) | fp16_mantissa;
    }
    
    // Convert fp16 to float32
    static float fp16_to_float(fp16_t value) {
        uint32_t sign = (value >> 15) & 0x1;
        uint32_t exponent = (value >> 10) & 0x1F;
        uint32_t mantissa = value & 0x3FF;
        
        // Handle special cases
        if (exponent == 0) {
            if (mantissa == 0) {
                // Zero
                uint32_t f32_bits = sign << 31;
                return *reinterpret_cast<float*>(&f32_bits);
            } else {
                // Denormalized
                float result = mantissa / 16384.0f;
                return sign ? -result : result;
            }
        } else if (exponent == 31) {
            // Infinity or NaN
            uint32_t f32_bits = (sign << 31) | 0x7F800000;
            if (mantissa != 0) {
                f32_bits |= (mantissa << 13);
            }
            return *reinterpret_cast<float*>(&f32_bits);
        }
        
        // Normal case
        uint32_t f32_exp = (exponent - 15 + 127) & 0xFF;
        uint32_t f32_mantissa = mantissa << 13;
        uint32_t f32_bits = (sign << 31) | (f32_exp << 23) | f32_mantissa;
        
        return *reinterpret_cast<float*>(&f32_bits);
    }
    
    // FP16 matrix multiplication
    static void gemm_fp16(const fp16_t* a, const fp16_t* b, float* c,
                         int m, int n, int k,
                         float alpha = 1.0f, float beta = 0.0f) {
        // Optimized FP16 GEMM with mixed precision accumulation
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                
                // Accumulate in FP32 for accuracy
                for (int l = 0; l < k; ++l) {
                    float a_val = fp16_to_float(a[i * k + l]);
                    float b_val = fp16_to_float(b[l * n + j]);
                    sum += a_val * b_val;
                }
                
                // Apply alpha and beta
                c[i * n + j] = alpha * sum + beta * c[i * n + j];
            }
        }
    }
    
    // Optimized FP16 convolution
    static void conv2d_fp16(const fp16_t* input, const fp16_t* weight,
                           float* output, const fp16_t* bias,
                           int batch, int in_channels, int out_channels,
                           int height, int width,
                           int kernel_h, int kernel_w,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w) {
        int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
        
        // Process each output position
        for (int b = 0; b < batch; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                // Add bias if provided
                float bias_val = bias ? fp16_to_float(bias[oc]) : 0.0f;
                
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        float sum = bias_val;
                        
                        // Convolution kernel
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    int ih = oh * stride_h - pad_h + kh;
                                    int iw = ow * stride_w - pad_w + kw;
                                    
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                        int input_idx = b * in_channels * height * width +
                                                       ic * height * width + ih * width + iw;
                                        int weight_idx = oc * in_channels * kernel_h * kernel_w +
                                                        ic * kernel_h * kernel_w + kh * kernel_w + kw;
                                        
                                        float in_val = fp16_to_float(input[input_idx]);
                                        float w_val = fp16_to_float(weight[weight_idx]);
                                        sum += in_val * w_val;
                                    }
                                }
                            }
                        }
                        
                        int output_idx = b * out_channels * out_h * out_w +
                                        oc * out_h * out_w + oh * out_w + ow;
                        output[output_idx] = sum;
                    }
                }
            }
        }
    }
    
    // Convert tensor to FP16
    static std::vector<fp16_t> tensor_to_fp16(const core::Tensor& tensor) {
        const float* data = tensor.data_float();
        std::vector<fp16_t> fp16_data(tensor.size());
        
        for (int i = 0; i < tensor.size(); ++i) {
            fp16_data[i] = float_to_fp16(data[i]);
        }
        
        return fp16_data;
    }
    
    // Convert FP16 to tensor
    static core::Tensor fp16_to_tensor(const std::vector<fp16_t>& fp16_data,
                                      const std::vector<int>& shape) {
        core::Tensor tensor(shape);
        float* data = tensor.data_float();
        
        for (size_t i = 0; i < fp16_data.size(); ++i) {
            data[i] = fp16_to_float(fp16_data[i]);
        }
        
        return tensor;
    }
};

} // namespace utils
} // namespace tacs