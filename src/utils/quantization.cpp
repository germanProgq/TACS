#include "utils/quantization.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace tacs {
namespace utils {

uint32_t FP16Quantization::as_uint(float f) {
    union { float f; uint32_t i; } u;
    u.f = f;
    return u.i;
}

float FP16Quantization::as_float(uint32_t i) {
    union { uint32_t i; float f; } u;
    u.i = i;
    return u.f;
}

fp16_t FP16Quantization::float_to_half(float value) {
    uint32_t f32 = as_uint(value);
    uint16_t f16 = 0;
    
    uint32_t sign = (f32 >> 31) & 0x1;
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127;
    uint32_t mantissa = f32 & 0x7FFFFF;
    
    // Handle special cases
    if (exponent == 128) {
        // Infinity or NaN
        f16 = (sign << 15) | 0x7C00;
        if (mantissa != 0) {
            // NaN
            f16 |= (mantissa >> 13);
        }
        return f16;
    }
    
    // Handle zero and denormals
    if (exponent < -14) {
        // Too small for FP16, return signed zero
        return sign << 15;
    }
    
    // Handle overflow
    if (exponent > 15) {
        // Too large for FP16, return signed infinity
        return (sign << 15) | 0x7C00;
    }
    
    // Normal FP16 range
    if (exponent >= -14) {
        // Normal number
        f16 = (sign << 15) | ((exponent + 15) << 10) | (mantissa >> 13);
    } else {
        // Denormal number
        int shift = -exponent - 14;
        mantissa |= 0x800000;  // Add implicit 1
        f16 = (sign << 15) | (mantissa >> (13 + shift));
    }
    
    return f16;
}

float FP16Quantization::half_to_float(fp16_t value) {
    uint32_t sign = (value >> 15) & 0x1;
    uint32_t exponent = (value >> 10) & 0x1F;
    uint32_t mantissa = value & 0x3FF;
    
    uint32_t f32 = sign << 31;
    
    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            return as_float(f32);
        } else {
            // Denormal FP16
            exponent = 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
        }
    } else if (exponent == 31) {
        // Infinity or NaN
        f32 |= 0x7F800000;
        if (mantissa != 0) {
            // NaN
            f32 |= mantissa << 13;
        }
        return as_float(f32);
    }
    
    // Normal number
    exponent = exponent - 15 + 127;
    f32 |= (exponent << 23) | (mantissa << 13);
    
    return as_float(f32);
}

void FP16Quantization::quantize_tensor(const core::Tensor& input, std::vector<fp16_t>& output) {
    const float* input_data = input.data_float();
    size_t size = input.size();
    
    output.resize(size);
    
    // Vectorized conversion with manual unrolling for performance
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        output[i] = float_to_half(input_data[i]);
        output[i + 1] = float_to_half(input_data[i + 1]);
        output[i + 2] = float_to_half(input_data[i + 2]);
        output[i + 3] = float_to_half(input_data[i + 3]);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        output[i] = float_to_half(input_data[i]);
    }
}

void FP16Quantization::dequantize_tensor(const std::vector<fp16_t>& input, core::Tensor& output) {
    float* output_data = output.data_float();
    size_t size = input.size();
    
    // Vectorized conversion with manual unrolling
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        output_data[i] = half_to_float(input[i]);
        output_data[i + 1] = half_to_float(input[i + 1]);
        output_data[i + 2] = half_to_float(input[i + 2]);
        output_data[i + 3] = half_to_float(input[i + 3]);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        output_data[i] = half_to_float(input[i]);
    }
}

void FP16Quantization::conv2d_fp16(const std::vector<fp16_t>& input,
                                   const std::vector<fp16_t>& weights,
                                   std::vector<fp16_t>& output,
                                   int batch_size, int in_channels, int out_channels,
                                   int height, int width, int kernel_size,
                                   int stride, int padding) {
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    output.resize(batch_size * out_channels * out_height * out_width);
    std::fill(output.begin(), output.end(), float_to_half(0.0f));
    
    // Optimized convolution with FP16 arithmetic
    for (int n = 0; n < batch_size; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    int input_idx = n * in_channels * height * width +
                                                   ic * height * width + ih * width + iw;
                                    int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                                    ic * kernel_size * kernel_size + kh * kernel_size + kw;
                                    
                                    // Convert to float for computation, then back to FP16
                                    float input_val = half_to_float(input[input_idx]);
                                    float weight_val = half_to_float(weights[weight_idx]);
                                    sum += input_val * weight_val;
                                }
                            }
                        }
                    }
                    
                    int output_idx = n * out_channels * out_height * out_width +
                                    oc * out_height * out_width + oh * out_width + ow;
                    output[output_idx] = float_to_half(sum);
                }
            }
        }
    }
}

void FP16Quantization::matmul_fp16(const std::vector<fp16_t>& a,
                                   const std::vector<fp16_t>& b,
                                   std::vector<fp16_t>& c,
                                   int m, int n, int k) {
    c.resize(m * n);
    
    // Blocked matrix multiplication for cache efficiency
    const int block_size = 64;
    
    for (int i = 0; i < m; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int p = 0; p < k; p += block_size) {
                // Process block
                int i_end = std::min(i + block_size, m);
                int j_end = std::min(j + block_size, n);
                int p_end = std::min(p + block_size, k);
                
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        float sum = (p == 0) ? 0.0f : half_to_float(c[ii * n + jj]);
                        
                        for (int pp = p; pp < p_end; ++pp) {
                            float a_val = half_to_float(a[ii * k + pp]);
                            float b_val = half_to_float(b[pp * n + jj]);
                            sum += a_val * b_val;
                        }
                        
                        c[ii * n + jj] = float_to_half(sum);
                    }
                }
            }
        }
    }
}

INT8Quantization::QuantizationParams INT8Quantization::calibrate(const core::Tensor& data) {
    const float* data_ptr = data.data_float();
    size_t size = data.size();
    
    // Find min and max values
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    
    for (size_t i = 0; i < size; ++i) {
        min_val = std::min(min_val, data_ptr[i]);
        max_val = std::max(max_val, data_ptr[i]);
    }
    
    // Calculate scale and zero point
    QuantizationParams params;
    params.scale = (max_val - min_val) / 255.0f;
    params.zero_point = static_cast<int8_t>(std::round(-min_val / params.scale - 128));
    
    return params;
}

void INT8Quantization::quantize_tensor(const core::Tensor& input,
                                      std::vector<int8_t>& output,
                                      const QuantizationParams& params) {
    const float* input_data = input.data_float();
    size_t size = input.size();
    
    output.resize(size);
    
    for (size_t i = 0; i < size; ++i) {
        int32_t quantized = std::round(input_data[i] / params.scale) + params.zero_point;
        quantized = std::clamp(quantized, -128, 127);
        output[i] = static_cast<int8_t>(quantized);
    }
}

void INT8Quantization::dequantize_tensor(const std::vector<int8_t>& input,
                                        core::Tensor& output,
                                        const QuantizationParams& params) {
    float* output_data = output.data_float();
    size_t size = input.size();
    
    for (size_t i = 0; i < size; ++i) {
        output_data[i] = params.scale * (input[i] - params.zero_point);
    }
}

void INT8Quantization::conv2d_int8(const std::vector<int8_t>& input,
                                  const std::vector<int8_t>& weights,
                                  std::vector<int32_t>& output,
                                  const QuantizationParams& input_params,
                                  const QuantizationParams& weight_params,
                                  int batch_size, int in_channels, int out_channels,
                                  int height, int width, int kernel_size,
                                  int stride, int padding) {
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    output.resize(batch_size * out_channels * out_height * out_width);
    std::fill(output.begin(), output.end(), 0);
    
    // INT8 convolution with INT32 accumulation
    for (int n = 0; n < batch_size; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    int32_t sum = 0;
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    int input_idx = n * in_channels * height * width +
                                                   ic * height * width + ih * width + iw;
                                    int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                                    ic * kernel_size * kernel_size + kh * kernel_size + kw;
                                    
                                    sum += static_cast<int32_t>(input[input_idx]) * 
                                           static_cast<int32_t>(weights[weight_idx]);
                                }
                            }
                        }
                    }
                    
                    int output_idx = n * out_channels * out_height * out_width +
                                    oc * out_height * out_width + oh * out_width + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

}
}