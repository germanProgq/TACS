/**
 * @file onnx_simd_kernels.h
 * @brief SIMD-optimized kernels for ONNX runtime operations
 * 
 * Production-ready SIMD implementations for maximum performance on CPU inference.
 * Supports SSE, AVX2, and ARM NEON with automatic detection and dispatch.
 */
#pragma once

#include "core/tensor.h"
#include <cstddef>
#include <vector>

#ifdef __x86_64__
#include <immintrin.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace tacs {
namespace utils {
namespace simd {

enum class SimdCapability {
    NONE = 0,
    SSE2 = 1,
    SSE4_1 = 2,
    AVX = 3,
    AVX2 = 4,
    AVX512 = 5,
    NEON = 6
};

class SimdDetector {
public:
    static SimdCapability detect_capability();
    static bool has_sse2();
    static bool has_sse4_1();
    static bool has_avx();
    static bool has_avx2();
    static bool has_avx512();
    static bool has_neon();
};

class SimdConv2D {
public:
    static void compute(
        const float* input, const float* weight, const float* bias,
        float* output,
        int batch, int in_channels, int out_channels,
        int in_height, int in_width,
        int kernel_height, int kernel_width,
        int out_height, int out_width,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        SimdCapability simd_cap = SimdCapability::NONE
    );
    
private:
    static void compute_avx2(
        const float* input, const float* weight, const float* bias,
        float* output,
        int batch, int in_channels, int out_channels,
        int in_height, int in_width,
        int kernel_height, int kernel_width,
        int out_height, int out_width,
        int stride_h, int stride_w,
        int pad_h, int pad_w
    );
    
    static void compute_neon(
        const float* input, const float* weight, const float* bias,
        float* output,
        int batch, int in_channels, int out_channels,
        int in_height, int in_width,
        int kernel_height, int kernel_width,
        int out_height, int out_width,
        int stride_h, int stride_w,
        int pad_h, int pad_w
    );
    
    static void compute_scalar(
        const float* input, const float* weight, const float* bias,
        float* output,
        int batch, int in_channels, int out_channels,
        int in_height, int in_width,
        int kernel_height, int kernel_width,
        int out_height, int out_width,
        int stride_h, int stride_w,
        int pad_h, int pad_w
    );
};

class SimdBatchNorm {
public:
    static void compute(
        const float* input, const float* scale, const float* bias,
        const float* mean, const float* var,
        float* output,
        int batch, int channels, int spatial_size,
        float epsilon = 1e-5f,
        SimdCapability simd_cap = SimdCapability::NONE
    );
    
private:
    static void compute_avx2(
        const float* input, const float* scale, const float* bias,
        const float* mean, const float* var,
        float* output,
        int batch, int channels, int spatial_size,
        float epsilon
    );
    
    static void compute_neon(
        const float* input, const float* scale, const float* bias,
        const float* mean, const float* var,
        float* output,
        int batch, int channels, int spatial_size,
        float epsilon
    );
};

class SimdActivations {
public:
    static void relu(const float* input, float* output, size_t size,
                     SimdCapability simd_cap = SimdCapability::NONE);
    
    static void leaky_relu(const float* input, float* output, size_t size,
                           float alpha = 0.1f,
                           SimdCapability simd_cap = SimdCapability::NONE);
    
    static void sigmoid(const float* input, float* output, size_t size,
                        SimdCapability simd_cap = SimdCapability::NONE);
    
    static void tanh(const float* input, float* output, size_t size,
                     SimdCapability simd_cap = SimdCapability::NONE);
    
private:
    static void relu_avx2(const float* input, float* output, size_t size);
    static void relu_neon(const float* input, float* output, size_t size);
    
    static void leaky_relu_avx2(const float* input, float* output, size_t size, float alpha);
    static void leaky_relu_neon(const float* input, float* output, size_t size, float alpha);
    
    static void sigmoid_avx2(const float* input, float* output, size_t size);
    static void sigmoid_neon(const float* input, float* output, size_t size);
    
    static void tanh_avx2(const float* input, float* output, size_t size);
    static void tanh_neon(const float* input, float* output, size_t size);
};

class SimdPooling {
public:
    static void max_pool_2d(
        const float* input, float* output,
        int batch, int channels,
        int in_height, int in_width,
        int out_height, int out_width,
        int kernel_h, int kernel_w,
        int stride_h, int stride_w,
        SimdCapability simd_cap = SimdCapability::NONE
    );
    
private:
    static void max_pool_2d_avx2(
        const float* input, float* output,
        int batch, int channels,
        int in_height, int in_width,
        int out_height, int out_width,
        int kernel_h, int kernel_w,
        int stride_h, int stride_w
    );
    
    static void max_pool_2d_neon(
        const float* input, float* output,
        int batch, int channels,
        int in_height, int in_width,
        int out_height, int out_width,
        int kernel_h, int kernel_w,
        int stride_h, int stride_w
    );
};

class SimdElementwise {
public:
    static void add(const float* a, const float* b, float* output, size_t size,
                    SimdCapability simd_cap = SimdCapability::NONE);
    
    static void multiply(const float* a, const float* b, float* output, size_t size,
                         SimdCapability simd_cap = SimdCapability::NONE);
    
    static void add_scalar(const float* input, float scalar, float* output, size_t size,
                           SimdCapability simd_cap = SimdCapability::NONE);
    
    static void multiply_scalar(const float* input, float scalar, float* output, size_t size,
                                SimdCapability simd_cap = SimdCapability::NONE);
    
private:
    static void add_avx2(const float* a, const float* b, float* output, size_t size);
    static void add_neon(const float* a, const float* b, float* output, size_t size);
    
    static void multiply_avx2(const float* a, const float* b, float* output, size_t size);
    static void multiply_neon(const float* a, const float* b, float* output, size_t size);
};

class SimdGemm {
public:
    static void gemm(
        const float* A, const float* B, float* C,
        int M, int N, int K,
        float alpha = 1.0f, float beta = 0.0f,
        bool transA = false, bool transB = false,
        SimdCapability simd_cap = SimdCapability::NONE
    );
    
private:
    static void gemm_avx2(
        const float* A, const float* B, float* C,
        int M, int N, int K,
        float alpha, float beta,
        bool transA, bool transB
    );
    
    static void gemm_neon(
        const float* A, const float* B, float* C,
        int M, int N, int K,
        float alpha, float beta,
        bool transA, bool transB
    );
};

class SimdQuantization {
public:
    static void quantize_int8(
        const float* input, int8_t* output, size_t size,
        float scale, int zero_point,
        SimdCapability simd_cap = SimdCapability::NONE
    );
    
    static void dequantize_int8(
        const int8_t* input, float* output, size_t size,
        float scale, int zero_point,
        SimdCapability simd_cap = SimdCapability::NONE
    );
    
    static void quantize_fp16(
        const float* input, uint16_t* output, size_t size,
        SimdCapability simd_cap = SimdCapability::NONE
    );
    
    static void dequantize_fp16(
        const uint16_t* input, float* output, size_t size,
        SimdCapability simd_cap = SimdCapability::NONE
    );
    
private:
    static void quantize_int8_avx2(
        const float* input, int8_t* output, size_t size,
        float scale, int zero_point
    );
    
    static void quantize_int8_neon(
        const float* input, int8_t* output, size_t size,
        float scale, int zero_point
    );
};

}
}
}