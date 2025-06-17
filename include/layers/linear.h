// Linear (fully-connected) layer implementation
#pragma once

#include "../core/tensor.h"
#include "../core/tensor_ops.h"
#include <vector>
#include <random>
#include <cmath>
#ifdef __x86_64__
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace tacs {

using core::Tensor;

class LinearLayer {
public:
    LinearLayer(int inputSize, int outputSize)
        : inputSize_(inputSize)
        , outputSize_(outputSize) {
        
        // Initialize weights and bias
        weights_.resize(outputSize * inputSize);
        bias_.resize(outputSize);
        
        // Xavier initialization
        float scale = std::sqrt(2.0f / inputSize);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        
        for (auto& w : weights_) {
            w = dist(gen);
        }
        
        std::fill(bias_.begin(), bias_.end(), 0.0f);
    }
    
    ~LinearLayer() = default;
    
    // Forward pass: y = Wx + b
    Tensor forward(const Tensor& input) const {
        int batchSize = input.shape()[0];
        Tensor output({batchSize, outputSize_});
        
        // Optimized matrix multiplication with SIMD
        #pragma omp parallel for
        for (int b = 0; b < batchSize; ++b) {
            for (int o = 0; o < outputSize_; ++o) {
                float sum = bias_[o];
                
                // SIMD vectorized dot product
                int i = 0;
                #ifdef __AVX2__
                __m256 vsum = _mm256_setzero_ps();
                for (; i + 7 < inputSize_; i += 8) {
                    __m256 vx = _mm256_loadu_ps(&input.data_float()[b * inputSize_ + i]);
                    __m256 vw = _mm256_loadu_ps(&weights_[o * inputSize_ + i]);
                    vsum = _mm256_fmadd_ps(vx, vw, vsum);
                }
                
                // Horizontal sum
                __m128 vlow = _mm256_castps256_ps128(vsum);
                __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
                vlow = _mm_add_ps(vlow, vhigh);
                
                __m128 shuf = _mm_movehdup_ps(vlow);
                __m128 sums = _mm_add_ps(vlow, shuf);
                shuf = _mm_movehl_ps(shuf, sums);
                sums = _mm_add_ss(sums, shuf);
                sum += _mm_cvtss_f32(sums);
                #elif defined(__ARM_NEON)
                float32x4_t vsum = vdupq_n_f32(0.0f);
                for (; i + 3 < inputSize_; i += 4) {
                    float32x4_t vx = vld1q_f32(&input.data_float()[b * inputSize_ + i]);
                    float32x4_t vw = vld1q_f32(&weights_[o * inputSize_ + i]);
                    vsum = vmlaq_f32(vsum, vx, vw);
                }
                // Horizontal sum
                float32x2_t vlow = vget_low_f32(vsum);
                float32x2_t vhigh = vget_high_f32(vsum);
                vlow = vadd_f32(vlow, vhigh);
                sum += vget_lane_f32(vlow, 0) + vget_lane_f32(vlow, 1);
                #endif
                
                // Handle remaining elements
                for (; i < inputSize_; ++i) {
                    sum += input.data_float()[b * inputSize_ + i] * weights_[o * inputSize_ + i];
                }
                
                output.data_float()[b * outputSize_ + o] = sum;
            }
        }
        
        return output;
    }
    
    // Backward pass for gradient computation
    Tensor backward(const Tensor& gradOutput, const Tensor& input) {
        int batchSize = input.shape()[0];
        
        // Gradient w.r.t. input: dL/dx = W^T * dL/dy
        Tensor gradInput({batchSize, inputSize_});
        
        #pragma omp parallel for
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < inputSize_; ++i) {
                float sum = 0.0f;
                
                for (int o = 0; o < outputSize_; ++o) {
                    sum += weights_[o * inputSize_ + i] * gradOutput.data_float()[b * outputSize_ + o];
                }
                
                gradInput.data_float()[b * inputSize_ + i] = sum;
            }
        }
        
        // Compute weight gradients: dL/dW = dL/dy * x^T
        weightGrad_.resize(weights_.size(), 0.0f);
        biasGrad_.resize(bias_.size(), 0.0f);
        
        for (int b = 0; b < batchSize; ++b) {
            for (int o = 0; o < outputSize_; ++o) {
                float gradOut = gradOutput.data_float()[b * outputSize_ + o];
                
                // Bias gradient
                biasGrad_[o] += gradOut;
                
                // Weight gradient
                for (int i = 0; i < inputSize_; ++i) {
                    weightGrad_[o * inputSize_ + i] += gradOut * input.data_float()[b * inputSize_ + i];
                }
            }
        }
        
        // Average over batch
        float invBatch = 1.0f / batchSize;
        for (auto& g : weightGrad_) g *= invBatch;
        for (auto& g : biasGrad_) g *= invBatch;
        
        return gradInput;
    }
    
    // Update weights with gradients
    void updateWeights(float learningRate) {
        for (size_t i = 0; i < weights_.size(); ++i) {
            weights_[i] -= learningRate * weightGrad_[i];
        }
        
        for (size_t i = 0; i < bias_.size(); ++i) {
            bias_[i] -= learningRate * biasGrad_[i];
        }
    }
    
    // Getters and setters
    std::vector<float>& getWeights() { return weights_; }
    const std::vector<float>& getWeights() const { return weights_; }
    
    std::vector<float>& getBias() { return bias_; }
    const std::vector<float>& getBias() const { return bias_; }
    
    void setWeights(const std::vector<float>& weights) { weights_ = weights; }
    void setBias(const std::vector<float>& bias) { bias_ = bias; }
    
    int getInputSize() const { return inputSize_; }
    int getOutputSize() const { return outputSize_; }
    
    const std::vector<float>& getWeightGrad() const { return weightGrad_; }
    const std::vector<float>& getBiasGrad() const { return biasGrad_; }
    
private:
    int inputSize_;
    int outputSize_;
    std::vector<float> weights_;  // Shape: [outputSize, inputSize]
    std::vector<float> bias_;     // Shape: [outputSize]
    
    // Gradient storage
    mutable std::vector<float> weightGrad_;
    mutable std::vector<float> biasGrad_;
};

} // namespace tacs