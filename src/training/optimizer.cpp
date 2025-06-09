#include "training/optimizer.h"
#include <cmath>
#include <algorithm>

namespace tacs {
namespace training {

SGDOptimizer::SGDOptimizer(float learning_rate, float momentum, float weight_decay)
    : learning_rate_(learning_rate), momentum_(momentum), weight_decay_(weight_decay) {}

void SGDOptimizer::step() {
    // In production systems, parameter updates would be handled through
    // the model's apply_gradients method which is already implemented
    // The optimizer maintains state for momentum-based updates
}

void SGDOptimizer::zero_grad() {
    // Clear velocity terms for momentum
    for (auto& [name, vel] : velocity_) {
        vel.zero();
    }
}

void SGDOptimizer::update_parameter(const std::string& param_name, 
                                    const core::Tensor& gradient, 
                                    core::Tensor& parameter) {
    if (gradient.dtype() != core::DataType::FLOAT32 || parameter.dtype() != core::DataType::FLOAT32) {
        throw std::runtime_error("SGD optimizer only supports FLOAT32 tensors");
    }
    
    if (gradient.size() != parameter.size()) {
        throw std::runtime_error("Gradient and parameter sizes must match");
    }
    
    // Initialize velocity if not exists
    if (velocity_.find(param_name) == velocity_.end()) {
        velocity_.emplace(param_name, core::Tensor(parameter.shape()));
        velocity_[param_name].zero();
    }
    
    const float* grad_data = gradient.data_float();
    float* param_data = parameter.data_float();
    float* vel_data = velocity_[param_name].data_float();
    
    size_t size = parameter.size();
    
    if (momentum_ > 0.0f) {
        // SGD with momentum: v = momentum * v + (1 - dampening) * g
        // param = param - lr * v
        for (size_t i = 0; i < size; ++i) {
            float grad_val = grad_data[i];
            
            // Apply weight decay if specified
            if (weight_decay_ > 0.0f) {
                grad_val += weight_decay_ * param_data[i];
            }
            
            // Update velocity with momentum
            vel_data[i] = momentum_ * vel_data[i] + grad_val;
            
            // Update parameter
            param_data[i] -= learning_rate_ * vel_data[i];
        }
    } else {
        // Standard SGD without momentum
        for (size_t i = 0; i < size; ++i) {
            float grad_val = grad_data[i];
            
            // Apply weight decay if specified
            if (weight_decay_ > 0.0f) {
                grad_val += weight_decay_ * param_data[i];
            }
            
            // Update parameter
            param_data[i] -= learning_rate_ * grad_val;
        }
    }
}

AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2, float eps, float weight_decay)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps), 
      weight_decay_(weight_decay), step_count_(0) {}

void AdamOptimizer::step() {
    step_count_++;
}

void AdamOptimizer::zero_grad() {
    // Adam maintains first and second moment estimates
    // These are preserved across gradient steps
}

void AdamOptimizer::update_parameter(const std::string& param_name,
                                     const core::Tensor& gradient,
                                     core::Tensor& parameter) {
    if (gradient.dtype() != core::DataType::FLOAT32 || parameter.dtype() != core::DataType::FLOAT32) {
        throw std::runtime_error("Adam optimizer only supports FLOAT32 tensors");
    }
    
    if (gradient.size() != parameter.size()) {
        throw std::runtime_error("Gradient and parameter sizes must match");
    }
    
    // Initialize moments if not exist
    if (m_.find(param_name) == m_.end()) {
        m_.emplace(param_name, core::Tensor(parameter.shape()));
        m_[param_name].zero();
    }
    
    if (v_.find(param_name) == v_.end()) {
        v_.emplace(param_name, core::Tensor(parameter.shape()));
        v_[param_name].zero();
    }
    
    const float* grad_data = gradient.data_float();
    float* param_data = parameter.data_float();
    float* m_data = m_[param_name].data_float();
    float* v_data = v_[param_name].data_float();
    
    size_t size = parameter.size();
    
    // Bias correction factors
    float bias_correction1 = 1.0f - std::pow(beta1_, step_count_);
    float bias_correction2 = 1.0f - std::pow(beta2_, step_count_);
    
    for (size_t i = 0; i < size; ++i) {
        float grad_val = grad_data[i];
        
        // Apply weight decay (L2 regularization)
        if (weight_decay_ > 0.0f) {
            grad_val += weight_decay_ * param_data[i];
        }
        
        // Update biased first moment estimate
        m_data[i] = beta1_ * m_data[i] + (1.0f - beta1_) * grad_val;
        
        // Update biased second raw moment estimate
        v_data[i] = beta2_ * v_data[i] + (1.0f - beta2_) * grad_val * grad_val;
        
        // Compute bias-corrected first moment estimate
        float m_hat = m_data[i] / bias_correction1;
        
        // Compute bias-corrected second raw moment estimate
        float v_hat = v_data[i] / bias_correction2;
        
        // Update parameter with numerical stability
        param_data[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + eps_);
    }
}

void AdamOptimizer::reset() {
    step_count_ = 0;
    m_.clear();
    v_.clear();
}

void SGDOptimizer::reset() {
    velocity_.clear();
}

}
}