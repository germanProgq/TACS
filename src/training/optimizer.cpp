#include "training/optimizer.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace tacs {
namespace training {

SGDOptimizer::SGDOptimizer(float learning_rate, float momentum, float weight_decay)
    : learning_rate_(learning_rate), momentum_(momentum), weight_decay_(weight_decay) {
    if (learning_rate <= 0.0f) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (momentum < 0.0f || momentum >= 1.0f) {
        throw std::invalid_argument("Momentum must be in range [0, 1)");
    }
    if (weight_decay < 0.0f) {
        throw std::invalid_argument("Weight decay must be non-negative");
    }
}

void SGDOptimizer::step() {
    // Apply parameter updates with momentum and weight decay
    for (auto& [name, param_info] : parameters_) {
        if (param_info.gradient.size() == 0) {
            continue; // Skip parameters without gradients
        }
        
        update_parameter(name, param_info.gradient, *param_info.parameter);
    }
}

void SGDOptimizer::zero_grad() {
    // Clear all gradients
    for (auto& [name, param_info] : parameters_) {
        if (param_info.gradient.size() > 0) {
            param_info.gradient.zero();
        }
    }
    
    // Clear velocity terms for momentum
    for (auto& [name, vel] : velocity_) {
        vel.zero();
    }
}

void SGDOptimizer::add_parameter_group(const std::string& name, core::Tensor& parameter) {
    if (parameters_.find(name) != parameters_.end()) {
        throw std::runtime_error("Parameter " + name + " already exists");
    }
    
    ParameterInfo info;
    info.parameter = &parameter;  // Store pointer to the actual parameter
    info.gradient = core::Tensor(parameter.shape());
    info.gradient.zero();
    
    parameters_[name] = info;
    
    // Initialize velocity for momentum
    if (momentum_ > 0.0f) {
        velocity_.emplace(name, core::Tensor(parameter.shape()));
        velocity_[name].zero();
    }
}

void SGDOptimizer::set_gradient(const std::string& name, const core::Tensor& gradient) {
    auto it = parameters_.find(name);
    if (it == parameters_.end()) {
        throw std::runtime_error("Parameter " + name + " not found");
    }
    
    if (gradient.size() != it->second.parameter->size()) {
        throw std::runtime_error("Gradient size mismatch for parameter " + name);
    }
    
    // Copy gradient data
    const float* grad_data = gradient.data_float();
    float* stored_grad = it->second.gradient.data_float();
    size_t size = gradient.size();
    
    for (size_t i = 0; i < size; ++i) {
        stored_grad[i] = grad_data[i];
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
    if (momentum_ > 0.0f && velocity_.find(param_name) == velocity_.end()) {
        velocity_.emplace(param_name, core::Tensor(parameter.shape()));
        velocity_[param_name].zero();
    }
    
    const float* grad_data = gradient.data_float();
    float* param_data = parameter.data_float();
    size_t size = parameter.size();
    
    if (momentum_ > 0.0f) {
        float* vel_data = velocity_[param_name].data_float();
        
        // SGD with momentum: v = momentum * v + lr * (g + weight_decay * param)
        // param = param - v
        for (size_t i = 0; i < size; ++i) {
            float grad_val = grad_data[i];
            
            // Apply weight decay (L2 regularization)
            if (weight_decay_ > 0.0f) {
                grad_val += weight_decay_ * param_data[i];
            }
            
            // Update velocity with momentum
            vel_data[i] = momentum_ * vel_data[i] + learning_rate_ * grad_val;
            
            // Update parameter
            param_data[i] -= vel_data[i];
        }
    } else {
        // Standard SGD without momentum
        for (size_t i = 0; i < size; ++i) {
            float grad_val = grad_data[i];
            
            // Apply weight decay
            if (weight_decay_ > 0.0f) {
                grad_val += weight_decay_ * param_data[i];
            }
            
            // Update parameter
            param_data[i] -= learning_rate_ * grad_val;
        }
    }
}

float SGDOptimizer::get_learning_rate() const {
    return learning_rate_;
}

void SGDOptimizer::set_learning_rate(float lr) {
    if (lr <= 0.0f) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    learning_rate_ = lr;
}

AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2, float eps, float weight_decay)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps), 
      weight_decay_(weight_decay), step_count_(0) {
    if (learning_rate <= 0.0f) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (beta1 < 0.0f || beta1 >= 1.0f) {
        throw std::invalid_argument("Beta1 must be in range [0, 1)");
    }
    if (beta2 < 0.0f || beta2 >= 1.0f) {
        throw std::invalid_argument("Beta2 must be in range [0, 1)");
    }
    if (eps <= 0.0f) {
        throw std::invalid_argument("Epsilon must be positive");
    }
    if (weight_decay < 0.0f) {
        throw std::invalid_argument("Weight decay must be non-negative");
    }
}

void AdamOptimizer::step() {
    step_count_++;
    
    // Apply parameter updates
    for (auto& [name, param_info] : parameters_) {
        if (param_info.gradient.size() == 0) {
            continue; // Skip parameters without gradients
        }
        
        update_parameter(name, param_info.gradient, *param_info.parameter);
    }
}

void AdamOptimizer::zero_grad() {
    // Clear all gradients
    for (auto& [name, param_info] : parameters_) {
        if (param_info.gradient.size() > 0) {
            param_info.gradient.zero();
        }
    }
}

void AdamOptimizer::add_parameter_group(const std::string& name, core::Tensor& parameter) {
    if (parameters_.find(name) != parameters_.end()) {
        throw std::runtime_error("Parameter " + name + " already exists");
    }
    
    ParameterInfo info;
    info.parameter = &parameter;  // Store pointer to the actual parameter
    info.gradient = core::Tensor(parameter.shape());
    info.gradient.zero();
    
    parameters_[name] = info;
    
    // Initialize moment estimates
    m_.emplace(name, core::Tensor(parameter.shape()));
    m_[name].zero();
    v_.emplace(name, core::Tensor(parameter.shape()));
    v_[name].zero();
}

void AdamOptimizer::set_gradient(const std::string& name, const core::Tensor& gradient) {
    auto it = parameters_.find(name);
    if (it == parameters_.end()) {
        throw std::runtime_error("Parameter " + name + " not found");
    }
    
    if (gradient.size() != it->second.parameter->size()) {
        throw std::runtime_error("Gradient size mismatch for parameter " + name);
    }
    
    // Copy gradient data
    const float* grad_data = gradient.data_float();
    float* stored_grad = it->second.gradient.data_float();
    size_t size = gradient.size();
    
    for (size_t i = 0; i < size; ++i) {
        stored_grad[i] = grad_data[i];
    }
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
    
    // Avoid division by zero
    bias_correction1 = std::max(bias_correction1, eps_);
    bias_correction2 = std::max(bias_correction2, eps_);
    
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
        
        // Update parameter with numerical stability and gradient clipping
        float update = learning_rate_ * m_hat / (std::sqrt(v_hat) + eps_);
        
        // Gradient clipping for numerical stability
        const float max_update = 10.0f;
        update = std::clamp(update, -max_update, max_update);
        
        param_data[i] -= update;
    }
}

float AdamOptimizer::get_learning_rate() const {
    return learning_rate_;
}

void AdamOptimizer::set_learning_rate(float lr) {
    if (lr <= 0.0f) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    learning_rate_ = lr;
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