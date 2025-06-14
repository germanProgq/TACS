/**
 * @file optimizer.h
 * @brief Production-ready optimizers for neural network training
 * 
 * Implements SGD with momentum and Adam optimizers with advanced features
 * including weight decay, bias correction, and numerical stability.
 * Optimized for high-performance training of TACS neural networks.
 */
#pragma once

#include "core/tensor.h"
#include <unordered_map>
#include <string>

namespace tacs {
namespace training {

// Base optimizer interface
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    virtual float get_learning_rate() const = 0;
    virtual void set_learning_rate(float lr) = 0;
    virtual void add_parameter_group(const std::string& name, core::Tensor& parameter) = 0;
    virtual void set_gradient(const std::string& name, const core::Tensor& gradient) = 0;
};

// Parameter information structure
struct ParameterInfo {
    core::Tensor parameter;
    core::Tensor gradient;
};

class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(float learning_rate = 0.01f, float momentum = 0.9f, float weight_decay = 0.0f);
    
    void step() override;
    void zero_grad() override;
    void set_learning_rate(float lr) override;
    float get_learning_rate() const override;
    
    // Parameter management
    void add_parameter_group(const std::string& name, core::Tensor& parameter) override;
    void set_gradient(const std::string& name, const core::Tensor& gradient) override;
    
    // Advanced parameter update with momentum and weight decay
    void update_parameter(const std::string& param_name, 
                          const core::Tensor& gradient, 
                          core::Tensor& parameter);
    
    void reset();

private:
    float learning_rate_;
    float momentum_;
    float weight_decay_;
    std::unordered_map<std::string, ParameterInfo> parameters_;
    std::unordered_map<std::string, core::Tensor> velocity_;
};

class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, 
                  float eps = 1e-8f, float weight_decay = 0.0f);
    
    void step() override;
    void zero_grad() override;
    void set_learning_rate(float lr) override;
    float get_learning_rate() const override;
    
    // Parameter management
    void add_parameter_group(const std::string& name, core::Tensor& parameter) override;
    void set_gradient(const std::string& name, const core::Tensor& gradient) override;
    
    // Advanced parameter update with adaptive learning rates
    void update_parameter(const std::string& param_name,
                          const core::Tensor& gradient,
                          core::Tensor& parameter);
    
    void reset();

private:
    float learning_rate_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    int step_count_;
    std::unordered_map<std::string, ParameterInfo> parameters_;
    std::unordered_map<std::string, core::Tensor> m_;  // First moment estimates
    std::unordered_map<std::string, core::Tensor> v_;  // Second moment estimates
};

}
}