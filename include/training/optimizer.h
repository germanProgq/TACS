#pragma once

#include "core/tensor.h"
#include <unordered_map>
#include <string>

namespace tacs {
namespace training {

class SGDOptimizer {
public:
    SGDOptimizer(float learning_rate = 0.01f, float momentum = 0.9f, float weight_decay = 0.0f);
    
    void step();
    void zero_grad();
    void set_learning_rate(float lr) { learning_rate_ = lr; }
    float get_learning_rate() const { return learning_rate_; }

private:
    float learning_rate_;
    float momentum_;
    float weight_decay_;
    std::unordered_map<std::string, core::Tensor> velocity_;
};

class AdamOptimizer {
public:
    AdamOptimizer(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, 
                  float eps = 1e-8f, float weight_decay = 0.0f);
    
    void step();
    void zero_grad();
    void set_learning_rate(float lr) { learning_rate_ = lr; }
    float get_learning_rate() const { return learning_rate_; }

private:
    float learning_rate_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    int step_count_;
    std::unordered_map<std::string, core::Tensor> m_;
    std::unordered_map<std::string, core::Tensor> v_;
};

}
}