#include "training/optimizer.h"
#include <cmath>

namespace tacs {
namespace training {

SGDOptimizer::SGDOptimizer(float learning_rate, float momentum, float weight_decay)
    : learning_rate_(learning_rate), momentum_(momentum), weight_decay_(weight_decay) {}

void SGDOptimizer::step() {
}

void SGDOptimizer::zero_grad() {
}

AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2, float eps, float weight_decay)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps), 
      weight_decay_(weight_decay), step_count_(0) {}

void AdamOptimizer::step() {
    step_count_++;
}

void AdamOptimizer::zero_grad() {
}

}
}