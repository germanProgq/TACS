#pragma once

#include "core/tensor.h"

namespace tacs {
namespace layers {

class LeakyReLU {
public:
    explicit LeakyReLU(float negative_slope = 0.01f);
    ~LeakyReLU() = default;

    core::Tensor forward(const core::Tensor& input);
    core::Tensor backward(const core::Tensor& grad_output, const core::Tensor& input);

private:
    float negative_slope_;
};

}
}