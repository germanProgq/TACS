#include <gtest/gtest.h>
#include "layers/conv2d.h"
#include "layers/batch_norm.h"
#include "layers/leaky_relu.h"

using namespace tacs::layers;
using namespace tacs::core;

class LayersTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(LayersTest, Conv2DForward) {
    Conv2D conv(3, 16, 3, 1, 1);
    
    Tensor input({1, 3, 32, 32});
    input.randn(0.0f, 0.1f);
    
    auto output = conv.forward(input);
    
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 32);
    EXPECT_EQ(output.shape()[3], 32);
}

TEST_F(LayersTest, BatchNorm2DForward) {
    BatchNorm2D bn(16);
    
    Tensor input({2, 16, 8, 8});
    input.randn(0.0f, 1.0f);
    
    auto output = bn.forward(input, true);
    
    EXPECT_EQ(output.shape(), input.shape());
}

TEST_F(LayersTest, LeakyReLUForward) {
    LeakyReLU relu(0.1f);
    
    Tensor input({2, 2});
    input({0, 0}) = 1.0f;
    input({0, 1}) = -1.0f;
    input({1, 0}) = 0.5f;
    input({1, 1}) = -0.5f;
    
    auto output = relu.forward(input);
    
    EXPECT_FLOAT_EQ(output({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(output({0, 1}), -0.1f);
    EXPECT_FLOAT_EQ(output({1, 0}), 0.5f);
    EXPECT_FLOAT_EQ(output({1, 1}), -0.05f);
}

TEST_F(LayersTest, Conv2DGradients) {
    Conv2D conv(1, 1, 3, 1, 1);
    
    Tensor input({1, 1, 5, 5});
    input.fill(1.0f);
    
    auto output = conv.forward(input);
    
    Tensor grad_output(output.shape());
    grad_output.fill(1.0f);
    
    conv.zero_grad();
    conv.backward(grad_output, input);
    
    const auto& weight_grad = conv.weight_grad();
    EXPECT_GT(weight_grad.size(), 0);
}