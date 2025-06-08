#include <gtest/gtest.h>
#include "core/tensor.h"
#include "core/memory_manager.h"

using namespace tacs::core;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        MemoryManager::instance().reset_stats();
    }
};

TEST_F(TensorTest, BasicConstruction) {
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.shape().size(), 3);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.shape()[2], 4);
    EXPECT_EQ(t.size(), 24);
    EXPECT_EQ(t.dtype(), DataType::FLOAT32);
}

TEST_F(TensorTest, ZeroAndFill) {
    Tensor t({2, 2});
    t.fill(5.0f);
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(t({i, j}), 5.0f);
        }
    }
    
    t.zero();
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(t({i, j}), 0.0f);
        }
    }
}

TEST_F(TensorTest, Reshape) {
    Tensor t({2, 3});
    t.fill(1.0f);
    
    auto reshaped = t.reshape({3, 2});
    EXPECT_EQ(reshaped.shape()[0], 3);
    EXPECT_EQ(reshaped.shape()[1], 2);
    EXPECT_EQ(reshaped.size(), 6);
}

TEST_F(TensorTest, Transpose) {
    Tensor t({2, 3});
    t.fill(1.0f);
    t({0, 0}) = 1.0f;
    t({0, 1}) = 2.0f;
    t({1, 0}) = 3.0f;
    
    auto transposed = t.transpose(0, 1);
    EXPECT_EQ(transposed.shape()[0], 3);
    EXPECT_EQ(transposed.shape()[1], 2);
    EXPECT_FLOAT_EQ(transposed({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(transposed({1, 0}), 2.0f);
    EXPECT_FLOAT_EQ(transposed({0, 1}), 3.0f);
}

TEST_F(TensorTest, CopyConstructor) {
    Tensor t1({2, 2});
    t1.fill(3.14f);
    
    Tensor t2(t1);
    EXPECT_EQ(t2.shape(), t1.shape());
    EXPECT_EQ(t2.size(), t1.size());
    EXPECT_FLOAT_EQ(t2({0, 0}), 3.14f);
}

TEST_F(TensorTest, MoveConstructor) {
    Tensor t1({2, 2});
    t1.fill(2.71f);
    
    Tensor t2(std::move(t1));
    EXPECT_EQ(t2.shape()[0], 2);
    EXPECT_EQ(t2.shape()[1], 2);
    EXPECT_FLOAT_EQ(t2({0, 0}), 2.71f);
}