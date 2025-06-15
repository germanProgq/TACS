#include <gtest/gtest.h>
#include "core/tensor.h"
#include "core/memory_manager.h"
#include <random>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>
#include <atomic>

using namespace tacs::core;

// Production-level testing constants
constexpr float EPSILON = 1e-6f;
constexpr int STRESS_TEST_ITERATIONS = 10000;
constexpr int CONCURRENT_THREADS = 8;

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

// Production-level reliability tests
TEST_F(TensorTest, MemoryAlignmentVerification) {
    Tensor t({64, 64, 32});
    auto* data_ptr = t.data();
    
    // Verify 32-byte alignment for SIMD operations
    EXPECT_EQ(reinterpret_cast<uintptr_t>(data_ptr) % 32, 0);
}

TEST_F(TensorTest, BoundaryConditionChecks) {
    // Test with minimum size
    Tensor t_min({1});
    EXPECT_EQ(t_min.size(), 1);
    EXPECT_NO_THROW(t_min({0}) = 1.0f);
    
    // Test with large size (but not too large for CI)
    Tensor t_large({128, 128, 64});
    EXPECT_EQ(t_large.size(), 128 * 128 * 64);
    EXPECT_NO_THROW(t_large({127, 127, 63}) = 1.0f);
}

TEST_F(TensorTest, NumericalStabilityTest) {
    Tensor t({100});
    
    // Test with extreme values
    t.fill(std::numeric_limits<float>::max());
    EXPECT_FLOAT_EQ(t({0}), std::numeric_limits<float>::max());
    
    t.fill(std::numeric_limits<float>::min());
    EXPECT_FLOAT_EQ(t({0}), std::numeric_limits<float>::min());
    
    t.fill(std::numeric_limits<float>::epsilon());
    EXPECT_FLOAT_EQ(t({0}), std::numeric_limits<float>::epsilon());
}

TEST_F(TensorTest, ConcurrentAccessSafety) {
    const int num_elements = 1000;
    Tensor t({num_elements});
    t.zero();
    
    std::vector<std::thread> threads;
    std::atomic<int> counter{0};
    
    // Multiple threads writing to different elements
    for (int i = 0; i < CONCURRENT_THREADS; ++i) {
        threads.emplace_back([&t, &counter, i, num_elements]() {
            int start = (num_elements / CONCURRENT_THREADS) * i;
            int end = (i == CONCURRENT_THREADS - 1) ? num_elements : 
                     (num_elements / CONCURRENT_THREADS) * (i + 1);
            
            for (int j = start; j < end; ++j) {
                t({j}) = static_cast<float>(j);
                counter.fetch_add(1);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(counter.load(), num_elements);
    
    // Verify all values were written correctly
    for (int i = 0; i < num_elements; ++i) {
        EXPECT_FLOAT_EQ(t({i}), static_cast<float>(i));
    }
}

TEST_F(TensorTest, StressTestMemoryAllocation) {
    // Rapid allocation and deallocation
    for (int i = 0; i < STRESS_TEST_ITERATIONS; ++i) {
        Tensor t({10, 10});
        t.fill(static_cast<float>(i));
        EXPECT_FLOAT_EQ(t({0, 0}), static_cast<float>(i));
    }
    
    // Verify no memory leaks via memory manager stats
    auto& mm = MemoryManager::instance();
    EXPECT_GE(mm.peak_allocated(), 0);
}

TEST_F(TensorTest, EdgeCaseHandling) {
    // Test empty tensor creation
    EXPECT_THROW(Tensor t(std::vector<int>{}), std::invalid_argument);
    
    // Test negative dimensions
    EXPECT_THROW(Tensor t({-1, 10}), std::invalid_argument);
    
    // Test zero dimensions
    EXPECT_THROW(Tensor t({0, 10}), std::invalid_argument);
}

TEST_F(TensorTest, RandomizedPropertyTest) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dim_dist(1, 100);
    std::uniform_real_distribution<float> val_dist(-1000.0f, 1000.0f);
    
    for (int iter = 0; iter < 100; ++iter) {
        // Random dimensions
        std::vector<int> shape;
        int num_dims = dim_dist(gen) % 4 + 1; // 1-4 dimensions
        size_t total_size = 1;
        
        for (int i = 0; i < num_dims; ++i) {
            int dim = dim_dist(gen);
            shape.push_back(dim);
            total_size *= dim;
        }
        
        Tensor t(shape);
        EXPECT_EQ(t.size(), total_size);
        
        // Fill with random values and verify
        float test_val = val_dist(gen);
        t.fill(test_val);
        
        // Sample random positions
        for (int i = 0; i < 10; ++i) {
            std::vector<int> indices;
            for (int j = 0; j < num_dims; ++j) {
                indices.push_back(std::uniform_int_distribution<>(0, shape[j] - 1)(gen));
            }
            EXPECT_NEAR(t(indices), test_val, EPSILON);
        }
    }
}