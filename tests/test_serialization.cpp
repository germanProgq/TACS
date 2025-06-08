#include <gtest/gtest.h>
#include "models/tacsnet.h"
#include "utils/serialization.h"
#include <filesystem>

using namespace tacs::models;
using namespace tacs::utils;

class SerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_file_path = "./test_model_serialization.tacs";
    }
    
    void TearDown() override {
        if (std::filesystem::exists(test_file_path)) {
            std::filesystem::remove(test_file_path);
        }
    }
    
    std::string test_file_path;
};

TEST_F(SerializationTest, SaveAndLoadModel) {
    TACSNet original_model;
    
    bool save_success = original_model.save_model(test_file_path);
    EXPECT_TRUE(save_success);
    EXPECT_TRUE(std::filesystem::exists(test_file_path));
    
    TACSNet loaded_model;
    bool load_success = loaded_model.load_model(test_file_path);
    EXPECT_TRUE(load_success);
    
    const auto& original_anchors = original_model.get_anchors();
    const auto& loaded_anchors = loaded_model.get_anchors();
    
    EXPECT_EQ(original_anchors.size(), loaded_anchors.size());
}

TEST_F(SerializationTest, InvalidFilePath) {
    TACSNet model;
    
    bool save_success = model.save_model("/invalid/path/model.tacs");
    EXPECT_FALSE(save_success);
    
    bool load_success = model.load_model("/nonexistent/model.tacs");
    EXPECT_FALSE(load_success);
}

TEST_F(SerializationTest, TensorSerialization) {
    std::ofstream test_file("test_tensor.bin", std::ios::binary);
    ASSERT_TRUE(test_file.is_open());
    
    tacs::core::Tensor original_tensor({2, 3, 4});
    original_tensor.randn(0.0f, 1.0f);
    
    bool save_success = ModelSerializer::save_tensor(original_tensor, test_file);
    test_file.close();
    EXPECT_TRUE(save_success);
    
    std::ifstream read_file("test_tensor.bin", std::ios::binary);
    ASSERT_TRUE(read_file.is_open());
    
    tacs::core::Tensor loaded_tensor({1});
    bool load_success = ModelSerializer::load_tensor(loaded_tensor, read_file);
    read_file.close();
    EXPECT_TRUE(load_success);
    
    EXPECT_EQ(original_tensor.shape(), loaded_tensor.shape());
    EXPECT_EQ(original_tensor.dtype(), loaded_tensor.dtype());
    EXPECT_EQ(original_tensor.size(), loaded_tensor.size());
    
    std::filesystem::remove("test_tensor.bin");
}