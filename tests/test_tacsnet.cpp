#include <gtest/gtest.h>
#include "models/tacsnet.h"

using namespace tacs::models;
using namespace tacs::core;

class TACSNetTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(TACSNetTest, ModelConstruction) {
    TACSNet model;
    
    const auto& anchors = model.get_anchors();
    EXPECT_EQ(anchors.size(), 3);
    
    for (const auto& anchor_set : anchors) {
        EXPECT_EQ(anchor_set.size(), 6);  // Each anchor set contains 6 values (3 anchors * 2 coordinates each)
        for (const auto& anchor : anchor_set) {
            EXPECT_GT(anchor, 0);
        }
    }
}

TEST_F(TACSNetTest, ForwardPass) {
    TACSNet model;
    
    Tensor input({1, 3, 416, 416});
    input.randn(0.0f, 0.1f);
    
    auto outputs = model.forward(input, false);
    
    EXPECT_EQ(outputs.size(), 3);
    
    for (const auto& output : outputs) {
        EXPECT_GT(output.bbox_predictions.size(), 0);
        EXPECT_GT(output.objectness_scores.size(), 0);
        EXPECT_GT(output.class_predictions.size(), 0);
        
        const auto& bbox_shape = output.bbox_predictions.shape();
        const auto& obj_shape = output.objectness_scores.shape();
        const auto& cls_shape = output.class_predictions.shape();
        
        EXPECT_EQ(bbox_shape[0], 1);
        EXPECT_EQ(bbox_shape[1], 3);
        EXPECT_EQ(bbox_shape[4], 4);
        
        EXPECT_EQ(obj_shape[0], 1);
        EXPECT_EQ(obj_shape[1], 3);
        EXPECT_EQ(obj_shape[4], 1);
        
        EXPECT_EQ(cls_shape[0], 1);
        EXPECT_EQ(cls_shape[1], 3);
        EXPECT_EQ(cls_shape[4], 3);
    }
}

TEST_F(TACSNetTest, TrainingMode) {
    TACSNet model;
    
    model.set_training(true);
    
    Tensor input({1, 3, 416, 416});
    input.randn(0.0f, 0.1f);
    
    auto outputs_train = model.forward(input, true);
    EXPECT_EQ(outputs_train.size(), 3);
    
    model.set_training(false);
    auto outputs_eval = model.forward(input, false);
    EXPECT_EQ(outputs_eval.size(), 3);
}

TEST_F(TACSNetTest, GradientOperations) {
    TACSNet model;
    
    Tensor input({1, 3, 416, 416});
    input.randn(0.0f, 0.1f);
    
    auto outputs = model.forward(input, true);
    
    std::vector<Tensor> grad_outputs;
    for (const auto& output : outputs) {
        Tensor grad(output.bbox_predictions.shape());
        grad.randn(0.0f, 0.01f);
        grad_outputs.push_back(grad);
    }
    
    model.zero_grad();
    model.backward(grad_outputs, input);
    
    model.apply_gradients(0.001f);
}