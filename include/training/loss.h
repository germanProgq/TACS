/**
 * @file loss.h
 * @brief YOLO loss functions for object detection training
 * 
 * Implements multi-component loss including objectness, bounding box regression
 * using GIoU, and classification loss. Weighted combination optimized for
 * traffic object detection with balanced precision and recall requirements.
 */
#pragma once

#include "core/tensor.h"
#include "models/tacsnet.h"

namespace tacs {
namespace training {

struct LossWeights {
    float objectness = 1.0f;
    float bbox = 5.0f;
    float classification = 1.0f;
};

class YOLOLoss {
public:
    explicit YOLOLoss(const LossWeights& weights = LossWeights{});
    
    float compute_loss(const std::vector<models::DetectionOutput>& predictions,
                       const core::Tensor& targets,
                       const std::vector<std::vector<float>>& anchors);
    
    std::vector<core::Tensor> backward(const std::vector<models::DetectionOutput>& predictions,
                                       const core::Tensor& targets,
                                       const std::vector<std::vector<float>>& anchors);

private:
    LossWeights weights_;
    
    float compute_objectness_loss(const core::Tensor& pred_objectness,
                                  const core::Tensor& target_objectness,
                                  const core::Tensor& ignore_mask);
    
    float compute_bbox_loss(const core::Tensor& pred_bbox,
                            const core::Tensor& target_bbox,
                            const core::Tensor& object_mask);
    
    float compute_classification_loss(const core::Tensor& pred_classes,
                                      const core::Tensor& target_classes,
                                      const core::Tensor& object_mask);
    
    void assign_targets_to_grid(const core::Tensor& targets,
                                core::Tensor& target_bbox,
                                core::Tensor& target_obj,
                                core::Tensor& target_cls,
                                core::Tensor& object_mask,
                                core::Tensor& ignore_mask,
                                const std::vector<float>& anchors,
                                int grid_h, int grid_w, int scale_idx);
    
    float compute_giou(const std::vector<float>& box1, const std::vector<float>& box2);
    float sigmoid(float x);
    float binary_cross_entropy(float pred, float target);
};

}
}