/**
 * @file gradient_converter.h
 * @brief Utility for converting between gradient tensor formats
 * 
 * Handles conversion between combined gradient tensors from loss functions
 * and separate bbox/objectness/classification gradients for model backward pass.
 */
#pragma once

#include "core/tensor.h"
#include "models/tacsnet.h"
#include <vector>

namespace tacs {
namespace training {

class GradientConverter {
public:
    /**
     * @brief Convert combined gradient tensor to separate detection output gradients
     * 
     * @param combined_grad Combined gradient tensor [batch, anchors*(5+classes), H, W]
     * @param num_anchors Number of anchors per scale
     * @param num_classes Number of object classes
     * @return Tuple of (bbox_grad, obj_grad, cls_grad)
     */
    static std::tuple<core::Tensor, core::Tensor, core::Tensor> 
    split_combined_gradient(const core::Tensor& combined_grad, 
                           int num_anchors, 
                           int num_classes);
    
    /**
     * @brief Convert detection output gradients to combined gradient tensor
     * 
     * @param bbox_grad Bounding box gradients [batch, anchors, H, W, 4]
     * @param obj_grad Objectness gradients [batch, anchors, H, W, 1]
     * @param cls_grad Classification gradients [batch, anchors, H, W, classes]
     * @return Combined gradient tensor
     */
    static core::Tensor combine_gradients(const core::Tensor& bbox_grad,
                                         const core::Tensor& obj_grad,
                                         const core::Tensor& cls_grad);
    
    /**
     * @brief Convert loss backward output to model backward input format
     * 
     * @param loss_gradients Vector of gradient tensors from loss backward
     * @param predictions Original predictions for shape reference
     * @return Vector of DetectionOutput with gradients
     */
    static std::vector<models::DetectionOutput> 
    convert_loss_gradients_to_detection_format(
        const std::vector<core::Tensor>& loss_gradients,
        const std::vector<models::DetectionOutput>& predictions);
};

} // namespace training
} // namespace tacs