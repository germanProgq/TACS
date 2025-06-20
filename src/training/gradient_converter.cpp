/**
 * @file gradient_converter.cpp
 * @brief Implementation of gradient format conversion utilities
 */

#include "training/gradient_converter.h"
#include <stdexcept>
#include <cstring>

namespace tacs {
namespace training {

std::tuple<core::Tensor, core::Tensor, core::Tensor> 
GradientConverter::split_combined_gradient(const core::Tensor& combined_grad, 
                                          int num_anchors, 
                                          int num_classes) {
    const auto& shape = combined_grad.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("Combined gradient must be 4D tensor");
    }
    
    int batch_size = shape[0];
    int total_channels = shape[1];
    int grid_h = shape[2];
    int grid_w = shape[3];
    
    int expected_channels = num_anchors * (5 + num_classes);
    if (total_channels != expected_channels) {
        throw std::runtime_error("Channel mismatch in combined gradient");
    }
    
    // Create output tensors
    core::Tensor bbox_grad({batch_size, num_anchors, grid_h, grid_w, 4});
    core::Tensor obj_grad({batch_size, num_anchors, grid_h, grid_w, 1});
    core::Tensor cls_grad({batch_size, num_anchors, grid_h, grid_w, num_classes});
    
    // Get data pointers
    const float* combined_data = combined_grad.data_float();
    float* bbox_data = bbox_grad.data_float();
    float* obj_data = obj_grad.data_float();
    float* cls_data = cls_grad.data_float();
    
    // Split the combined gradient
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch_size; ++b) {
        for (int a = 0; a < num_anchors; ++a) {
            for (int h = 0; h < grid_h; ++h) {
                for (int w = 0; w < grid_w; ++w) {
                    // Calculate indices for NCHW format
                    int anchor_offset = a * (5 + num_classes);
                    
                    // Calculate destination indices for 5D tensors [batch, anchors, H, W, channels]
                    int spatial_idx = ((b * num_anchors + a) * grid_h + h) * grid_w + w;
                    int bbox_base = spatial_idx * 4;
                    int obj_base = spatial_idx;
                    int cls_base = spatial_idx * num_classes;
                    
                    // Copy bbox gradients (4 values: x, y, w, h)
                    for (int i = 0; i < 4; ++i) {
                        int src_idx = ((b * total_channels + anchor_offset + i) * grid_h + h) * grid_w + w;
                        bbox_data[bbox_base + i] = combined_data[src_idx];
                    }
                    
                    // Copy objectness gradient (1 value)
                    int obj_src_idx = ((b * total_channels + anchor_offset + 4) * grid_h + h) * grid_w + w;
                    obj_data[obj_base] = combined_data[obj_src_idx];
                    
                    // Copy classification gradients (num_classes values)
                    for (int c = 0; c < num_classes; ++c) {
                        int cls_src_idx = ((b * total_channels + anchor_offset + 5 + c) * grid_h + h) * grid_w + w;
                        cls_data[cls_base + c] = combined_data[cls_src_idx];
                    }
                }
            }
        }
    }
    
    return std::make_tuple(bbox_grad, obj_grad, cls_grad);
}

core::Tensor GradientConverter::combine_gradients(const core::Tensor& bbox_grad,
                                                  const core::Tensor& obj_grad,
                                                  const core::Tensor& cls_grad) {
    // Validate input shapes
    const auto& bbox_shape = bbox_grad.shape();
    const auto& obj_shape = obj_grad.shape();
    const auto& cls_shape = cls_grad.shape();
    
    if (bbox_shape.size() != 5 || obj_shape.size() != 5 || cls_shape.size() != 5) {
        throw std::runtime_error("Gradient tensors must be 5D");
    }
    
    int batch_size = bbox_shape[0];
    int num_anchors = bbox_shape[1];
    int grid_h = bbox_shape[2];
    int grid_w = bbox_shape[3];
    int num_classes = cls_shape[4];
    
    // Validate dimensions match
    if (obj_shape[0] != batch_size || cls_shape[0] != batch_size ||
        obj_shape[1] != num_anchors || cls_shape[1] != num_anchors ||
        obj_shape[2] != grid_h || cls_shape[2] != grid_h ||
        obj_shape[3] != grid_w || cls_shape[3] != grid_w) {
        throw std::runtime_error("Gradient tensor dimensions mismatch");
    }
    
    // Create combined tensor
    int total_channels = num_anchors * (5 + num_classes);
    core::Tensor combined({batch_size, total_channels, grid_h, grid_w});
    combined.zero();
    
    // Get data pointers
    const float* bbox_data = bbox_grad.data_float();
    const float* obj_data = obj_grad.data_float();
    const float* cls_data = cls_grad.data_float();
    float* combined_data = combined.data_float();
    
    // Combine gradients
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch_size; ++b) {
        for (int a = 0; a < num_anchors; ++a) {
            for (int h = 0; h < grid_h; ++h) {
                for (int w = 0; w < grid_w; ++w) {
                    int anchor_offset = a * (5 + num_classes);
                    
                    // Calculate source indices for 5D tensors [batch, anchors, H, W, channels]
                    int spatial_idx = ((b * num_anchors + a) * grid_h + h) * grid_w + w;
                    int bbox_base = spatial_idx * 4;
                    int obj_base = spatial_idx;
                    int cls_base = spatial_idx * num_classes;
                    
                    // Copy bbox gradients
                    for (int i = 0; i < 4; ++i) {
                        int dst_idx = ((b * total_channels + anchor_offset + i) * grid_h + h) * grid_w + w;
                        combined_data[dst_idx] = bbox_data[bbox_base + i];
                    }
                    
                    // Copy objectness gradient
                    int obj_dst_idx = ((b * total_channels + anchor_offset + 4) * grid_h + h) * grid_w + w;
                    combined_data[obj_dst_idx] = obj_data[obj_base];
                    
                    // Copy classification gradients
                    for (int c = 0; c < num_classes; ++c) {
                        int cls_dst_idx = ((b * total_channels + anchor_offset + 5 + c) * grid_h + h) * grid_w + w;
                        combined_data[cls_dst_idx] = cls_data[cls_base + c];
                    }
                }
            }
        }
    }
    
    return combined;
}

std::vector<models::DetectionOutput> 
GradientConverter::convert_loss_gradients_to_detection_format(
    const std::vector<core::Tensor>& loss_gradients,
    const std::vector<models::DetectionOutput>& predictions) {
    
    if (loss_gradients.size() != predictions.size()) {
        throw std::runtime_error("Gradient and prediction count mismatch");
    }
    
    std::vector<models::DetectionOutput> gradient_outputs;
    
    for (size_t i = 0; i < loss_gradients.size(); ++i) {
        const auto& combined_grad = loss_gradients[i];
        const auto& pred = predictions[i];
        
        // Get dimensions from prediction
        const auto& bbox_shape = pred.bbox_predictions.shape();
        int num_anchors = bbox_shape[1];
        int num_classes = pred.class_predictions.shape()[4];
        
        // Split combined gradient
        auto [bbox_grad, obj_grad, cls_grad] = split_combined_gradient(
            combined_grad, num_anchors, num_classes);
        
        // Create DetectionOutput with gradients
        models::DetectionOutput grad_output;
        grad_output.bbox_predictions = bbox_grad;
        grad_output.objectness_scores = obj_grad;
        grad_output.class_predictions = cls_grad;
        
        gradient_outputs.push_back(grad_output);
    }
    
    return gradient_outputs;
}

} // namespace training
} // namespace tacs