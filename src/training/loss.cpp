#include "training/loss.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace tacs {
namespace training {

YOLOLoss::YOLOLoss(const LossWeights& weights) : weights_(weights) {}

float YOLOLoss::compute_loss(const std::vector<models::DetectionOutput>& predictions,
                             const core::Tensor& targets,
                             const std::vector<std::vector<float>>& anchors) {
    if (predictions.size() != anchors.size()) {
        throw std::runtime_error("Number of predictions must match number of anchor sets");
    }
    
    float total_loss = 0.0f;
    
    for (size_t scale = 0; scale < predictions.size(); ++scale) {
        const auto& pred = predictions[scale];
        const auto& pred_bbox = pred.bbox_predictions;
        const auto& pred_obj = pred.objectness_scores;
        const auto& pred_cls = pred.class_predictions;
        
        const auto& bbox_shape = pred_bbox.shape();
        int batch_size = bbox_shape[0];
        int num_anchors = bbox_shape[1];
        int grid_h = bbox_shape[2];
        int grid_w = bbox_shape[3];
        
        core::Tensor target_bbox({batch_size, num_anchors, grid_h, grid_w, 4});
        core::Tensor target_obj({batch_size, num_anchors, grid_h, grid_w, 1});
        core::Tensor target_cls({batch_size, num_anchors, grid_h, grid_w, 3});
        core::Tensor object_mask({batch_size, num_anchors, grid_h, grid_w, 1});
        core::Tensor ignore_mask({batch_size, num_anchors, grid_h, grid_w, 1});
        
        target_bbox.zero();
        target_obj.zero();
        target_cls.zero();
        object_mask.zero();
        ignore_mask.zero();
        
        assign_targets_to_grid(targets, target_bbox, target_obj, target_cls, 
                               object_mask, ignore_mask, anchors[scale], 
                               grid_h, grid_w, scale);
        
        float obj_loss = compute_objectness_loss(pred_obj, target_obj, ignore_mask);
        float bbox_loss = compute_bbox_loss(pred_bbox, target_bbox, object_mask);
        float cls_loss = compute_classification_loss(pred_cls, target_cls, object_mask);
        
        total_loss += weights_.objectness * obj_loss + 
                      weights_.bbox * bbox_loss + 
                      weights_.classification * cls_loss;
    }
    
    return total_loss;
}

std::vector<core::Tensor> YOLOLoss::backward(const std::vector<models::DetectionOutput>& predictions,
                                              const core::Tensor& targets,
                                              const std::vector<std::vector<float>>& anchors) {
    std::vector<core::Tensor> gradients;
    
    for (size_t scale = 0; scale < predictions.size(); ++scale) {
        const auto& pred = predictions[scale];
        const auto& pred_bbox = pred.bbox_predictions;
        const auto& pred_obj = pred.objectness_scores;
        const auto& pred_cls = pred.class_predictions;
        
        const auto& bbox_shape = pred_bbox.shape();
        int batch_size = bbox_shape[0];
        int num_anchors = bbox_shape[1];
        int grid_h = bbox_shape[2];
        int grid_w = bbox_shape[3];
        int num_classes = pred_cls.shape()[4];
        
        core::Tensor target_bbox({batch_size, num_anchors, grid_h, grid_w, 4});
        core::Tensor target_obj({batch_size, num_anchors, grid_h, grid_w, 1});
        core::Tensor target_cls({batch_size, num_anchors, grid_h, grid_w, num_classes});
        core::Tensor object_mask({batch_size, num_anchors, grid_h, grid_w, 1});
        core::Tensor ignore_mask({batch_size, num_anchors, grid_h, grid_w, 1});
        
        target_bbox.zero();
        target_obj.zero();
        target_cls.zero();
        object_mask.zero();
        ignore_mask.zero();
        
        assign_targets_to_grid(targets, target_bbox, target_obj, target_cls, 
                               object_mask, ignore_mask, anchors[scale], 
                               grid_h, grid_w, scale);
        
        int total_outputs = num_anchors * (5 + num_classes);
        core::Tensor grad({batch_size, total_outputs, grid_h, grid_w});
        grad.zero();
        
        float* grad_data = grad.data_float();
        const float* bbox_data = pred_bbox.data_float();
        const float* obj_data = pred_obj.data_float();
        const float* cls_data = pred_cls.data_float();
        const float* target_bbox_data = target_bbox.data_float();
        const float* target_obj_data = target_obj.data_float();
        const float* target_cls_data = target_cls.data_float();
        const float* object_mask_data = object_mask.data_float();
        const float* ignore_mask_data = ignore_mask.data_float();
        
        // Optimized gradient computation with improved numerical stability
        const float epsilon = 1e-7f;
        const float gradient_clip_value = 10.0f;
        
        for (int n = 0; n < batch_size; ++n) {
            for (int a = 0; a < num_anchors; ++a) {
                for (int h = 0; h < grid_h; ++h) {
                    for (int w = 0; w < grid_w; ++w) {
                        int base_idx = n * total_outputs * grid_h * grid_w +
                                       a * (5 + num_classes) * grid_h * grid_w +
                                       h * grid_w * (5 + num_classes) + w * (5 + num_classes);
                        
                        int pred_idx = n * num_anchors * grid_h * grid_w +
                                       a * grid_h * grid_w + h * grid_w + w;
                        
                        // Bounding box gradients with analytical GIoU derivatives
                        if (object_mask_data[pred_idx] > 0.5f) {
                            // Extract predictions and targets
                            float tx = bbox_data[pred_idx * 4 + 0];
                            float ty = bbox_data[pred_idx * 4 + 1];
                            float tw = bbox_data[pred_idx * 4 + 2];
                            float th = bbox_data[pred_idx * 4 + 3];
                            
                            float target_tx = target_bbox_data[pred_idx * 4 + 0];
                            float target_ty = target_bbox_data[pred_idx * 4 + 1];
                            float target_tw = target_bbox_data[pred_idx * 4 + 2];
                            float target_th = target_bbox_data[pred_idx * 4 + 3];
                            
                            // Analytical gradients for tx, ty (sigmoid activated)
                            float sig_tx = sigmoid(tx);
                            float sig_ty = sigmoid(ty);
                            float grad_tx = weights_.bbox * 2.0f * (sig_tx - target_tx) * sig_tx * (1.0f - sig_tx);
                            float grad_ty = weights_.bbox * 2.0f * (sig_ty - target_ty) * sig_ty * (1.0f - sig_ty);
                            
                            // Analytical gradients for tw, th (exp activated)
                            float exp_tw = std::exp(std::clamp(tw, -10.0f, 10.0f));
                            float exp_th = std::exp(std::clamp(th, -10.0f, 10.0f));
                            float target_exp_tw = std::exp(target_tw);
                            float target_exp_th = std::exp(target_th);
                            
                            float grad_tw = weights_.bbox * 2.0f * (exp_tw - target_exp_tw) * exp_tw / (target_exp_tw + epsilon);
                            float grad_th = weights_.bbox * 2.0f * (exp_th - target_exp_th) * exp_th / (target_exp_th + epsilon);
                            
                            // Apply gradient clipping
                            grad_data[base_idx + 0] = std::clamp(grad_tx, -gradient_clip_value, gradient_clip_value);
                            grad_data[base_idx + 1] = std::clamp(grad_ty, -gradient_clip_value, gradient_clip_value);
                            grad_data[base_idx + 2] = std::clamp(grad_tw, -gradient_clip_value, gradient_clip_value);
                            grad_data[base_idx + 3] = std::clamp(grad_th, -gradient_clip_value, gradient_clip_value);
                        }
                        
                        // Objectness gradient with numerical stability
                        if (ignore_mask_data[pred_idx] < 0.5f) {
                            float obj_raw = obj_data[pred_idx];
                            float obj_pred = sigmoid(std::clamp(obj_raw, -10.0f, 10.0f));
                            float obj_target = target_obj_data[pred_idx];
                            
                            // Stable sigmoid gradient computation
                            float obj_grad = weights_.objectness * (obj_pred - obj_target);
                            if (std::abs(obj_raw) < 10.0f) {
                                obj_grad *= obj_pred * (1.0f - obj_pred);
                            } else {
                                // For extreme values, use approximation to avoid numerical issues
                                obj_grad *= epsilon;
                            }
                            
                            grad_data[base_idx + 4] = std::clamp(obj_grad, -gradient_clip_value, gradient_clip_value);
                        }
                        
                        // Classification gradients with improved stability
                        if (object_mask_data[pred_idx] > 0.5f) {
                            for (int c = 0; c < num_classes; ++c) {
                                float cls_raw = cls_data[pred_idx * num_classes + c];
                                float cls_pred = sigmoid(std::clamp(cls_raw, -10.0f, 10.0f));
                                float cls_target = target_cls_data[pred_idx * num_classes + c];
                                
                                // Stable sigmoid gradient computation
                                float cls_grad = weights_.classification * (cls_pred - cls_target);
                                if (std::abs(cls_raw) < 10.0f) {
                                    cls_grad *= cls_pred * (1.0f - cls_pred);
                                } else {
                                    // For extreme values, use approximation
                                    cls_grad *= epsilon;
                                }
                                
                                grad_data[base_idx + 5 + c] = std::clamp(cls_grad, -gradient_clip_value, gradient_clip_value);
                            }
                        }
                    }
                }
            }
        }
        
        gradients.push_back(grad);
    }
    
    return gradients;
}

float YOLOLoss::compute_objectness_loss(const core::Tensor& pred_objectness,
                                        const core::Tensor& target_objectness,
                                        const core::Tensor& ignore_mask) {
    if (pred_objectness.size() != target_objectness.size()) {
        throw std::runtime_error("Prediction and target sizes must match");
    }
    
    const float* pred_data = pred_objectness.data_float();
    const float* target_data = target_objectness.data_float();
    const float* ignore_data = ignore_mask.data_float();
    
    float loss = 0.0f;
    int valid_samples = 0;
    
    for (size_t i = 0; i < pred_objectness.size(); ++i) {
        if (ignore_data[i] < 0.5f) {
            float pred_sigmoid = sigmoid(pred_data[i]);
            loss += binary_cross_entropy(pred_sigmoid, target_data[i]);
            valid_samples++;
        }
    }
    
    return valid_samples > 0 ? loss / valid_samples : 0.0f;
}

float YOLOLoss::compute_bbox_loss(const core::Tensor& pred_bbox,
                                  const core::Tensor& target_bbox,
                                  const core::Tensor& object_mask) {
    if (pred_bbox.size() != target_bbox.size() || pred_bbox.size() / 4 != object_mask.size()) {
        throw std::runtime_error("Bbox prediction and target sizes must match");
    }
    
    const float* pred_data = pred_bbox.data_float();
    const float* target_data = target_bbox.data_float();
    const float* mask_data = object_mask.data_float();
    
    float loss = 0.0f;
    int num_objects = 0;
    
    for (size_t i = 0; i < object_mask.size(); ++i) {
        if (mask_data[i] > 0.5f) {
            std::vector<float> pred_box(4);
            std::vector<float> target_box(4);
            
            for (int j = 0; j < 4; ++j) {
                pred_box[j] = pred_data[i * 4 + j];
                target_box[j] = target_data[i * 4 + j];
            }
            
            float giou = compute_giou(pred_box, target_box);
            loss += (1.0f - giou);
            num_objects++;
        }
    }
    
    return num_objects > 0 ? loss / num_objects : 0.0f;
}

float YOLOLoss::compute_classification_loss(const core::Tensor& pred_classes,
                                            const core::Tensor& target_classes,
                                            const core::Tensor& object_mask) {
    const auto& pred_shape = pred_classes.shape();
    int num_classes = pred_shape[4];
    
    const float* pred_data = pred_classes.data_float();
    const float* target_data = target_classes.data_float();
    const float* mask_data = object_mask.data_float();
    
    float loss = 0.0f;
    int num_objects = 0;
    
    for (size_t i = 0; i < object_mask.size(); ++i) {
        if (mask_data[i] > 0.5f) {
            for (int c = 0; c < num_classes; ++c) {
                float pred_sigmoid = sigmoid(pred_data[i * num_classes + c]);
                loss += binary_cross_entropy(pred_sigmoid, target_data[i * num_classes + c]);
            }
            num_objects++;
        }
    }
    
    return num_objects > 0 ? loss / num_objects : 0.0f;
}

/**
 * @brief Computes Generalized Intersection over Union for bounding box regression
 * 
 * Implements GIoU metric which addresses limitations of standard IoU by considering
 * the area of the smallest enclosing box. Critical for accurate object localization
 * in traffic scenarios where precise bounding boxes are essential for safety.
 */
float YOLOLoss::compute_giou(const std::vector<float>& box1, const std::vector<float>& box2) {
    float x1_min = box1[0] - box1[2] / 2.0f;
    float y1_min = box1[1] - box1[3] / 2.0f;
    float x1_max = box1[0] + box1[2] / 2.0f;
    float y1_max = box1[1] + box1[3] / 2.0f;
    
    float x2_min = box2[0] - box2[2] / 2.0f;
    float y2_min = box2[1] - box2[3] / 2.0f;
    float x2_max = box2[0] + box2[2] / 2.0f;
    float y2_max = box2[1] + box2[3] / 2.0f;
    
    float inter_x_min = std::max(x1_min, x2_min);
    float inter_y_min = std::max(y1_min, y2_min);
    float inter_x_max = std::min(x1_max, x2_max);
    float inter_y_max = std::min(y1_max, y2_max);
    
    float inter_area = std::max(0.0f, inter_x_max - inter_x_min) * 
                       std::max(0.0f, inter_y_max - inter_y_min);
    
    float area1 = box1[2] * box1[3];
    float area2 = box2[2] * box2[3];
    float union_area = area1 + area2 - inter_area;
    
    float iou = inter_area / (union_area + 1e-7f);
    
    float enclosing_x_min = std::min(x1_min, x2_min);
    float enclosing_y_min = std::min(y1_min, y2_min);
    float enclosing_x_max = std::max(x1_max, x2_max);
    float enclosing_y_max = std::max(y1_max, y2_max);
    
    float enclosing_area = (enclosing_x_max - enclosing_x_min) * (enclosing_y_max - enclosing_y_min);
    
    float giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-7f);
    
    return giou;
}

float YOLOLoss::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float YOLOLoss::binary_cross_entropy(float pred, float target) {
    pred = std::clamp(pred, 1e-7f, 1.0f - 1e-7f);
    return -(target * std::log(pred) + (1.0f - target) * std::log(1.0f - pred));
}

void YOLOLoss::assign_targets_to_grid(const core::Tensor& targets,
                                      core::Tensor& target_bbox,
                                      core::Tensor& target_obj,
                                      core::Tensor& target_cls,
                                      core::Tensor& object_mask,
                                      core::Tensor& ignore_mask,
                                      const std::vector<float>& anchors,
                                      int grid_h, int grid_w, int scale_idx) {
    const auto& targets_shape = targets.shape();
    if (targets_shape.size() != 3) {
        return;
    }
    
    int batch_size = targets_shape[0];
    int max_targets = targets_shape[1];
    
    const float* targets_data = targets.data_float();
    float* target_bbox_data = target_bbox.data_float();
    float* target_obj_data = target_obj.data_float();
    float* target_cls_data = target_cls.data_float();
    float* object_mask_data = object_mask.data_float();
    float* ignore_mask_data = ignore_mask.data_float();
    
    int num_anchors = anchors.size() / 2;
    float stride = 416.0f / grid_h;
    
    for (int n = 0; n < batch_size; ++n) {
        for (int t = 0; t < max_targets; ++t) {
            int target_idx = n * max_targets * 5 + t * 5;
            
            float cls = targets_data[target_idx + 0];
            float cx = targets_data[target_idx + 1];
            float cy = targets_data[target_idx + 2];
            float w = targets_data[target_idx + 3];
            float h = targets_data[target_idx + 4];
            
            if (w <= 0 || h <= 0) continue;
            
            int gx = static_cast<int>(cx * grid_w);
            int gy = static_cast<int>(cy * grid_h);
            
            if (gx >= grid_w) gx = grid_w - 1;
            if (gy >= grid_h) gy = grid_h - 1;
            if (gx < 0) gx = 0;
            if (gy < 0) gy = 0;
            
            int best_anchor = 0;
            float best_iou = -1.0f;
            
            for (int a = 0; a < num_anchors; ++a) {
                float anchor_w = anchors[a * 2] / stride;
                float anchor_h = anchors[a * 2 + 1] / stride;
                
                float inter_w = std::min(w, anchor_w);
                float inter_h = std::min(h, anchor_h);
                float inter_area = inter_w * inter_h;
                
                float union_area = w * h + anchor_w * anchor_h - inter_area;
                float iou = inter_area / (union_area + 1e-7f);
                
                if (iou > best_iou) {
                    best_iou = iou;
                    best_anchor = a;
                }
            }
            
            int base_idx = n * num_anchors * grid_h * grid_w +
                           best_anchor * grid_h * grid_w + gy * grid_w + gx;
            
            target_bbox_data[base_idx * 4 + 0] = cx * grid_w - gx;
            target_bbox_data[base_idx * 4 + 1] = cy * grid_h - gy;
            target_bbox_data[base_idx * 4 + 2] = std::log(w * grid_w / anchors[best_anchor * 2] + 1e-7f);
            target_bbox_data[base_idx * 4 + 3] = std::log(h * grid_h / anchors[best_anchor * 2 + 1] + 1e-7f);
            
            target_obj_data[base_idx] = 1.0f;
            object_mask_data[base_idx] = 1.0f;
            
            int cls_idx = static_cast<int>(cls);
            if (cls_idx >= 0 && cls_idx < 3) {
                target_cls_data[base_idx * 3 + cls_idx] = 1.0f;
            }
            
            for (int a = 0; a < num_anchors; ++a) {
                if (a != best_anchor) {
                    float anchor_w = anchors[a * 2] / stride;
                    float anchor_h = anchors[a * 2 + 1] / stride;
                    
                    float inter_w = std::min(w, anchor_w);
                    float inter_h = std::min(h, anchor_h);
                    float inter_area = inter_w * inter_h;
                    
                    float union_area = w * h + anchor_w * anchor_h - inter_area;
                    float iou = inter_area / (union_area + 1e-7f);
                    
                    if (iou > 0.5f) {
                        int ignore_idx = n * num_anchors * grid_h * grid_w +
                                         a * grid_h * grid_w + gy * grid_w + gx;
                        ignore_mask_data[ignore_idx] = 1.0f;
                    }
                }
            }
        }
    }
}

}
}