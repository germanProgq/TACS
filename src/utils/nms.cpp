#include "utils/nms.h"
#include <algorithm>
#include <cmath>

// Compatibility for older compilers
#ifndef __cpp_lib_clamp
namespace std {
    template<typename T>
    constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
        return (v < lo) ? lo : (hi < v) ? hi : v;
    }
}
#endif

namespace tacs {
namespace utils {

NonMaxSuppression::NonMaxSuppression(const NMSConfig& config) : config_(config) {}

std::vector<NMSDetection> NonMaxSuppression::apply(const std::vector<models::DetectionOutput>& predictions,
                                                    const std::vector<std::vector<float>>& anchors,
                                                    int image_width, int image_height) {
    std::vector<NMSDetection> all_detections;
    
    // Process each scale
    for (size_t scale = 0; scale < predictions.size(); ++scale) {
        const auto& pred = predictions[scale];
        const auto& scale_anchors = anchors[scale];
        
        const auto& bbox_shape = pred.bbox_predictions.shape();
        int batch_size = bbox_shape[0];
        int num_anchors = bbox_shape[1];
        int grid_h = bbox_shape[2];
        int grid_w = bbox_shape[3];
        
        float stride = static_cast<float>(image_width) / grid_w;
        
        auto scale_detections = decode_predictions(pred, scale_anchors, grid_w, grid_h, 
                                                   stride, image_width, image_height);
        
        all_detections.insert(all_detections.end(), scale_detections.begin(), scale_detections.end());
    }
    
    // Apply NMS per class
    return apply_nms_per_class(all_detections);
}

std::vector<NMSDetection> NonMaxSuppression::decode_predictions(const models::DetectionOutput& prediction,
                                                                 const std::vector<float>& anchors,
                                                                 int grid_w, int grid_h,
                                                                 float stride,
                                                                 int image_width, int image_height) {
    std::vector<NMSDetection> detections;
    
    const float* bbox_data = prediction.bbox_predictions.data_float();
    const float* obj_data = prediction.objectness_scores.data_float();
    const float* cls_data = prediction.class_predictions.data_float();
    
    const auto& bbox_shape = prediction.bbox_predictions.shape();
    int batch_size = bbox_shape[0];
    int num_anchors = bbox_shape[1];
    int num_classes = prediction.class_predictions.shape()[4];
    
    // Process only first batch for inference
    for (int a = 0; a < num_anchors; ++a) {
        float anchor_w = anchors[a * 2];
        float anchor_h = anchors[a * 2 + 1];
        
        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                int idx = a * grid_h * grid_w + gy * grid_w + gx;
                
                // Get objectness score
                float objectness = sigmoid(obj_data[idx]);
                
                // Skip low objectness
                if (objectness < 0.1f) continue;
                
                // Get bbox predictions
                float tx = sigmoid(bbox_data[idx * 4 + 0]);
                float ty = sigmoid(bbox_data[idx * 4 + 1]);
                float tw = bbox_data[idx * 4 + 2];
                float th = bbox_data[idx * 4 + 3];
                
                // Decode bbox
                float cx = (gx + tx) * stride;
                float cy = (gy + ty) * stride;
                float w = anchor_w * std::exp(tw);
                float h = anchor_h * std::exp(th);
                
                // Clip to image bounds
                cx = std::clamp(cx, 0.0f, static_cast<float>(image_width));
                cy = std::clamp(cy, 0.0f, static_cast<float>(image_height));
                w = std::clamp(w, 0.0f, static_cast<float>(image_width));
                h = std::clamp(h, 0.0f, static_cast<float>(image_height));
                
                // Get class probabilities
                float max_class_prob = 0.0f;
                int best_class = -1;
                
                for (int c = 0; c < num_classes; ++c) {
                    float class_prob = sigmoid(cls_data[idx * num_classes + c]);
                    if (class_prob > max_class_prob) {
                        max_class_prob = class_prob;
                        best_class = c;
                    }
                }
                
                // Calculate final confidence
                float confidence = objectness * max_class_prob;
                
                // Apply class-specific threshold
                if (best_class >= 0 && best_class < config_.class_confidence_thresholds.size() &&
                    confidence >= config_.class_confidence_thresholds[best_class]) {
                    NMSDetection det;
                    det.x = cx;
                    det.y = cy;
                    det.w = w;
                    det.h = h;
                    det.confidence = confidence;
                    det.class_id = best_class;
                    det.class_prob = max_class_prob;
                    detections.push_back(det);
                }
            }
        }
    }
    
    return detections;
}

std::vector<NMSDetection> NonMaxSuppression::apply_nms_per_class(std::vector<NMSDetection>& detections) {
    std::vector<NMSDetection> final_detections;
    
    // Group detections by class
    std::vector<std::vector<NMSDetection>> class_detections(3); // 3 classes
    
    for (const auto& det : detections) {
        if (det.class_id >= 0 && det.class_id < 3) {
            class_detections[det.class_id].push_back(det);
        }
    }
    
    // Apply NMS for each class separately
    for (int class_id = 0; class_id < 3; ++class_id) {
        auto& dets = class_detections[class_id];
        
        // Sort by confidence
        std::sort(dets.begin(), dets.end(), [](const NMSDetection& a, const NMSDetection& b) {
            return a.confidence > b.confidence;
        });
        
        std::vector<bool> keep(dets.size(), true);
        
        // Apply NMS
        for (size_t i = 0; i < dets.size(); ++i) {
            if (!keep[i]) continue;
            
            for (size_t j = i + 1; j < dets.size(); ++j) {
                if (!keep[j]) continue;
                
                float iou = compute_iou(dets[i], dets[j]);
                if (iou > config_.iou_threshold) {
                    keep[j] = false;
                }
            }
        }
        
        // Collect kept detections
        for (size_t i = 0; i < dets.size(); ++i) {
            if (keep[i]) {
                final_detections.push_back(dets[i]);
            }
        }
    }
    
    // Sort all detections by confidence and limit to max_detections
    std::sort(final_detections.begin(), final_detections.end(), 
              [](const NMSDetection& a, const NMSDetection& b) {
                  return a.confidence > b.confidence;
              });
    
    if (final_detections.size() > config_.max_detections) {
        final_detections.resize(config_.max_detections);
    }
    
    return final_detections;
}

float NonMaxSuppression::compute_iou(const NMSDetection& det1, const NMSDetection& det2) {
    float x1_min = det1.x - det1.w / 2.0f;
    float y1_min = det1.y - det1.h / 2.0f;
    float x1_max = det1.x + det1.w / 2.0f;
    float y1_max = det1.y + det1.h / 2.0f;
    
    float x2_min = det2.x - det2.w / 2.0f;
    float y2_min = det2.y - det2.h / 2.0f;
    float x2_max = det2.x + det2.w / 2.0f;
    float y2_max = det2.y + det2.h / 2.0f;
    
    float inter_x_min = std::max(x1_min, x2_min);
    float inter_y_min = std::max(y1_min, y2_min);
    float inter_x_max = std::min(x1_max, x2_max);
    float inter_y_max = std::min(y1_max, y2_max);
    
    float inter_area = std::max(0.0f, inter_x_max - inter_x_min) * 
                       std::max(0.0f, inter_y_max - inter_y_min);
    
    float area1 = det1.w * det1.h;
    float area2 = det2.w * det2.h;
    float union_area = area1 + area2 - inter_area;
    
    return inter_area / (union_area + 1e-7f);
}

float NonMaxSuppression::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

}
}