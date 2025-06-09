#include "utils/metrics.h"
#include <algorithm>
#include <numeric>

namespace tacs {
namespace utils {

MetricsCalculator::MetricsCalculator(int num_classes, float iou_threshold)
    : num_classes_(num_classes), iou_threshold_(iou_threshold) {}

void MetricsCalculator::add_batch_predictions(const std::vector<Detection>& detections,
                                              const std::vector<GroundTruth>& ground_truths) {
    // Sort detections by class
    for (const auto& det : detections) {
        if (det.class_id >= 0 && det.class_id < num_classes_) {
            class_detections_[det.class_id].push_back(det);
        }
    }
    
    // Sort ground truths by class
    for (const auto& gt : ground_truths) {
        if (gt.class_id >= 0 && gt.class_id < num_classes_) {
            class_ground_truths_[gt.class_id].push_back(gt);
        }
    }
}

float MetricsCalculator::calculate_map() {
    float total_ap = 0.0f;
    int valid_classes = 0;
    
    for (int cls = 0; cls < num_classes_; ++cls) {
        float ap = calculate_ap(cls);
        if (ap >= 0.0f) {  // Valid AP (class has ground truths)
            total_ap += ap;
            valid_classes++;
        }
    }
    
    return valid_classes > 0 ? total_ap / valid_classes : 0.0f;
}

float MetricsCalculator::calculate_ap(int class_id) {
    if (class_ground_truths_[class_id].empty()) {
        return -1.0f;  // No ground truths for this class
    }
    
    auto& detections = class_detections_[class_id];
    auto& ground_truths = class_ground_truths_[class_id];
    
    // Sort detections by confidence in descending order
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    
    // Reset matched flags
    for (auto& gt : ground_truths) {
        gt.matched = false;
    }
    
    std::vector<float> precisions, recalls;
    int true_positives = 0;
    int false_positives = 0;
    int total_ground_truths = ground_truths.size();
    
    for (const auto& det : detections) {
        float best_iou = 0.0f;
        int best_gt_idx = -1;
        
        // Find best matching ground truth
        for (size_t i = 0; i < ground_truths.size(); ++i) {
            if (!ground_truths[i].matched) {
                float iou = calculate_iou(det, ground_truths[i]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_gt_idx = i;
                }
            }
        }
        
        // Check if detection is true positive
        if (best_iou >= iou_threshold_ && best_gt_idx >= 0) {
            true_positives++;
            ground_truths[best_gt_idx].matched = true;
        } else {
            false_positives++;
        }
        
        // Calculate precision and recall at this point
        float precision = static_cast<float>(true_positives) / 
                         (true_positives + false_positives);
        float recall = static_cast<float>(true_positives) / total_ground_truths;
        
        precisions.push_back(precision);
        recalls.push_back(recall);
    }
    
    return calculate_ap_from_pr_curve(precisions, recalls);
}

void MetricsCalculator::reset() {
    class_detections_.clear();
    class_ground_truths_.clear();
}

std::unordered_map<int, float> MetricsCalculator::get_per_class_ap() {
    std::unordered_map<int, float> per_class_ap;
    
    for (int cls = 0; cls < num_classes_; ++cls) {
        float ap = calculate_ap(cls);
        if (ap >= 0.0f) {
            per_class_ap[cls] = ap;
        }
    }
    
    return per_class_ap;
}

float MetricsCalculator::calculate_iou(const Detection& det, const GroundTruth& gt) {
    // Convert from center format to corner format
    float det_x1 = det.x - det.w / 2.0f;
    float det_y1 = det.y - det.h / 2.0f;
    float det_x2 = det.x + det.w / 2.0f;
    float det_y2 = det.y + det.h / 2.0f;
    
    float gt_x1 = gt.x - gt.w / 2.0f;
    float gt_y1 = gt.y - gt.h / 2.0f;
    float gt_x2 = gt.x + gt.w / 2.0f;
    float gt_y2 = gt.y + gt.h / 2.0f;
    
    // Calculate intersection
    float inter_x1 = std::max(det_x1, gt_x1);
    float inter_y1 = std::max(det_y1, gt_y1);
    float inter_x2 = std::min(det_x2, gt_x2);
    float inter_y2 = std::min(det_y2, gt_y2);
    
    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * 
                       std::max(0.0f, inter_y2 - inter_y1);
    
    // Calculate union
    float det_area = det.w * det.h;
    float gt_area = gt.w * gt.h;
    float union_area = det_area + gt_area - inter_area;
    
    return union_area > 0.0f ? inter_area / union_area : 0.0f;
}

float MetricsCalculator::calculate_ap_from_pr_curve(const std::vector<float>& precisions,
                                                    const std::vector<float>& recalls) {
    if (precisions.empty() || recalls.empty()) {
        return 0.0f;
    }
    
    // 11-point interpolation (PASCAL VOC style)
    float ap = 0.0f;
    
    for (float r = 0.0f; r <= 1.0f; r += 0.1f) {
        float max_precision = 0.0f;
        
        for (size_t i = 0; i < recalls.size(); ++i) {
            if (recalls[i] >= r) {
                max_precision = std::max(max_precision, precisions[i]);
            }
        }
        
        ap += max_precision;
    }
    
    return ap / 11.0f;
}

}
}