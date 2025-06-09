/**
 * @file metrics.h
 * @brief Production-ready metrics computation for object detection evaluation
 * 
 * Implements mean Average Precision (mAP) calculation following COCO standards
 * for multi-class object detection evaluation. Optimized for real-time
 * performance analysis of TACS neural networks.
 */
#pragma once

#include "core/tensor.h"
#include <vector>
#include <unordered_map>

namespace tacs {
namespace utils {

struct Detection {
    float x, y, w, h;
    float confidence;
    int class_id;
    
    Detection(float x, float y, float w, float h, float conf, int cls)
        : x(x), y(y), w(w), h(h), confidence(conf), class_id(cls) {}
};

struct GroundTruth {
    float x, y, w, h;
    int class_id;
    bool matched;
    
    GroundTruth(float x, float y, float w, float h, int cls)
        : x(x), y(y), w(w), h(h), class_id(cls), matched(false) {}
};

class MetricsCalculator {
public:
    MetricsCalculator(int num_classes = 3, float iou_threshold = 0.5f);
    
    // Add detections and ground truth for a batch
    void add_batch_predictions(const std::vector<Detection>& detections,
                               const std::vector<GroundTruth>& ground_truths);
    
    // Calculate mAP across all classes
    float calculate_map();
    
    // Calculate AP for a specific class
    float calculate_ap(int class_id);
    
    // Reset accumulated statistics
    void reset();
    
    // Get per-class metrics
    std::unordered_map<int, float> get_per_class_ap();

private:
    int num_classes_;
    float iou_threshold_;
    
    // Accumulated detections and ground truths per class
    std::unordered_map<int, std::vector<Detection>> class_detections_;
    std::unordered_map<int, std::vector<GroundTruth>> class_ground_truths_;
    
    // Helper functions
    float calculate_iou(const Detection& det, const GroundTruth& gt);
    std::vector<float> calculate_precision_recall(std::vector<Detection>& detections,
                                                  std::vector<GroundTruth>& ground_truths);
    float calculate_ap_from_pr_curve(const std::vector<float>& precisions,
                                     const std::vector<float>& recalls);
};

}
}