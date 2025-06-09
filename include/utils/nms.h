/**
 * @file nms.h
 * @brief Non-Maximum Suppression for multi-class object detection
 * 
 * Implements efficient NMS with class-specific thresholding optimized for
 * real-time traffic object detection. Supports cars, pedestrians, and cyclists
 * with configurable per-class confidence and IoU thresholds.
 */
#pragma once

#include "core/tensor.h"
#include "models/tacsnet.h"
#include <vector>

namespace tacs {
namespace utils {

struct NMSDetection {
    float x, y, w, h;
    float confidence;
    int class_id;
    float class_prob;
};

struct NMSConfig {
    float iou_threshold = 0.45f;
    std::vector<float> class_confidence_thresholds = {0.5f, 0.4f, 0.4f}; // cars, pedestrians, cyclists
    int max_detections = 100;
};

class NonMaxSuppression {
public:
    explicit NonMaxSuppression(const NMSConfig& config = NMSConfig{});
    
    std::vector<NMSDetection> apply(const std::vector<models::DetectionOutput>& predictions,
                                     const std::vector<std::vector<float>>& anchors,
                                     int image_width, int image_height);
    
    void set_config(const NMSConfig& config) { config_ = config; }
    const NMSConfig& get_config() const { return config_; }

private:
    NMSConfig config_;
    
    std::vector<NMSDetection> decode_predictions(const models::DetectionOutput& prediction,
                                                  const std::vector<float>& anchors,
                                                  int grid_w, int grid_h,
                                                  float stride,
                                                  int image_width, int image_height);
    
    std::vector<NMSDetection> apply_nms_per_class(std::vector<NMSDetection>& detections);
    
    float compute_iou(const NMSDetection& det1, const NMSDetection& det2);
    float sigmoid(float x);
};

}
}