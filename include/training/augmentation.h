/**
 * @file augmentation.h
 * @brief Enhanced data augmentation for 99% accuracy
 */
#pragma once

#include "core/tensor.h"
#include "data/data_loader.h"
#include <random>
#include <vector>

namespace tacs {
namespace training {

class DataAugmentation {
public:
    struct AugmentationConfig {
        // Geometric transformations
        float horizontal_flip_prob = 0.5f;
        float rotation_max_angle = 15.0f;  // degrees
        float scale_min = 0.8f;
        float scale_max = 1.2f;
        float translate_max = 0.1f;  // fraction of image size
        float shear_max = 0.1f;
        
        // Color augmentations
        float brightness_max = 0.3f;
        float contrast_max = 0.3f;
        float saturation_max = 0.3f;
        float hue_max = 0.1f;
        
        // Advanced augmentations
        float mixup_alpha = 0.2f;
        float cutout_prob = 0.5f;
        float cutout_max_size = 0.2f;  // fraction of image
        float mosaic_prob = 0.5f;
        
        // Noise and blur
        float gaussian_noise_std = 0.01f;
        float blur_prob = 0.1f;
        float blur_kernel_size = 5;
    };
    
    explicit DataAugmentation(const AugmentationConfig& config = AugmentationConfig{});
    
    // Apply augmentations to a batch
    void augment_batch(std::vector<data::Sample>& batch);
    
    // Individual augmentation functions
    void apply_horizontal_flip(core::Tensor& image, std::vector<data::BoundingBox>& boxes);
    void apply_rotation(core::Tensor& image, std::vector<data::BoundingBox>& boxes, float angle);
    void apply_scale(core::Tensor& image, std::vector<data::BoundingBox>& boxes, float scale);
    void apply_translation(core::Tensor& image, std::vector<data::BoundingBox>& boxes, float dx, float dy);
    void apply_color_jitter(core::Tensor& image);
    void apply_cutout(core::Tensor& image);
    void apply_mixup(core::Tensor& image1, core::Tensor& image2, 
                     std::vector<data::BoundingBox>& boxes1, 
                     std::vector<data::BoundingBox>& boxes2, float alpha);
    void apply_mosaic(std::vector<data::Sample>& samples);
    
private:
    AugmentationConfig config_;
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_dist_;
    std::normal_distribution<float> normal_dist_;
    
    // Helper functions
    void adjust_brightness(core::Tensor& image, float delta);
    void adjust_contrast(core::Tensor& image, float factor);
    void adjust_saturation(core::Tensor& image, float factor);
    void adjust_hue(core::Tensor& image, float delta);
    void add_gaussian_noise(core::Tensor& image, float std_dev);
    void apply_gaussian_blur(core::Tensor& image, int kernel_size);
    
    // Bounding box transformations
    void transform_bbox(data::BoundingBox& box, float angle, float scale, float dx, float dy, int img_w, int img_h);
    bool is_bbox_valid(const data::BoundingBox& box, int img_w, int img_h);
    void clip_bbox(data::BoundingBox& box, int img_w, int img_h);
};

}
}