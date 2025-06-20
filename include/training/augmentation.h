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
        float horizontal_flip_prob;
        float rotation_max_angle;  // degrees
        float scale_min;
        float scale_max;
        float translate_max;  // fraction of image size
        float shear_max;
        
        // Color augmentations
        float brightness_max;
        float contrast_max;
        float saturation_max;
        float hue_max;
        
        // Advanced augmentations
        float mixup_alpha;
        float cutout_prob;
        float cutout_max_size;  // fraction of image
        float mosaic_prob;
        
        // Noise and blur
        float gaussian_noise_std;
        float blur_prob;
        float blur_kernel_size;
        
        // Constructor with default values
        AugmentationConfig() : 
            horizontal_flip_prob(0.5f),
            rotation_max_angle(15.0f),
            scale_min(0.8f),
            scale_max(1.2f),
            translate_max(0.1f),
            shear_max(0.1f),
            brightness_max(0.3f),
            contrast_max(0.3f),
            saturation_max(0.3f),
            hue_max(0.1f),
            mixup_alpha(0.2f),
            cutout_prob(0.5f),
            cutout_max_size(0.2f),
            mosaic_prob(0.5f),
            gaussian_noise_std(0.01f),
            blur_prob(0.1f),
            blur_kernel_size(5) {}
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