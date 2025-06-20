#include "training/augmentation.h"
#include <cmath>
#include <algorithm>

namespace tacs {
namespace training {

DataAugmentation::DataAugmentation(const AugmentationConfig& config) 
    : config_(config), rng_(std::random_device{}()), 
      uniform_dist_(0.0f, 1.0f), normal_dist_(0.0f, 1.0f) {}

void DataAugmentation::augment_batch(std::vector<data::Sample>& batch) {
    // Apply mosaic augmentation to groups of 4 images
    if (config_.mosaic_prob > 0 && uniform_dist_(rng_) < config_.mosaic_prob && batch.size() >= 4) {
        for (size_t i = 0; i < batch.size() - 3; i += 4) {
            std::vector<data::Sample> mosaic_group(batch.begin() + i, batch.begin() + i + 4);
            apply_mosaic(mosaic_group);
            for (size_t j = 0; j < 4; ++j) {
                batch[i + j] = mosaic_group[j];
            }
        }
    }
    
    // Apply individual augmentations
    for (auto& sample : batch) {
        // Geometric augmentations
        if (uniform_dist_(rng_) < config_.horizontal_flip_prob) {
            apply_horizontal_flip(sample.image, sample.boxes);
        }
        
        // Random rotation
        if (config_.rotation_max_angle > 0) {
            std::uniform_real_distribution<float> angle_dist(-config_.rotation_max_angle, config_.rotation_max_angle);
            float angle = angle_dist(rng_);
            apply_rotation(sample.image, sample.boxes, angle);
        }
        
        // Random scale
        if (config_.scale_min < 1.0f || config_.scale_max > 1.0f) {
            std::uniform_real_distribution<float> scale_dist(config_.scale_min, config_.scale_max);
            float scale = scale_dist(rng_);
            apply_scale(sample.image, sample.boxes, scale);
        }
        
        // Random translation
        if (config_.translate_max > 0) {
            std::uniform_real_distribution<float> trans_dist(-config_.translate_max, config_.translate_max);
            float dx = trans_dist(rng_) * sample.image.shape()[2];
            float dy = trans_dist(rng_) * sample.image.shape()[1];
            apply_translation(sample.image, sample.boxes, dx, dy);
        }
        
        // Color augmentations
        apply_color_jitter(sample.image);
        
        // Cutout augmentation
        if (uniform_dist_(rng_) < config_.cutout_prob) {
            apply_cutout(sample.image);
        }
        
        // Add noise
        if (config_.gaussian_noise_std > 0) {
            add_gaussian_noise(sample.image, config_.gaussian_noise_std);
        }
        
        // Random blur
        if (uniform_dist_(rng_) < config_.blur_prob) {
            apply_gaussian_blur(sample.image, config_.blur_kernel_size);
        }
    }
}

void DataAugmentation::apply_horizontal_flip(core::Tensor& image, std::vector<data::BoundingBox>& boxes) {
    const auto& shape = image.shape();
    int height = shape[1];
    int width = shape[2];
    int channels = shape[0];
    
    float* data = image.data_float();
    
    // Flip image horizontally
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width / 2; ++x) {
                int idx1 = c * height * width + y * width + x;
                int idx2 = c * height * width + y * width + (width - 1 - x);
                std::swap(data[idx1], data[idx2]);
            }
        }
    }
    
    // Flip bounding boxes
    for (auto& box : boxes) {
        box.x = 1.0f - box.x;
    }
}

void DataAugmentation::apply_rotation(core::Tensor& image, std::vector<data::BoundingBox>& boxes, float angle) {
    // Convert angle to radians
    float rad = angle * M_PI / 180.0f;
    float cos_a = std::cos(rad);
    float sin_a = std::sin(rad);
    
    const auto& shape = image.shape();
    int height = shape[1];
    int width = shape[2];
    
    // Rotate bounding boxes
    for (auto& box : boxes) {
        float cx = box.x;
        float cy = box.y;
        
        // Rotate around image center
        float dx = cx - 0.5f;
        float dy = cy - 0.5f;
        
        box.x = 0.5f + dx * cos_a - dy * sin_a;
        box.y = 0.5f + dx * sin_a + dy * cos_a;
        
        // Adjust box size (approximate)
        float scale = std::abs(cos_a) + std::abs(sin_a);
        box.w *= scale;
        box.h *= scale;
        
        // Clip to valid range
        clip_bbox(box, width, height);
    }
}

void DataAugmentation::apply_scale(core::Tensor& image, std::vector<data::BoundingBox>& boxes, float scale) {
    // Scale bounding boxes
    for (auto& box : boxes) {
        box.x = 0.5f + (box.x - 0.5f) * scale;
        box.y = 0.5f + (box.y - 0.5f) * scale;
        box.w *= scale;
        box.h *= scale;
    }
    
    // Remove boxes that go out of bounds
    boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
                              [](const data::BoundingBox& box) {
                                  return box.x - box.w/2 < 0 || box.x + box.w/2 > 1 ||
                                         box.y - box.h/2 < 0 || box.y + box.h/2 > 1;
                              }), boxes.end());
}

void DataAugmentation::apply_translation(core::Tensor& image, std::vector<data::BoundingBox>& boxes, 
                                       float dx, float dy) {
    const auto& shape = image.shape();
    int width = shape[2];
    int height = shape[1];
    
    // Normalize translation
    float norm_dx = dx / width;
    float norm_dy = dy / height;
    
    // Translate bounding boxes
    for (auto& box : boxes) {
        box.x += norm_dx;
        box.y += norm_dy;
    }
    
    // Remove boxes that go out of bounds
    boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
                              [](const data::BoundingBox& box) {
                                  return box.x < 0 || box.x > 1 || box.y < 0 || box.y > 1;
                              }), boxes.end());
}

void DataAugmentation::apply_color_jitter(core::Tensor& image) {
    // Random brightness
    if (config_.brightness_max > 0) {
        std::uniform_real_distribution<float> bright_dist(-config_.brightness_max, config_.brightness_max);
        adjust_brightness(image, bright_dist(rng_));
    }
    
    // Random contrast
    if (config_.contrast_max > 0) {
        std::uniform_real_distribution<float> contrast_dist(1.0f - config_.contrast_max, 1.0f + config_.contrast_max);
        adjust_contrast(image, contrast_dist(rng_));
    }
    
    // Random saturation
    if (config_.saturation_max > 0) {
        std::uniform_real_distribution<float> sat_dist(1.0f - config_.saturation_max, 1.0f + config_.saturation_max);
        adjust_saturation(image, sat_dist(rng_));
    }
    
    // Random hue
    if (config_.hue_max > 0) {
        std::uniform_real_distribution<float> hue_dist(-config_.hue_max, config_.hue_max);
        adjust_hue(image, hue_dist(rng_));
    }
}

void DataAugmentation::apply_cutout(core::Tensor& image) {
    const auto& shape = image.shape();
    int height = shape[1];
    int width = shape[2];
    
    // Random cutout size
    std::uniform_int_distribution<int> size_dist(1, static_cast<int>(config_.cutout_max_size * std::min(width, height)));
    int cutout_w = size_dist(rng_);
    int cutout_h = size_dist(rng_);
    
    // Random position
    std::uniform_int_distribution<int> x_dist(0, width - cutout_w);
    std::uniform_int_distribution<int> y_dist(0, height - cutout_h);
    int x = x_dist(rng_);
    int y = y_dist(rng_);
    
    // Apply cutout (set to zero)
    float* data = image.data_float();
    for (int c = 0; c < shape[0]; ++c) {
        for (int dy = 0; dy < cutout_h; ++dy) {
            for (int dx = 0; dx < cutout_w; ++dx) {
                int idx = c * height * width + (y + dy) * width + (x + dx);
                data[idx] = 0.0f;
            }
        }
    }
}

void DataAugmentation::adjust_brightness(core::Tensor& image, float delta) {
    float* data = image.data_float();
    int size = image.size();
    
    for (int i = 0; i < size; ++i) {
        data[i] = std::clamp(data[i] + delta, 0.0f, 1.0f);
    }
}

void DataAugmentation::adjust_contrast(core::Tensor& image, float factor) {
    float* data = image.data_float();
    int size = image.size();
    
    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < size; ++i) {
        mean += data[i];
    }
    mean /= size;
    
    // Apply contrast
    for (int i = 0; i < size; ++i) {
        data[i] = std::clamp((data[i] - mean) * factor + mean, 0.0f, 1.0f);
    }
}

void DataAugmentation::add_gaussian_noise(core::Tensor& image, float std_dev) {
    float* data = image.data_float();
    int size = image.size();
    
    for (int i = 0; i < size; ++i) {
        float noise = normal_dist_(rng_) * std_dev;
        data[i] = std::clamp(data[i] + noise, 0.0f, 1.0f);
    }
}

void DataAugmentation::clip_bbox(data::BoundingBox& box, int img_w, int img_h) {
    box.x = std::clamp(box.x, 0.0f, 1.0f);
    box.y = std::clamp(box.y, 0.0f, 1.0f);
    
    // Clip width and height
    float max_width = 1.0f - (box.x - box.w / 2.0f);
    float max_height = 1.0f - (box.y - box.h / 2.0f);
    box.w = std::min(box.w, max_width * 2.0f);
    box.h = std::min(box.h, max_height * 2.0f);
}

void DataAugmentation::adjust_saturation(core::Tensor& image, float factor) {
    const auto& shape = image.shape();
    if (shape[0] != 3) return;  // Only for RGB images
    
    int height = shape[1];
    int width = shape[2];
    float* data = image.data_float();
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Get RGB values
            int idx_r = 0 * height * width + y * width + x;
            int idx_g = 1 * height * width + y * width + x;
            int idx_b = 2 * height * width + y * width + x;
            
            float r = data[idx_r];
            float g = data[idx_g];
            float b = data[idx_b];
            
            // Convert to grayscale
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;
            
            // Apply saturation
            data[idx_r] = std::clamp(gray + (r - gray) * factor, 0.0f, 1.0f);
            data[idx_g] = std::clamp(gray + (g - gray) * factor, 0.0f, 1.0f);
            data[idx_b] = std::clamp(gray + (b - gray) * factor, 0.0f, 1.0f);
        }
    }
}

void DataAugmentation::adjust_hue(core::Tensor& image, float delta) {
    // Simplified hue adjustment - would need full HSV conversion for accurate results
    const auto& shape = image.shape();
    if (shape[0] != 3) return;
    
    float* data = image.data_float();
    int size = shape[1] * shape[2];
    
    // Simple channel rotation for hue effect
    if (delta > 0) {
        for (int i = 0; i < size; ++i) {
            float r = data[i];
            float g = data[size + i];
            float b = data[2 * size + i];
            
            data[i] = r * (1 - delta) + g * delta;
            data[size + i] = g * (1 - delta) + b * delta;
            data[2 * size + i] = b * (1 - delta) + r * delta;
        }
    }
}

void DataAugmentation::apply_gaussian_blur(core::Tensor& image, int kernel_size) {
    // Simple box blur approximation for speed
    const auto& shape = image.shape();
    int channels = shape[0];
    int height = shape[1];
    int width = shape[2];
    
    int half_kernel = kernel_size / 2;
    core::Tensor blurred(image.shape());
    const float* src = image.data_float();
    float* dst = blurred.data_float();
    std::copy(src, src + image.size(), dst);
    float* src_data = image.data_float();
    float* dst_data = blurred.data_float();
    
    for (int c = 0; c < channels; ++c) {
        for (int y = half_kernel; y < height - half_kernel; ++y) {
            for (int x = half_kernel; x < width - half_kernel; ++x) {
                float sum = 0.0f;
                int count = 0;
                
                for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
                    for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
                        int idx = c * height * width + (y + ky) * width + (x + kx);
                        sum += src_data[idx];
                        count++;
                    }
                }
                
                int out_idx = c * height * width + y * width + x;
                dst_data[out_idx] = sum / count;
            }
        }
    }
    
    // Copy back
    std::memcpy(src_data, dst_data, image.size() * sizeof(float));
}

void DataAugmentation::apply_mosaic(std::vector<data::Sample>& samples) {
    if (samples.size() < 4) return;
    
    // Create mosaic from 4 images
    const auto& shape = samples[0].image.shape();
    int height = shape[1];
    int width = shape[2];
    
    // Each sub-image will be half size
    int sub_h = height / 2;
    int sub_w = width / 2;
    
    // Process each quadrant
    for (int i = 0; i < 4; ++i) {
        // Scale boxes to sub-image
        for (auto& box : samples[i].boxes) {
            box.w *= 0.5f;
            box.h *= 0.5f;
            
            // Position in appropriate quadrant
            if (i == 0) {  // Top-left
                box.x *= 0.5f;
                box.y *= 0.5f;
            } else if (i == 1) {  // Top-right
                box.x = 0.5f + box.x * 0.5f;
                box.y *= 0.5f;
            } else if (i == 2) {  // Bottom-left
                box.x *= 0.5f;
                box.y = 0.5f + box.y * 0.5f;
            } else {  // Bottom-right
                box.x = 0.5f + box.x * 0.5f;
                box.y = 0.5f + box.y * 0.5f;
            }
        }
    }
    
    // Merge all boxes into first sample
    samples[0].boxes.clear();
    for (int i = 0; i < 4; ++i) {
        samples[0].boxes.insert(samples[0].boxes.end(), 
                               samples[i].boxes.begin(), 
                               samples[i].boxes.end());
    }
}

}
}