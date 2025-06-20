/**
 * @file adaptive_optimizer.h
 * @brief Adaptive optimizer with dynamic weight adjustment for improved accuracy
 * 
 * Implements advanced optimization techniques including:
 * - Dynamic learning rate adaptation based on gradient statistics
 * - Layer-wise learning rate scaling
 * - Accuracy-based weight adjustment
 * - Gradient variance tracking for stability
 */
#pragma once

#include "training/optimizer.h"
#include <deque>
#include <cmath>
#include <numeric>

namespace tacs {
namespace training {

// Statistics for adaptive optimization
struct GradientStatistics {
    float mean = 0.0f;
    float variance = 0.0f;
    float magnitude = 0.0f;
    float update_ratio = 0.0f;
    std::deque<float> history;
    static constexpr size_t HISTORY_SIZE = 100;
    
    void update(const core::Tensor& gradient);
    float get_adaptive_scale() const;
};

// Layer-wise adaptation state
struct LayerAdaptationState {
    float learning_rate_scale = 1.0f;
    float weight_decay_scale = 1.0f;
    float gradient_norm = 0.0f;
    float weight_norm = 0.0f;
    int stagnation_counter = 0;
    float last_loss_contribution = 0.0f;
};

// Configuration for adaptive optimization
struct AdaptiveConfig {
    // Base optimizer type
    bool use_adam_base = true;
    
    // Adaptive learning rate settings
    float lr_adaptation_rate = 0.01f;
    float min_lr_scale = 0.1f;
    float max_lr_scale = 10.0f;
    
    // Gradient statistics window
    int gradient_history_size = 100;
    float gradient_clip_percentile = 95.0f;
    
    // Layer-wise adaptation
    bool enable_layer_wise_lr = true;
    float layer_lr_decay = 0.9f;  // Deeper layers get smaller LR
    
    // Accuracy-based adjustment
    bool enable_accuracy_feedback = true;
    float accuracy_momentum = 0.95f;
    float accuracy_threshold = 0.99f;
    
    // Stability controls
    float min_gradient_norm = 1e-8f;
    float max_gradient_norm = 10.0f;
    bool enable_gradient_centralization = true;
    
    // Weight adjustment parameters
    float weight_adjustment_rate = 0.001f;
    float weight_noise_scale = 0.0001f;
    bool enable_weight_standardization = true;
};

class AdaptiveOptimizer : public Optimizer {
public:
    AdaptiveOptimizer(const AdaptiveConfig& config = AdaptiveConfig());
    
    void step() override;
    void zero_grad() override;
    float get_learning_rate() const override;
    void set_learning_rate(float lr) override;
    
    // Parameter management
    void add_parameter_group(const std::string& name, core::Tensor& parameter) override;
    void set_gradient(const std::string& name, const core::Tensor& gradient) override;
    
    // Adaptive features
    void update_accuracy_feedback(float current_accuracy, float target_accuracy = 0.99f);
    void update_loss_feedback(float current_loss, const std::unordered_map<std::string, float>& layer_contributions = {});
    
    // Layer-wise adaptation
    void set_layer_adaptation(const std::string& layer_name, const LayerAdaptationState& state);
    LayerAdaptationState get_layer_adaptation(const std::string& layer_name) const;
    
    // Statistics and monitoring
    std::unordered_map<std::string, GradientStatistics> get_gradient_statistics() const;
    float get_effective_learning_rate(const std::string& param_name) const;
    
    // Advanced features
    void enable_lookahead(int k = 5, float alpha = 0.5f);
    void enable_gradient_centralization(bool enable = true);
    void enable_weight_standardization(bool enable = true);
    
    // Reset adaptation state
    void reset_adaptation();
    
private:
    AdaptiveConfig config_;
    std::unique_ptr<Optimizer> base_optimizer_;
    
    // Store our own parameter references
    std::unordered_map<std::string, ParameterInfo> parameters_;
    
    // Adaptation state
    std::unordered_map<std::string, GradientStatistics> gradient_stats_;
    std::unordered_map<std::string, LayerAdaptationState> layer_states_;
    std::unordered_map<std::string, core::Tensor> param_momentum_;
    
    // Global adaptation metrics
    float global_lr_scale_ = 1.0f;
    float accuracy_feedback_scale_ = 1.0f;
    float loss_feedback_scale_ = 1.0f;
    std::deque<float> accuracy_history_;
    std::deque<float> loss_history_;
    
    // Lookahead state
    bool lookahead_enabled_ = false;
    int lookahead_k_ = 5;
    float lookahead_alpha_ = 0.5f;
    int lookahead_step_ = 0;
    std::unordered_map<std::string, core::Tensor> slow_weights_;
    
    // Helper methods
    void update_gradient_statistics(const std::string& name, const core::Tensor& gradient);
    float compute_adaptive_lr_scale(const std::string& name) const;
    void apply_gradient_centralization(core::Tensor& gradient);
    void apply_weight_standardization(core::Tensor& weights);
    void update_parameter_adaptive(const std::string& name, core::Tensor& parameter);
    void apply_weight_noise(core::Tensor& weights, float scale);
    
    // Stability and normalization
    float compute_gradient_norm(const core::Tensor& gradient) const;
    void clip_gradient_norm(core::Tensor& gradient, float max_norm);
    float compute_update_ratio(const core::Tensor& update, const core::Tensor& parameter) const;
};

// Helper class for tracking optimization metrics
class OptimizationMonitor {
public:
    void record_gradient(const std::string& param_name, float grad_norm);
    void record_weight_update(const std::string& param_name, float update_norm);
    void record_learning_rate(const std::string& param_name, float lr);
    void record_accuracy(float accuracy);
    void record_loss(float loss);
    
    void print_summary() const;
    void save_to_file(const std::string& filename) const;
    
private:
    struct ParamMetrics {
        std::vector<float> gradient_norms;
        std::vector<float> update_norms;
        std::vector<float> learning_rates;
    };
    
    std::unordered_map<std::string, ParamMetrics> param_metrics_;
    std::vector<float> accuracies_;
    std::vector<float> losses_;
};

}
}