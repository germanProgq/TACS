#include "training/adaptive_optimizer.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace tacs {
namespace training {

// GradientStatistics implementation
void GradientStatistics::update(const core::Tensor& gradient) {
    const float* grad_data = gradient.data_float();
    size_t size = gradient.size();
    
    // Compute gradient magnitude
    magnitude = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        magnitude += grad_data[i] * grad_data[i];
    }
    magnitude = std::sqrt(magnitude / size);
    
    // Update history
    history.push_back(magnitude);
    if (history.size() > HISTORY_SIZE) {
        history.pop_front();
    }
    
    // Compute statistics
    if (!history.empty()) {
        mean = std::accumulate(history.begin(), history.end(), 0.0f) / history.size();
        
        variance = 0.0f;
        for (float val : history) {
            float diff = val - mean;
            variance += diff * diff;
        }
        variance = (history.size() > 1) ? variance / (history.size() - 1) : 0.0f;
    }
}

float GradientStatistics::get_adaptive_scale() const {
    if (variance < 1e-8f) return 1.0f;
    
    // High variance indicates unstable gradients - reduce learning rate
    // Low variance indicates stable gradients - can increase learning rate
    float stability_factor = 1.0f / (1.0f + std::sqrt(variance));
    
    // Scale based on gradient magnitude trend
    float trend_factor = 1.0f;
    if (history.size() >= 10) {
        float recent_mean = 0.0f;
        float old_mean = 0.0f;
        size_t half_size = history.size() / 2;
        
        for (size_t i = 0; i < half_size; ++i) {
            old_mean += history[i];
            recent_mean += history[i + half_size];
        }
        old_mean /= half_size;
        recent_mean /= half_size;
        
        // If gradients are decreasing, we can increase learning rate
        if (old_mean > 1e-8f) {
            trend_factor = std::sqrt(old_mean / (recent_mean + 1e-8f));
            trend_factor = std::clamp(trend_factor, 0.5f, 2.0f);
        }
    }
    
    return stability_factor * trend_factor;
}

// AdaptiveOptimizer implementation
AdaptiveOptimizer::AdaptiveOptimizer(const AdaptiveConfig& config)
    : config_(config) {
    
    // Create base optimizer
    if (config.use_adam_base) {
        base_optimizer_ = std::make_unique<AdamOptimizer>(
            0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);
    } else {
        base_optimizer_ = std::make_unique<SGDOptimizer>(
            0.01f, 0.9f, 0.0f);
    }
}

void AdaptiveOptimizer::step() {
    // Update gradient statistics
    for (auto& [name, param_info] : parameters_) {
        if (param_info.gradient.size() > 0) {
            update_gradient_statistics(name, param_info.gradient);
        }
    }
    
    // Apply adaptive modifications to gradients
    for (auto& [name, param_info] : parameters_) {
        if (param_info.gradient.size() == 0) continue;
        
        // Apply gradient centralization
        if (config_.enable_gradient_centralization) {
            apply_gradient_centralization(param_info.gradient);
        }
        
        // Apply adaptive scaling
        float adaptive_scale = compute_adaptive_lr_scale(name);
        float* grad_data = param_info.gradient.data_float();
        size_t size = param_info.gradient.size();
        
        for (size_t i = 0; i < size; ++i) {
            grad_data[i] *= adaptive_scale;
        }
        
        // Gradient clipping based on statistics
        if (gradient_stats_.find(name) != gradient_stats_.end()) {
            const auto& stats = gradient_stats_[name];
            float clip_value = stats.mean + 3.0f * std::sqrt(stats.variance);
            clip_value = std::max(clip_value, config_.max_gradient_norm);
            clip_gradient_norm(param_info.gradient, clip_value);
        }
    }
    
    // Perform base optimizer step
    base_optimizer_->step();
    
    // Apply additional adaptive updates
    for (auto& [name, param_info] : parameters_) {
        update_parameter_adaptive(name, *param_info.parameter);
    }
    
    // Lookahead update
    if (lookahead_enabled_) {
        lookahead_step_++;
        if (lookahead_step_ % lookahead_k_ == 0) {
            for (auto& [name, param_info] : parameters_) {
                if (slow_weights_.find(name) != slow_weights_.end()) {
                    float* fast_data = param_info.parameter->data_float();
                    float* slow_data = slow_weights_[name].data_float();
                    size_t size = param_info.parameter->size();
                    
                    // Update slow weights: slow = slow + alpha * (fast - slow)
                    for (size_t i = 0; i < size; ++i) {
                        slow_data[i] += lookahead_alpha_ * (fast_data[i] - slow_data[i]);
                        fast_data[i] = slow_data[i];
                    }
                }
            }
        }
    }
}

void AdaptiveOptimizer::zero_grad() {
    base_optimizer_->zero_grad();
}

float AdaptiveOptimizer::get_learning_rate() const {
    return base_optimizer_->get_learning_rate() * global_lr_scale_;
}

void AdaptiveOptimizer::set_learning_rate(float lr) {
    base_optimizer_->set_learning_rate(lr);
}

void AdaptiveOptimizer::add_parameter_group(const std::string& name, core::Tensor& parameter) {
    // Store in our own parameters map
    ParameterInfo info;
    info.parameter = &parameter;  // Store pointer to the actual parameter
    info.gradient = core::Tensor(parameter.shape());
    info.gradient.zero();
    parameters_[name] = info;
    
    // Also add to base optimizer
    base_optimizer_->add_parameter_group(name, parameter);
    
    // Initialize gradient statistics
    gradient_stats_[name] = GradientStatistics();
    
    // Initialize layer adaptation state
    layer_states_[name] = LayerAdaptationState();
    
    // Initialize parameter momentum for adaptive updates
    param_momentum_[name] = core::Tensor(parameter.shape());
    param_momentum_[name].zero();
    
    // Initialize slow weights for lookahead
    if (lookahead_enabled_) {
        slow_weights_[name] = core::Tensor(parameter.shape());
        const float* param_data = parameter.data_float();
        float* slow_data = slow_weights_[name].data_float();
        size_t size = parameter.size();
        
        for (size_t i = 0; i < size; ++i) {
            slow_data[i] = param_data[i];
        }
    }
}

void AdaptiveOptimizer::set_gradient(const std::string& name, const core::Tensor& gradient) {
    // Update our gradient
    if (parameters_.find(name) != parameters_.end()) {
        const float* grad_data = gradient.data_float();
        float* stored_grad = parameters_[name].gradient.data_float();
        size_t size = gradient.size();
        
        for (size_t i = 0; i < size; ++i) {
            stored_grad[i] = grad_data[i];
        }
    }
    
    // Also set in base optimizer
    base_optimizer_->set_gradient(name, gradient);
}

void AdaptiveOptimizer::update_accuracy_feedback(float current_accuracy, float target_accuracy) {
    accuracy_history_.push_back(current_accuracy);
    if (accuracy_history_.size() > 100) {
        accuracy_history_.pop_front();
    }
    
    // Compute accuracy trend
    float accuracy_trend = 0.0f;
    if (accuracy_history_.size() >= 10) {
        float recent_avg = 0.0f;
        float old_avg = 0.0f;
        size_t half_size = accuracy_history_.size() / 2;
        
        for (size_t i = 0; i < half_size; ++i) {
            old_avg += accuracy_history_[i];
            recent_avg += accuracy_history_[i + half_size];
        }
        old_avg /= half_size;
        recent_avg /= half_size;
        
        accuracy_trend = recent_avg - old_avg;
    }
    
    // Adjust global learning rate scale based on accuracy
    float accuracy_gap = target_accuracy - current_accuracy;
    
    if (accuracy_gap > 0.1f) {
        // Far from target - increase learning rate
        global_lr_scale_ *= 1.05f;
    } else if (accuracy_gap > 0.01f) {
        // Getting closer - maintain or slightly increase
        global_lr_scale_ *= 1.01f;
    } else if (accuracy_gap > 0.001f) {
        // Very close - reduce learning rate for fine-tuning
        global_lr_scale_ *= 0.98f;
    } else {
        // Achieved target - maintain with small adjustments
        global_lr_scale_ *= 0.99f;
    }
    
    // Consider accuracy trend
    if (accuracy_trend < -0.01f) {
        // Accuracy decreasing - reduce learning rate
        global_lr_scale_ *= 0.9f;
    }
    
    // Apply bounds
    global_lr_scale_ = std::clamp(global_lr_scale_, 
                                  config_.min_lr_scale, 
                                  config_.max_lr_scale);
    
    // Update accuracy feedback scale
    accuracy_feedback_scale_ = config_.accuracy_momentum * accuracy_feedback_scale_ + 
                              (1.0f - config_.accuracy_momentum) * (current_accuracy / target_accuracy);
}

void AdaptiveOptimizer::update_loss_feedback(float current_loss, 
                                            const std::unordered_map<std::string, float>& layer_contributions) {
    loss_history_.push_back(current_loss);
    if (loss_history_.size() > 100) {
        loss_history_.pop_front();
    }
    
    // Update layer-specific adaptation based on loss contributions
    for (const auto& [layer_name, contribution] : layer_contributions) {
        if (layer_states_.find(layer_name) != layer_states_.end()) {
            auto& state = layer_states_[layer_name];
            
            // Check if layer is contributing to loss reduction
            float contribution_change = contribution - state.last_loss_contribution;
            state.last_loss_contribution = contribution;
            
            if (std::abs(contribution_change) < 1e-6f) {
                state.stagnation_counter++;
                if (state.stagnation_counter > 10) {
                    // Layer is stagnating - increase its learning rate
                    state.learning_rate_scale *= 1.1f;
                    state.stagnation_counter = 0;
                }
            } else {
                state.stagnation_counter = 0;
                
                // Adjust learning rate based on contribution
                if (contribution > 0.5f) {
                    // High contribution to loss - increase learning rate
                    state.learning_rate_scale *= 1.05f;
                } else if (contribution < 0.1f) {
                    // Low contribution - maybe already optimized
                    state.learning_rate_scale *= 0.95f;
                }
            }
            
            // Apply bounds
            state.learning_rate_scale = std::clamp(state.learning_rate_scale, 0.1f, 10.0f);
        }
    }
}

void AdaptiveOptimizer::update_gradient_statistics(const std::string& name, const core::Tensor& gradient) {
    if (gradient_stats_.find(name) == gradient_stats_.end()) {
        gradient_stats_[name] = GradientStatistics();
    }
    
    gradient_stats_[name].update(gradient);
    
    // Update layer state with gradient norm
    if (layer_states_.find(name) != layer_states_.end()) {
        layer_states_[name].gradient_norm = gradient_stats_[name].magnitude;
    }
}

float AdaptiveOptimizer::compute_adaptive_lr_scale(const std::string& name) const {
    float scale = global_lr_scale_ * accuracy_feedback_scale_;
    
    // Apply gradient-based scaling
    if (gradient_stats_.find(name) != gradient_stats_.end()) {
        scale *= gradient_stats_.at(name).get_adaptive_scale();
    }
    
    // Apply layer-wise scaling
    if (config_.enable_layer_wise_lr && layer_states_.find(name) != layer_states_.end()) {
        scale *= layer_states_.at(name).learning_rate_scale;
        
        // Apply depth-based decay (assuming layer names contain depth info)
        if (name.find("layer") != std::string::npos) {
            size_t layer_num = 0;
            try {
                size_t pos = name.find("layer") + 5;
                layer_num = std::stoul(name.substr(pos, 1));
                scale *= std::pow(config_.layer_lr_decay, layer_num);
            } catch (...) {
                // Ignore parsing errors
            }
        }
    }
    
    return scale;
}

void AdaptiveOptimizer::apply_gradient_centralization(core::Tensor& gradient) {
    if (gradient.shape().size() < 2) return;  // Only for weight matrices
    
    float* grad_data = gradient.data_float();
    const auto& shape = gradient.shape();
    
    // Compute mean for each output channel
    int out_channels = shape[0];
    int elements_per_channel = gradient.size() / out_channels;
    
    for (int oc = 0; oc < out_channels; ++oc) {
        float mean = 0.0f;
        int offset = oc * elements_per_channel;
        
        for (int i = 0; i < elements_per_channel; ++i) {
            mean += grad_data[offset + i];
        }
        mean /= elements_per_channel;
        
        // Center gradients
        for (int i = 0; i < elements_per_channel; ++i) {
            grad_data[offset + i] -= mean;
        }
    }
}

void AdaptiveOptimizer::apply_weight_standardization(core::Tensor& weights) {
    if (weights.shape().size() < 2) return;  // Only for weight matrices
    
    float* weight_data = weights.data_float();
    const auto& shape = weights.shape();
    
    int out_channels = shape[0];
    int elements_per_channel = weights.size() / out_channels;
    
    for (int oc = 0; oc < out_channels; ++oc) {
        int offset = oc * elements_per_channel;
        
        // Compute mean and variance
        float mean = 0.0f;
        float variance = 0.0f;
        
        for (int i = 0; i < elements_per_channel; ++i) {
            mean += weight_data[offset + i];
        }
        mean /= elements_per_channel;
        
        for (int i = 0; i < elements_per_channel; ++i) {
            float diff = weight_data[offset + i] - mean;
            variance += diff * diff;
        }
        variance = std::sqrt(variance / elements_per_channel + 1e-8f);
        
        // Standardize weights
        for (int i = 0; i < elements_per_channel; ++i) {
            weight_data[offset + i] = (weight_data[offset + i] - mean) / variance;
        }
    }
}

void AdaptiveOptimizer::update_parameter_adaptive(const std::string& name, core::Tensor& parameter) {
    // Apply weight standardization
    if (config_.enable_weight_standardization) {
        apply_weight_standardization(parameter);
    }
    
    // Apply adaptive weight noise for exploration
    if (config_.weight_noise_scale > 0.0f && accuracy_feedback_scale_ < 0.95f) {
        // Add noise when not close to target accuracy
        float noise_scale = config_.weight_noise_scale * (1.0f - accuracy_feedback_scale_);
        apply_weight_noise(parameter, noise_scale);
    }
    
    // Update weight norm in layer state
    if (layer_states_.find(name) != layer_states_.end()) {
        float* param_data = parameter.data_float();
        size_t size = parameter.size();
        
        float norm = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            norm += param_data[i] * param_data[i];
        }
        layer_states_[name].weight_norm = std::sqrt(norm / size);
    }
}

void AdaptiveOptimizer::apply_weight_noise(core::Tensor& weights, float scale) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    
    float* weight_data = weights.data_float();
    size_t size = weights.size();
    
    for (size_t i = 0; i < size; ++i) {
        weight_data[i] += dist(gen) * std::abs(weight_data[i]);
    }
}

float AdaptiveOptimizer::compute_gradient_norm(const core::Tensor& gradient) const {
    const float* grad_data = gradient.data_float();
    size_t size = gradient.size();
    
    float norm = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        norm += grad_data[i] * grad_data[i];
    }
    
    return std::sqrt(norm);
}

void AdaptiveOptimizer::clip_gradient_norm(core::Tensor& gradient, float max_norm) {
    float grad_norm = compute_gradient_norm(gradient);
    
    if (grad_norm > max_norm) {
        float scale = max_norm / grad_norm;
        float* grad_data = gradient.data_float();
        size_t size = gradient.size();
        
        for (size_t i = 0; i < size; ++i) {
            grad_data[i] *= scale;
        }
    }
}

void AdaptiveOptimizer::enable_lookahead(int k, float alpha) {
    lookahead_enabled_ = true;
    lookahead_k_ = k;
    lookahead_alpha_ = alpha;
    lookahead_step_ = 0;
    
    // Initialize slow weights for existing parameters
    for (const auto& [name, param_info] : parameters_) {
        slow_weights_[name] = core::Tensor(param_info.parameter->shape());
        const float* param_data = param_info.parameter->data_float();
        float* slow_data = slow_weights_[name].data_float();
        size_t size = param_info.parameter->size();
        
        for (size_t i = 0; i < size; ++i) {
            slow_data[i] = param_data[i];
        }
    }
}

void AdaptiveOptimizer::enable_gradient_centralization(bool enable) {
    config_.enable_gradient_centralization = enable;
}

void AdaptiveOptimizer::enable_weight_standardization(bool enable) {
    config_.enable_weight_standardization = enable;
}

void AdaptiveOptimizer::reset_adaptation() {
    gradient_stats_.clear();
    layer_states_.clear();
    param_momentum_.clear();
    accuracy_history_.clear();
    loss_history_.clear();
    global_lr_scale_ = 1.0f;
    accuracy_feedback_scale_ = 1.0f;
    loss_feedback_scale_ = 1.0f;
    lookahead_step_ = 0;
}

std::unordered_map<std::string, GradientStatistics> AdaptiveOptimizer::get_gradient_statistics() const {
    return gradient_stats_;
}

float AdaptiveOptimizer::get_effective_learning_rate(const std::string& param_name) const {
    return get_learning_rate() * compute_adaptive_lr_scale(param_name);
}

void AdaptiveOptimizer::set_layer_adaptation(const std::string& layer_name, const LayerAdaptationState& state) {
    layer_states_[layer_name] = state;
}

LayerAdaptationState AdaptiveOptimizer::get_layer_adaptation(const std::string& layer_name) const {
    if (layer_states_.find(layer_name) != layer_states_.end()) {
        return layer_states_.at(layer_name);
    }
    return LayerAdaptationState();
}

// OptimizationMonitor implementation
void OptimizationMonitor::record_gradient(const std::string& param_name, float grad_norm) {
    param_metrics_[param_name].gradient_norms.push_back(grad_norm);
}

void OptimizationMonitor::record_weight_update(const std::string& param_name, float update_norm) {
    param_metrics_[param_name].update_norms.push_back(update_norm);
}

void OptimizationMonitor::record_learning_rate(const std::string& param_name, float lr) {
    param_metrics_[param_name].learning_rates.push_back(lr);
}

void OptimizationMonitor::record_accuracy(float accuracy) {
    accuracies_.push_back(accuracy);
}

void OptimizationMonitor::record_loss(float loss) {
    losses_.push_back(loss);
}

void OptimizationMonitor::print_summary() const {
    std::cout << "\n=== Optimization Summary ===\n";
    
    if (!losses_.empty()) {
        float avg_loss = std::accumulate(losses_.begin(), losses_.end(), 0.0f) / losses_.size();
        std::cout << "Average Loss: " << avg_loss << "\n";
    }
    
    if (!accuracies_.empty()) {
        float best_accuracy = *std::max_element(accuracies_.begin(), accuracies_.end());
        float final_accuracy = accuracies_.back();
        std::cout << "Best Accuracy: " << best_accuracy << "\n";
        std::cout << "Final Accuracy: " << final_accuracy << "\n";
    }
    
    std::cout << "\nPer-Parameter Statistics:\n";
    for (const auto& [name, metrics] : param_metrics_) {
        if (!metrics.gradient_norms.empty()) {
            float avg_grad = std::accumulate(metrics.gradient_norms.begin(), 
                                           metrics.gradient_norms.end(), 0.0f) / metrics.gradient_norms.size();
            std::cout << name << " - Avg Gradient Norm: " << avg_grad << "\n";
        }
    }
}

void OptimizationMonitor::save_to_file(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "epoch,loss,accuracy\n";
    size_t max_len = std::max(losses_.size(), accuracies_.size());
    
    for (size_t i = 0; i < max_len; ++i) {
        file << i << ",";
        file << (i < losses_.size() ? std::to_string(losses_[i]) : "") << ",";
        file << (i < accuracies_.size() ? std::to_string(accuracies_[i]) : "") << "\n";
    }
    
    file.close();
}

}
}