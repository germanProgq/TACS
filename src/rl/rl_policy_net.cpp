/*
 * A2C implementation with SIMD optimization for real-time traffic control
 */

#include "../../include/rl/rl_policy_net.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>
#include <iostream>
#include <fstream>

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

namespace tacs {

using core::Tensor;
using core::MemoryManager;

// RLState implementation
std::vector<float> RLState::toFeatureVector() const {
    std::vector<float> features;
    features.reserve(getFeatureSize());
    
    // Queue lengths (normalized by max expected queue length)
    for (float q : queue_lengths) {
        features.push_back(q / 50.0f);  // Normalize by max 50 vehicles
    }
    
    // Pedestrian counts (normalized)
    for (float p : pedestrian_counts) {
        features.push_back(p / 20.0f);  // Max 20 pedestrians
    }
    
    // Cyclist counts (normalized)
    for (float c : cyclist_counts) {
        features.push_back(c / 10.0f);  // Max 10 cyclists
    }
    
    // One-hot encode weather condition
    for (int i = 0; i < 4; ++i) {
        features.push_back(i == static_cast<int>(weather_condition) ? 1.0f : 0.0f);
    }
    
    // Accident indicator
    features.push_back(accident_indicator);
    
    // Current phase duration (normalized by max phase time)
    features.push_back(current_phase_duration / 120.0f);  // Max 2 minutes
    
    // Time of day (already normalized 0-1)
    features.push_back(time_of_day);
    
    // Add cyclic time encoding for better time representation
    features.push_back(std::sin(2.0f * M_PI * time_of_day));
    features.push_back(std::cos(2.0f * M_PI * time_of_day));
    
    return features;
}

size_t RLState::getFeatureSize() const {
    return queue_lengths.size() + pedestrian_counts.size() + 
           cyclist_counts.size() + 4 + 1 + 1 + 1 + 2;  // +2 for cyclic time
}

// RLPolicyNet implementation
RLPolicyNet::RLPolicyNet(const Config& config) 
    : config_(config),
      memory_manager_(&MemoryManager::instance()) {
    
    // Allocate replay buffer
    replay_buffer_.buffer.reserve(config_.replay_buffer_size);
    replay_buffer_.priorities.reserve(config_.replay_buffer_size);
    
    // Initialize networks
    initializeWeights();
    
    // Reset stats
    stats_ = PerformanceStats{};
}

RLPolicyNet::~RLPolicyNet() = default;

void RLPolicyNet::initializeWeights() {
    // Xavier initialization for better gradient flow
    float actor_scale1 = std::sqrt(2.0f / config_.state_dim);
    float actor_scale2 = std::sqrt(2.0f / config_.hidden_dim);
    float critic_scale1 = std::sqrt(2.0f / config_.state_dim);
    float critic_scale2 = std::sqrt(2.0f / config_.hidden_dim);
    
    // Actor network initialization
    actor_.W1 = Tensor({static_cast<int>(config_.hidden_dim), static_cast<int>(config_.state_dim)});
    actor_.b1 = Tensor({static_cast<int>(config_.hidden_dim)});
    actor_.W2 = Tensor({static_cast<int>(config_.num_actions), static_cast<int>(config_.hidden_dim)});
    actor_.b2 = Tensor({static_cast<int>(config_.num_actions)});
    
    // Initialize with Xavier/He initialization
    std::mt19937 rng(42);
    std::normal_distribution<float> actor_dist1(0.0f, actor_scale1);
    std::normal_distribution<float> actor_dist2(0.0f, actor_scale2);
    
    for (size_t i = 0; i < actor_.W1.size(); ++i) {
        actor_.W1.data_float()[i] = actor_dist1(rng);
    }
    for (size_t i = 0; i < actor_.W2.size(); ++i) {
        actor_.W2.data_float()[i] = actor_dist2(rng);
    }
    
    // Biases start at zero
    std::memset(actor_.b1.data_float(), 0, actor_.b1.size() * sizeof(float));
    std::memset(actor_.b2.data_float(), 0, actor_.b2.size() * sizeof(float));
    
    // Critic network initialization
    critic_.W1 = Tensor({static_cast<int>(config_.hidden_dim), static_cast<int>(config_.state_dim)});
    critic_.b1 = Tensor({static_cast<int>(config_.hidden_dim)});
    critic_.W2 = Tensor({1, static_cast<int>(config_.hidden_dim)});  // Single value output
    critic_.b2 = Tensor({1});
    
    std::normal_distribution<float> critic_dist1(0.0f, critic_scale1);
    std::normal_distribution<float> critic_dist2(0.0f, critic_scale2);
    
    for (size_t i = 0; i < critic_.W1.size(); ++i) {
        critic_.W1.data_float()[i] = critic_dist1(rng);
    }
    for (size_t i = 0; i < critic_.W2.size(); ++i) {
        critic_.W2.data_float()[i] = critic_dist2(rng);
    }
    
    std::memset(critic_.b1.data_float(), 0, critic_.b1.size() * sizeof(float));
    std::memset(critic_.b2.data_float(), 0, critic_.b2.size() * sizeof(float));
}

SignalPhase RLPolicyNet::selectAction(const RLState& state, bool training) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Check for emergency override first
    if (checkEmergencyOverride(state)) {
        return getEmergencyPhase(state);
    }
    
    // Check peer consensus if enabled
    if (config_.enable_peer_voting && !peer_votes_.empty()) {
        SignalPhase consensus = computeConsensusPhase(state);
        if (consensus != SignalPhase::ALL_RED) {  // Valid consensus found
            return consensus;
        }
    }
    
    // Convert state to tensor
    auto features = state.toFeatureVector();
    Tensor state_tensor({1, static_cast<int>(features.size())});
    std::memcpy(state_tensor.data_float(), features.data(), 
                features.size() * sizeof(float));
    
    // Forward pass through actor network
    Tensor logits = actorForward(state_tensor);
    
    // Apply softmax to get probabilities
    float* logit_data = logits.data_float();
    float max_logit = *std::max_element(logit_data, 
                                       logit_data + config_.num_actions);
    
    // Numerically stable softmax
    float sum_exp = 0.0f;
    std::vector<float> probs(config_.num_actions);
    for (size_t i = 0; i < config_.num_actions; ++i) {
        probs[i] = std::exp(logit_data[i] - max_logit);
        sum_exp += probs[i];
    }
    
    for (size_t i = 0; i < config_.num_actions; ++i) {
        probs[i] /= sum_exp;
    }
    
    // Select action
    int action_idx;
    if (training) {
        // Sample from probability distribution
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        action_idx = dist(gen);
    } else {
        // Choose action with highest probability (greedy)
        action_idx = std::distance(probs.begin(), 
                                  std::max_element(probs.begin(), probs.end()));
    }
    
    // Update performance stats
    auto end_time = std::chrono::high_resolution_clock::now();
    float decision_time_ms = std::chrono::duration<float, std::milli>(
        end_time - start_time).count();
    
    stats_.decisions_made++;
    stats_.avg_decision_time_ms = (stats_.avg_decision_time_ms * 
        (stats_.decisions_made - 1) + decision_time_ms) / stats_.decisions_made;
    stats_.max_decision_time_ms = std::max(stats_.max_decision_time_ms, 
                                          decision_time_ms);
    
    return static_cast<SignalPhase>(action_idx);
}

std::pair<Tensor, Tensor> RLPolicyNet::forward(const Tensor& state_batch) {
    Tensor policy_logits = actorForward(state_batch);
    Tensor values = criticForward(state_batch);
    return {policy_logits, values};
}

Tensor RLPolicyNet::actorForward(const Tensor& states) {
    size_t batch_size = states.shape()[0];
    
    // Hidden layer: h1 = ReLU(W1 @ states + b1)
    Tensor h1({static_cast<int>(batch_size), static_cast<int>(config_.hidden_dim)});
    
    // Optimized matrix multiplication
    optimizedMatMul(actor_.W1.data_float(), states.data_float(), h1.data_float(),
                   config_.hidden_dim, batch_size, config_.state_dim);
    
    // Add bias and apply ReLU
    float* h1_data = h1.data_float();
    const float* b1_data = actor_.b1.data_float();
    
#ifdef __ARM_NEON
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < config_.hidden_dim; j += 4) {
            float32x4_t h = vld1q_f32(h1_data + i * config_.hidden_dim + j);
            float32x4_t b = vld1q_f32(b1_data + j);
            h = vaddq_f32(h, b);
            h = vmaxq_f32(h, vdupq_n_f32(0.0f));  // ReLU
            vst1q_f32(h1_data + i * config_.hidden_dim + j, h);
        }
    }
#else
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            size_t idx = i * config_.hidden_dim + j;
            h1_data[idx] = std::max(0.0f, h1_data[idx] + b1_data[j]);
        }
    }
#endif
    
    // Output layer: logits = W2 @ h1 + b2
    Tensor logits({static_cast<int>(batch_size), static_cast<int>(config_.num_actions)});
    
    optimizedMatMul(actor_.W2.data_float(), h1.data_float(), logits.data_float(),
                   config_.num_actions, batch_size, config_.hidden_dim);
    
    // Add output bias
    float* logits_data = logits.data_float();
    const float* b2_data = actor_.b2.data_float();
    
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < config_.num_actions; ++j) {
            logits_data[i * config_.num_actions + j] += b2_data[j];
        }
    }
    
    return logits;
}

Tensor RLPolicyNet::criticForward(const Tensor& states) {
    size_t batch_size = states.shape()[0];
    
    // Hidden layer
    Tensor h1({static_cast<int>(batch_size), static_cast<int>(config_.hidden_dim)});
    
    optimizedMatMul(critic_.W1.data_float(), states.data_float(), h1.data_float(),
                   config_.hidden_dim, batch_size, config_.state_dim);
    
    // Add bias and ReLU
    float* h1_data = h1.data_float();
    const float* b1_data = critic_.b1.data_float();
    
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            size_t idx = i * config_.hidden_dim + j;
            h1_data[idx] = std::max(0.0f, h1_data[idx] + b1_data[j]);
        }
    }
    
    // Output layer (single value)
    Tensor values({static_cast<int>(batch_size), 1});
    
    optimizedMatMul(critic_.W2.data_float(), h1.data_float(), values.data_float(),
                   1, batch_size, config_.hidden_dim);
    
    // Add output bias
    float* values_data = values.data_float();
    for (size_t i = 0; i < batch_size; ++i) {
        values_data[i] += critic_.b2.data_float()[0];
    }
    
    return values;
}

void RLPolicyNet::optimizedMatMul(const float* A, const float* B, float* C,
                                 size_t M, size_t N, size_t K) {
    // Initialize C to zero
    std::memset(C, 0, M * N * sizeof(float));
    
    // Optimized matrix multiplication with cache blocking
    const size_t BLOCK_SIZE = 64;
    
    for (size_t i = 0; i < M; i += BLOCK_SIZE) {
        for (size_t j = 0; j < N; j += BLOCK_SIZE) {
            for (size_t k = 0; k < K; k += BLOCK_SIZE) {
                // Process block
                size_t i_end = std::min(i + BLOCK_SIZE, M);
                size_t j_end = std::min(j + BLOCK_SIZE, N);
                size_t k_end = std::min(k + BLOCK_SIZE, K);
                
                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t kk = k; kk < k_end; ++kk) {
                        float a_val = A[ii * K + kk];
                        
#ifdef __ARM_NEON
                        // NEON vectorized inner loop
                        size_t jj = j;
                        for (; jj + 4 <= j_end; jj += 4) {
                            float32x4_t b_vec = vld1q_f32(&B[kk * N + jj]);
                            float32x4_t c_vec = vld1q_f32(&C[ii * N + jj]);
                            c_vec = vmlaq_n_f32(c_vec, b_vec, a_val);
                            vst1q_f32(&C[ii * N + jj], c_vec);
                        }
                        // Handle remainder
                        for (; jj < j_end; ++jj) {
                            C[ii * N + jj] += a_val * B[kk * N + jj];
                        }
#elif defined(__SSE2__)
                        // SSE vectorized inner loop
                        size_t jj = j;
                        __m128 a_vec = _mm_set1_ps(a_val);
                        for (; jj + 4 <= j_end; jj += 4) {
                            __m128 b_vec = _mm_loadu_ps(&B[kk * N + jj]);
                            __m128 c_vec = _mm_loadu_ps(&C[ii * N + jj]);
                            c_vec = _mm_add_ps(c_vec, _mm_mul_ps(a_vec, b_vec));
                            _mm_storeu_ps(&C[ii * N + jj], c_vec);
                        }
                        // Handle remainder
                        for (; jj < j_end; ++jj) {
                            C[ii * N + jj] += a_val * B[kk * N + jj];
                        }
#else
                        // Scalar fallback
                        for (size_t jj = j; jj < j_end; ++jj) {
                            C[ii * N + jj] += a_val * B[kk * N + jj];
                        }
#endif
                    }
                }
            }
        }
    }
}

float RLPolicyNet::computeAdvantage(const Experience& exp) {
    // Convert states to tensors
    auto state_features = exp.state.toFeatureVector();
    auto next_state_features = exp.next_state.toFeatureVector();
    
    Tensor state_tensor({1, static_cast<int>(state_features.size())});
    Tensor next_state_tensor({1, static_cast<int>(next_state_features.size())});
    
    std::memcpy(state_tensor.data_float(), state_features.data(), 
                state_features.size() * sizeof(float));
    std::memcpy(next_state_tensor.data_float(), next_state_features.data(),
                next_state_features.size() * sizeof(float));
    
    // Get value estimates
    Tensor value = criticForward(state_tensor);
    Tensor next_value = criticForward(next_state_tensor);
    
    // Compute TD error (advantage)
    float v = value.data_float()[0];
    float v_next = exp.done ? 0.0f : next_value.data_float()[0];
    float advantage = exp.reward + config_.gamma * v_next - v;
    
    return advantage;
}

RLPolicyNet::A2CLoss RLPolicyNet::computeLoss(const std::vector<Experience>& batch) {
    size_t batch_size = batch.size();
    
    // Prepare batch tensors
    size_t state_dim = batch[0].state.getFeatureSize();
    Tensor states({static_cast<int>(batch_size), static_cast<int>(state_dim)});
    Tensor actions({static_cast<int>(batch_size), 1});
    Tensor rewards({static_cast<int>(batch_size), 1});
    Tensor advantages({static_cast<int>(batch_size), 1});
    Tensor returns({static_cast<int>(batch_size), 1});
    
    // Fill batch data
    for (size_t i = 0; i < batch_size; ++i) {
        auto features = batch[i].state.toFeatureVector();
        std::memcpy(states.data_float() + i * state_dim, features.data(),
                   state_dim * sizeof(float));
        
        actions.data_float()[i] = static_cast<float>(batch[i].action);
        rewards.data_float()[i] = batch[i].reward;
        
        // Compute advantage and return
        float advantage = computeAdvantage(batch[i]);
        advantages.data_float()[i] = advantage;
        
        // Compute return (for value function training)
        Tensor single_state({1, static_cast<int>(state_dim)});
        std::memcpy(single_state.data_float(), states.data_float() + i * state_dim, state_dim * sizeof(float));
        float value = criticForward(single_state).data_float()[0];
        returns.data_float()[i] = value + advantage;
    }
    
    // Forward pass
    auto [logits, values] = forward(states);
    
    // Compute policy loss
    float policy_loss = 0.0f;
    float entropy = 0.0f;
    
    for (size_t i = 0; i < batch_size; ++i) {
        // Get action probabilities for this sample
        const float* sample_logits = logits.data_float() + i * config_.num_actions;
        
        // Compute log softmax
        float max_logit = *std::max_element(sample_logits, 
                                          sample_logits + config_.num_actions);
        float sum_exp = 0.0f;
        std::vector<float> log_probs(config_.num_actions);
        
        for (size_t j = 0; j < config_.num_actions; ++j) {
            sum_exp += std::exp(sample_logits[j] - max_logit);
        }
        
        for (size_t j = 0; j < config_.num_actions; ++j) {
            log_probs[j] = sample_logits[j] - max_logit - std::log(sum_exp);
        }
        
        // Policy gradient loss
        int action = static_cast<int>(actions.data_float()[i]);
        policy_loss -= log_probs[action] * advantages.data_float()[i];
        
        // Entropy for exploration
        for (size_t j = 0; j < config_.num_actions; ++j) {
            float prob = std::exp(log_probs[j]);
            if (prob > 1e-8f) {
                entropy -= prob * log_probs[j];
            }
        }
    }
    
    policy_loss /= batch_size;
    entropy /= batch_size;
    
    // Compute value loss (MSE)
    float value_loss = 0.0f;
    for (size_t i = 0; i < batch_size; ++i) {
        float diff = values.data_float()[i] - returns.data_float()[i];
        value_loss += diff * diff;
    }
    value_loss = 0.5f * value_loss / batch_size;
    
    // Total loss
    float total_loss = policy_loss + 
                      config_.value_loss_coef * value_loss - 
                      config_.entropy_coef * entropy;
    
    // Add EWC penalty if enabled
    if (ewc_.enabled && !ewc_.reference_weights_.empty() && !ewc_.fisher_information_.empty()) {
        float ewc_loss = 0.0f;
        
        // Actor EWC penalty
        auto computeEWCPenalty = [](const Tensor& current, const Tensor& reference,
                                   const Tensor& fisher, float lambda) {
            float penalty = 0.0f;
            if (current.size() == reference.size() && current.size() == fisher.size()) {
                for (size_t i = 0; i < current.size(); ++i) {
                    float diff = current.data_float()[i] - reference.data_float()[i];
                    penalty += fisher.data_float()[i] * diff * diff;
                }
            }
            return 0.5f * lambda * penalty;
        };
        
        if (ewc_.reference_weights_.count("actor_W1") && ewc_.fisher_information_.count("actor_W1")) {
            ewc_loss += computeEWCPenalty(actor_.W1, ewc_.reference_weights_["actor_W1"],
                                          ewc_.fisher_information_["actor_W1"], ewc_.lambda);
        }
        if (ewc_.reference_weights_.count("actor_W2") && ewc_.fisher_information_.count("actor_W2")) {
            ewc_loss += computeEWCPenalty(actor_.W2, ewc_.reference_weights_["actor_W2"],
                                          ewc_.fisher_information_["actor_W2"], ewc_.lambda);
        }
        
        total_loss += ewc_loss;
    }
    
    return A2CLoss{policy_loss, value_loss, -entropy, total_loss};
}

void RLPolicyNet::train(const std::vector<Experience>& batch) {
    // Compute loss
    auto loss = computeLoss(batch);
    
    // Prepare batch tensors
    size_t batch_size = batch.size();
    size_t state_dim = batch[0].state.getFeatureSize();
    
    Tensor states({static_cast<int>(batch_size), static_cast<int>(state_dim)});
    Tensor actions({static_cast<int>(batch_size), 1});
    Tensor advantages({static_cast<int>(batch_size), 1});
    Tensor returns({static_cast<int>(batch_size), 1});
    
    // Fill batch data
    for (size_t i = 0; i < batch_size; ++i) {
        auto features = batch[i].state.toFeatureVector();
        std::memcpy(states.data_float() + i * state_dim, features.data(),
                   state_dim * sizeof(float));
        
        actions.data_float()[i] = static_cast<float>(batch[i].action);
        
        // Compute advantage and return
        float advantage = computeAdvantage(batch[i]);
        advantages.data_float()[i] = advantage;
        
        // Compute return for value function training
        Tensor single_state({1, static_cast<int>(state_dim)});
        std::memcpy(single_state.data_float(), states.data_float() + i * state_dim, 
                   state_dim * sizeof(float));
        float value = criticForward(single_state).data_float()[0];
        returns.data_float()[i] = value + advantage;
    }
    
    // Normalize advantages
    float adv_mean = 0.0f, adv_std = 0.0f;
    for (size_t i = 0; i < batch_size; ++i) {
        adv_mean += advantages.data_float()[i];
    }
    adv_mean /= batch_size;
    
    for (size_t i = 0; i < batch_size; ++i) {
        float diff = advantages.data_float()[i] - adv_mean;
        adv_std += diff * diff;
    }
    adv_std = std::sqrt(adv_std / batch_size + 1e-8f);
    
    for (size_t i = 0; i < batch_size; ++i) {
        advantages.data_float()[i] = (advantages.data_float()[i] - adv_mean) / adv_std;
    }
    
    // Update actor and critic networks
    updateActor(states, actions, advantages);
    updateCritic(states, returns);
    
    // Update performance stats
    float avg_reward = 0.0f;
    for (const auto& exp : batch) {
        avg_reward += exp.reward;
    }
    avg_reward /= batch.size();
    
    stats_.avg_reward = (stats_.avg_reward * 0.99f + avg_reward * 0.01f);
}

void RLPolicyNet::addExperience(const Experience& exp) {
    // Compute initial priority based on TD error
    float td_error = std::abs(computeAdvantage(exp));
    float priority = std::pow(td_error + 1e-6f, config_.priority_alpha);
    
    if (replay_buffer_.size < config_.replay_buffer_size) {
        replay_buffer_.buffer.push_back(exp);
        replay_buffer_.priorities.push_back(priority);
        replay_buffer_.size++;
    } else {
        // Overwrite oldest experience
        replay_buffer_.buffer[replay_buffer_.position] = exp;
        replay_buffer_.priorities[replay_buffer_.position] = priority;
    }
    
    replay_buffer_.position = (replay_buffer_.position + 1) % 
                             config_.replay_buffer_size;
}

std::vector<Experience> RLPolicyNet::sampleBatch() {
    if (replay_buffer_.size < config_.batch_size) {
        return {};
    }
    
    std::vector<Experience> batch;
    batch.reserve(config_.batch_size);
    
    // Compute sampling probabilities
    float priority_sum = std::accumulate(replay_buffer_.priorities.begin(),
                                       replay_buffer_.priorities.begin() + replay_buffer_.size,
                                       0.0f);
    
    std::vector<float> probs(replay_buffer_.size);
    for (size_t i = 0; i < replay_buffer_.size; ++i) {
        probs[i] = replay_buffer_.priorities[i] / priority_sum;
    }
    
    // Sample with replacement
    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    
    for (size_t i = 0; i < config_.batch_size; ++i) {
        size_t idx = dist(replay_buffer_.rng);
        batch.push_back(replay_buffer_.buffer[idx]);
    }
    
    return batch;
}

void RLPolicyNet::updatePriorities(const std::vector<size_t>& indices, 
                                  const std::vector<float>& td_errors) {
    if (indices.size() != td_errors.size()) {
        throw std::invalid_argument("Indices and TD errors must have same size");
    }
    
    // Update priorities based on TD errors
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < replay_buffer_.size) {
            // Priority = |TD-error| + epsilon, raised to alpha power
            float priority = std::pow(std::abs(td_errors[i]) + 1e-6f, config_.priority_alpha);
            replay_buffer_.priorities[indices[i]] = priority;
        }
    }
}

float RLPolicyNet::computeEntropy(const Tensor& logits) {
    float entropy = 0.0f;
    size_t batch_size = logits.shape()[0];
    
    for (size_t i = 0; i < batch_size; ++i) {
        const float* sample_logits = logits.data_float() + i * config_.num_actions;
        
        // Compute softmax probabilities
        float max_logit = *std::max_element(sample_logits, 
                                          sample_logits + config_.num_actions);
        
        float sum_exp = 0.0f;
        std::vector<float> probs(config_.num_actions);
        for (size_t j = 0; j < config_.num_actions; ++j) {
            probs[j] = std::exp(sample_logits[j] - max_logit);
            sum_exp += probs[j];
        }
        
        for (size_t j = 0; j < config_.num_actions; ++j) {
            probs[j] /= sum_exp;
            if (probs[j] > 1e-8f) {
                entropy -= probs[j] * std::log(probs[j]);
            }
        }
    }
    
    return entropy / batch_size;
}

bool RLPolicyNet::checkEmergencyOverride(const RLState& state) {
    // High accident severity triggers emergency
    if (state.accident_indicator > 0.8f) {
        emergency_active_ = true;
        return true;
    }
    
    // Emergency vehicle detection from TACSNet integration
    // Queue pattern analysis for emergency route detection
    
    // Unusual queue pattern might indicate emergency vehicle
    float max_queue = *std::max_element(state.queue_lengths.begin(),
                                       state.queue_lengths.end());
    float min_queue = *std::min_element(state.queue_lengths.begin(),
                                       state.queue_lengths.end());
    
    if (max_queue > 40.0f && min_queue < 5.0f) {
        // Significant imbalance might indicate blocked emergency route
        return true;
    }
    
    emergency_active_ = false;
    return false;
}

SignalPhase RLPolicyNet::getEmergencyPhase(const RLState& state) {
    // Determine which direction needs priority
    size_t max_queue_idx = std::distance(state.queue_lengths.begin(),
        std::max_element(state.queue_lengths.begin(), state.queue_lengths.end()));
    
    // Simple logic: give green to direction with highest queue
    if (max_queue_idx < state.queue_lengths.size() / 2) {
        return SignalPhase::NS_GREEN_EW_RED;
    } else {
        return SignalPhase::NS_RED_EW_GREEN;
    }
}

void RLPolicyNet::receivePeerVote(const PeerVoteMessage& vote) {
    std::lock_guard<std::mutex> lock(vote_mutex_);
    
    // Remove old votes (older than 5 seconds)
    auto now = std::chrono::system_clock::now();
    peer_votes_.erase(
        std::remove_if(peer_votes_.begin(), peer_votes_.end(),
            [now](const PeerVoteMessage& v) {
                return std::chrono::duration_cast<std::chrono::seconds>(
                    now - v.timestamp).count() > 5;
            }),
        peer_votes_.end()
    );
    
    // Add new vote
    peer_votes_.push_back(vote);
}

SignalPhase RLPolicyNet::computeConsensusPhase(const RLState& state) {
    std::lock_guard<std::mutex> lock(vote_mutex_);
    
    if (peer_votes_.size() < config_.min_voters) {
        return SignalPhase::ALL_RED;  // No consensus
    }
    
    // Count votes by phase
    std::unordered_map<SignalPhase, float> phase_scores;
    
    for (const auto& vote : peer_votes_) {
        phase_scores[vote.proposed_phase] += vote.confidence;
    }
    
    // Find phase with highest score
    SignalPhase best_phase = SignalPhase::ALL_RED;
    float best_score = 0.0f;
    
    for (const auto& [phase, score] : phase_scores) {
        if (score > best_score) {
            best_score = score;
            best_phase = phase;
        }
    }
    
    // Check if consensus threshold is met
    float total_score = 0.0f;
    for (const auto& [phase, score] : phase_scores) {
        total_score += score;
    }
    
    if (best_score / total_score >= config_.vote_threshold) {
        return best_phase;
    }
    
    return SignalPhase::ALL_RED;  // No consensus
}

void RLPolicyNet::enableElasticWeightConsolidation(bool enable) {
    ewc_.enabled = enable;
    
    if (enable && ewc_.reference_weights_.empty()) {
        // Store current weights as reference
        ewc_.reference_weights_["actor_W1"] = actor_.W1;
        ewc_.reference_weights_["actor_W2"] = actor_.W2;
        ewc_.reference_weights_["critic_W1"] = critic_.W1;
        ewc_.reference_weights_["critic_W2"] = critic_.W2;
    }
}

void RLPolicyNet::computeFisherInformation(const std::vector<RLState>& states) {
    if (!ewc_.enabled) return;
    
    // Initialize Fisher information matrices
    ewc_.fisher_information_["actor_W1"] = Tensor(actor_.W1.shape());
    ewc_.fisher_information_["actor_W2"] = Tensor(actor_.W2.shape());
    ewc_.fisher_information_["critic_W1"] = Tensor(critic_.W1.shape());
    ewc_.fisher_information_["critic_W2"] = Tensor(critic_.W2.shape());
    
    // Zero out
    std::memset(ewc_.fisher_information_["actor_W1"].data_float(), 0,
               ewc_.fisher_information_["actor_W1"].size() * sizeof(float));
    std::memset(ewc_.fisher_information_["actor_W2"].data_float(), 0,
               ewc_.fisher_information_["actor_W2"].size() * sizeof(float));
    std::memset(ewc_.fisher_information_["critic_W1"].data_float(), 0,
               ewc_.fisher_information_["critic_W1"].size() * sizeof(float));
    std::memset(ewc_.fisher_information_["critic_W2"].data_float(), 0,
               ewc_.fisher_information_["critic_W2"].size() * sizeof(float));
    
    // Compute Fisher information by sampling gradients from important states
    // Fisher information approximates the curvature of the loss landscape
    
    for (const auto& state : states) {
        // Convert state to tensor
        auto features = state.toFeatureVector();
        Tensor state_tensor({1, static_cast<int>(features.size())});
        std::memcpy(state_tensor.data_float(), features.data(),
                   features.size() * sizeof(float));
        
        // Forward pass
        Tensor logits = actorForward(state_tensor);
        Tensor values = criticForward(state_tensor);
        
        // Compute policy distribution
        float* logit_data = logits.data_float();
        float max_logit = *std::max_element(logit_data, logit_data + config_.num_actions);
        
        std::vector<float> probs(config_.num_actions);
        float sum_exp = 0.0f;
        for (size_t i = 0; i < config_.num_actions; ++i) {
            probs[i] = std::exp(logit_data[i] - max_logit);
            sum_exp += probs[i];
        }
        for (size_t i = 0; i < config_.num_actions; ++i) {
            probs[i] /= sum_exp;
        }
        
        // Sample action from current policy
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        std::mt19937 gen(std::random_device{}());
        int sampled_action = dist(gen);
        
        // Compute gradients for sampled action (log likelihood)
        // For actor network
        {
            // Hidden layer activations
            Tensor h1({1, static_cast<int>(config_.hidden_dim)});
            optimizedMatMul(actor_.W1.data_float(), state_tensor.data_float(), 
                           h1.data_float(), config_.hidden_dim, 1, config_.state_dim);
            
            float* h1_data = h1.data_float();
            const float* b1_data = actor_.b1.data_float();
            for (size_t j = 0; j < config_.hidden_dim; ++j) {
                h1_data[j] = std::max(0.0f, h1_data[j] + b1_data[j]);
            }
            
            // Gradient of log probability w.r.t. output layer
            for (size_t j = 0; j < config_.num_actions; ++j) {
                float grad = (j == sampled_action ? 1.0f : 0.0f) - probs[j];
                grad = grad * grad;  // Square for Fisher information
                
                // Accumulate Fisher information for W2
                for (size_t k = 0; k < config_.hidden_dim; ++k) {
                    size_t idx = j * config_.hidden_dim + k;
                    ewc_.fisher_information_["actor_W2"].data_float()[idx] += 
                        grad * h1_data[k] * h1_data[k];
                }
            }
            
            // Gradient w.r.t. hidden layer (backprop through ReLU)
            std::vector<float> hidden_grads(config_.hidden_dim, 0.0f);
            for (size_t j = 0; j < config_.hidden_dim; ++j) {
                if (h1_data[j] > 0) {  // ReLU derivative
                    for (size_t k = 0; k < config_.num_actions; ++k) {
                        float output_grad = (k == sampled_action ? 1.0f : 0.0f) - probs[k];
                        hidden_grads[j] += output_grad * actor_.W2.data_float()[k * config_.hidden_dim + j];
                    }
                    hidden_grads[j] = hidden_grads[j] * hidden_grads[j];  // Square for Fisher
                    
                    // Accumulate Fisher information for W1
                    for (size_t k = 0; k < config_.state_dim; ++k) {
                        size_t idx = j * config_.state_dim + k;
                        ewc_.fisher_information_["actor_W1"].data_float()[idx] += 
                            hidden_grads[j] * features[k] * features[k];
                    }
                }
            }
        }
        
        // For critic network (using squared gradient of value prediction)
        {
            // Hidden layer activations
            Tensor h1({1, static_cast<int>(config_.hidden_dim)});
            optimizedMatMul(critic_.W1.data_float(), state_tensor.data_float(), 
                           h1.data_float(), config_.hidden_dim, 1, config_.state_dim);
            
            float* h1_data = h1.data_float();
            const float* b1_data = critic_.b1.data_float();
            std::vector<bool> relu_mask(config_.hidden_dim);
            
            for (size_t j = 0; j < config_.hidden_dim; ++j) {
                h1_data[j] += b1_data[j];
                relu_mask[j] = h1_data[j] > 0;
                h1_data[j] = std::max(0.0f, h1_data[j]);
            }
            
            // Gradient of value w.r.t. output is 1
            float value_grad = 1.0f;
            
            // Fisher information for critic W2
            for (size_t j = 0; j < config_.hidden_dim; ++j) {
                ewc_.fisher_information_["critic_W2"].data_float()[j] += 
                    value_grad * value_grad * h1_data[j] * h1_data[j];
            }
            
            // Backprop to hidden layer
            for (size_t j = 0; j < config_.hidden_dim; ++j) {
                if (relu_mask[j]) {
                    float hidden_grad = value_grad * critic_.W2.data_float()[j];
                    hidden_grad = hidden_grad * hidden_grad;  // Square for Fisher
                    
                    // Fisher information for critic W1
                    for (size_t k = 0; k < config_.state_dim; ++k) {
                        size_t idx = j * config_.state_dim + k;
                        ewc_.fisher_information_["critic_W1"].data_float()[idx] += 
                            hidden_grad * features[k] * features[k];
                    }
                }
            }
        }
    }
    
    // Normalize by number of samples and add small epsilon for numerical stability
    float norm = 1.0f / states.size();
    float epsilon = 1e-3f;
    
    for (auto& [name, fisher] : ewc_.fisher_information_) {
        for (size_t i = 0; i < fisher.size(); ++i) {
            fisher.data_float()[i] = fisher.data_float()[i] * norm + epsilon;
        }
    }
}

void RLPolicyNet::broadcastVote(SignalPhase phase, float confidence, 
                               const std::string& reason) {
    // Prepare V2X message for broadcast
    PeerVoteMessage vote;
    vote.agent_id = "local_agent";  // Configured per intersection in deployment
    vote.proposed_phase = phase;
    vote.confidence = confidence;
    vote.reason = reason;
    vote.timestamp = std::chrono::system_clock::now();
    
    // V2X broadcast implementation:
    // 1. Message serialization to standardized format
    // 2. Transmission via DSRC/C-V2X protocols
    // 3. Network failure handling with retry logic
    // 4. Message authentication and encryption
    
    // Debug logging when peer voting enabled
    if (config_.enable_peer_voting) {
        std::cout << "Broadcasting vote: phase=" << static_cast<int>(phase) 
                  << ", confidence=" << confidence 
                  << ", reason=" << reason << std::endl;
    }
}

void RLPolicyNet::saveWeights(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving weights: " + path);
    }
    
    // Write magic number and version
    uint32_t magic = 0x524C504E;  // "RLPN"
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Write config
    file.write(reinterpret_cast<const char*>(&config_.state_dim), sizeof(config_.state_dim));
    file.write(reinterpret_cast<const char*>(&config_.hidden_dim), sizeof(config_.hidden_dim));
    file.write(reinterpret_cast<const char*>(&config_.num_actions), sizeof(config_.num_actions));
    
    // Helper to write tensor
    auto writeTensor = [&file](const Tensor& tensor) {
        size_t num_dims = tensor.shape().size();
        file.write(reinterpret_cast<const char*>(&num_dims), sizeof(num_dims));
        // Write shape as int values
        for (int dim : tensor.shape()) {
            file.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        }
        file.write(reinterpret_cast<const char*>(tensor.data_float()),
                  tensor.size() * sizeof(float));
    };
    
    // Write actor weights
    writeTensor(actor_.W1);
    writeTensor(actor_.b1);
    writeTensor(actor_.W2);
    writeTensor(actor_.b2);
    
    // Write critic weights
    writeTensor(critic_.W1);
    writeTensor(critic_.b1);
    writeTensor(critic_.W2);
    writeTensor(critic_.b2);
    
    file.close();
}

void RLPolicyNet::loadWeights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading weights: " + path);
    }
    
    // Check magic number and version
    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    if (magic != 0x524C504E || version != 1) {
        throw std::runtime_error("Invalid RLPolicyNet weight file");
    }
    
    // Read config and verify compatibility
    size_t state_dim, hidden_dim, num_actions;
    file.read(reinterpret_cast<char*>(&state_dim), sizeof(state_dim));
    file.read(reinterpret_cast<char*>(&hidden_dim), sizeof(hidden_dim));
    file.read(reinterpret_cast<char*>(&num_actions), sizeof(num_actions));
    
    if (state_dim != config_.state_dim || 
        hidden_dim != config_.hidden_dim ||
        num_actions != config_.num_actions) {
        throw std::runtime_error("Weight file configuration mismatch");
    }
    
    // Helper to read tensor
    auto readTensor = [&file](Tensor& tensor) {
        size_t num_dims;
        file.read(reinterpret_cast<char*>(&num_dims), sizeof(num_dims));
        
        std::vector<int> shape(num_dims);
        // Read shape as int values
        for (size_t i = 0; i < num_dims; ++i) {
            file.read(reinterpret_cast<char*>(&shape[i]), sizeof(int));
        }
        
        tensor = Tensor(shape);
        file.read(reinterpret_cast<char*>(tensor.data_float()),
                 tensor.size() * sizeof(float));
    };
    
    // Read actor weights
    readTensor(actor_.W1);
    readTensor(actor_.b1);
    readTensor(actor_.W2);
    readTensor(actor_.b2);
    
    // Read critic weights
    readTensor(critic_.W1);
    readTensor(critic_.b1);
    readTensor(critic_.W2);
    readTensor(critic_.b2);
    
    file.close();
}

void RLPolicyNet::updateActor(const Tensor& states, const Tensor& actions, 
                              const Tensor& advantages) {
    size_t batch_size = states.shape()[0];
    
    // Forward pass through actor
    Tensor logits = actorForward(states);
    
    // Compute policy gradients
    Tensor actor_grads_W2({static_cast<int>(config_.num_actions), 
                          static_cast<int>(config_.hidden_dim)});
    Tensor actor_grads_b2({static_cast<int>(config_.num_actions)});
    std::memset(actor_grads_W2.data_float(), 0, actor_grads_W2.size() * sizeof(float));
    std::memset(actor_grads_b2.data_float(), 0, actor_grads_b2.size() * sizeof(float));
    
    // Compute hidden layer activations for gradient computation
    Tensor h1({static_cast<int>(batch_size), static_cast<int>(config_.hidden_dim)});
    optimizedMatMul(actor_.W1.data_float(), states.data_float(), h1.data_float(),
                   config_.hidden_dim, batch_size, config_.state_dim);
    
    // Add bias and apply ReLU
    float* h1_data = h1.data_float();
    const float* b1_data = actor_.b1.data_float();
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            size_t idx = i * config_.hidden_dim + j;
            h1_data[idx] = std::max(0.0f, h1_data[idx] + b1_data[j]);
        }
    }
    
    // Gradient computation for output layer
    for (size_t i = 0; i < batch_size; ++i) {
        // Get softmax probabilities
        const float* sample_logits = logits.data_float() + i * config_.num_actions;
        float max_logit = *std::max_element(sample_logits, 
                                          sample_logits + config_.num_actions);
        
        std::vector<float> probs(config_.num_actions);
        float sum_exp = 0.0f;
        for (size_t j = 0; j < config_.num_actions; ++j) {
            probs[j] = std::exp(sample_logits[j] - max_logit);
            sum_exp += probs[j];
        }
        for (size_t j = 0; j < config_.num_actions; ++j) {
            probs[j] /= sum_exp;
        }
        
        // Compute gradients
        int action = static_cast<int>(actions.data_float()[i]);
        float advantage = advantages.data_float()[i];
        
        for (size_t j = 0; j < config_.num_actions; ++j) {
            float grad = -probs[j] * advantage;
            if (j == action) {
                grad += advantage;
            }
            
            // Accumulate gradients for W2
            for (size_t k = 0; k < config_.hidden_dim; ++k) {
                actor_grads_W2.data_float()[j * config_.hidden_dim + k] += 
                    grad * h1_data[i * config_.hidden_dim + k] / batch_size;
            }
            
            // Accumulate gradients for b2
            actor_grads_b2.data_float()[j] += grad / batch_size;
        }
    }
    
    // Add entropy gradient for exploration
    for (size_t i = 0; i < batch_size; ++i) {
        const float* sample_logits = logits.data_float() + i * config_.num_actions;
        float max_logit = *std::max_element(sample_logits, 
                                          sample_logits + config_.num_actions);
        
        std::vector<float> probs(config_.num_actions);
        float sum_exp = 0.0f;
        for (size_t j = 0; j < config_.num_actions; ++j) {
            probs[j] = std::exp(sample_logits[j] - max_logit);
            sum_exp += probs[j];
        }
        for (size_t j = 0; j < config_.num_actions; ++j) {
            probs[j] /= sum_exp;
        }
        
        // Entropy gradient
        for (size_t j = 0; j < config_.num_actions; ++j) {
            float entropy_grad = config_.entropy_coef * (1.0f + std::log(probs[j] + 1e-8f));
            
            for (size_t k = 0; k < config_.hidden_dim; ++k) {
                actor_grads_W2.data_float()[j * config_.hidden_dim + k] += 
                    entropy_grad * probs[j] * h1_data[i * config_.hidden_dim + k] / batch_size;
            }
            actor_grads_b2.data_float()[j] += entropy_grad * probs[j] / batch_size;
        }
    }
    
    // Apply gradient clipping
    float grad_norm = 0.0f;
    for (size_t i = 0; i < actor_grads_W2.size(); ++i) {
        grad_norm += actor_grads_W2.data_float()[i] * actor_grads_W2.data_float()[i];
    }
    for (size_t i = 0; i < actor_grads_b2.size(); ++i) {
        grad_norm += actor_grads_b2.data_float()[i] * actor_grads_b2.data_float()[i];
    }
    grad_norm = std::sqrt(grad_norm);
    
    if (grad_norm > config_.max_grad_norm) {
        float scale = config_.max_grad_norm / grad_norm;
        for (size_t i = 0; i < actor_grads_W2.size(); ++i) {
            actor_grads_W2.data_float()[i] *= scale;
        }
        for (size_t i = 0; i < actor_grads_b2.size(); ++i) {
            actor_grads_b2.data_float()[i] *= scale;
        }
    }
    
    // Update weights using Adam optimizer
    auto updateWithAdam = [this](Tensor& weights, const Tensor& grads, 
                                AdamState& state, const std::string& name) {
        if (state.m.size() != weights.size()) {
            state.m = Tensor(weights.shape());
            state.v = Tensor(weights.shape());
            std::memset(state.m.data_float(), 0, state.m.size() * sizeof(float));
            std::memset(state.v.data_float(), 0, state.v.size() * sizeof(float));
        }
        
        state.t++;
        float lr_t = config_.learning_rate * std::sqrt(1.0f - std::pow(state.beta2, state.t)) / 
                     (1.0f - std::pow(state.beta1, state.t));
        
        float* w_data = weights.data_float();
        const float* g_data = grads.data_float();
        float* m_data = state.m.data_float();
        float* v_data = state.v.data_float();
        
        for (size_t i = 0; i < weights.size(); ++i) {
            m_data[i] = state.beta1 * m_data[i] + (1.0f - state.beta1) * g_data[i];
            v_data[i] = state.beta2 * v_data[i] + (1.0f - state.beta2) * g_data[i] * g_data[i];
            w_data[i] -= lr_t * m_data[i] / (std::sqrt(v_data[i]) + state.epsilon);
        }
    };
    
    // Update actor weights
    updateWithAdam(actor_.W2, actor_grads_W2, actor_optimizer_state_["W2"], "actor_W2");
    updateWithAdam(actor_.b2, actor_grads_b2, actor_optimizer_state_["b2"], "actor_b2");
    
    // Compute gradients for hidden layer (backprop through ReLU)
    Tensor actor_grads_W1({static_cast<int>(config_.hidden_dim), 
                          static_cast<int>(config_.state_dim)});
    Tensor actor_grads_b1({static_cast<int>(config_.hidden_dim)});
    std::memset(actor_grads_W1.data_float(), 0, actor_grads_W1.size() * sizeof(float));
    std::memset(actor_grads_b1.data_float(), 0, actor_grads_b1.size() * sizeof(float));
    
    // Backpropagate through hidden layer
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            float h1_val = h1_data[i * config_.hidden_dim + j];
            if (h1_val > 0) {  // ReLU derivative
                float grad = 0.0f;
                for (size_t k = 0; k < config_.num_actions; ++k) {
                    grad += actor_grads_W2.data_float()[k * config_.hidden_dim + j] * 
                           logits.data_float()[i * config_.num_actions + k];
                }
                
                // Update W1 gradients
                for (size_t k = 0; k < config_.state_dim; ++k) {
                    actor_grads_W1.data_float()[j * config_.state_dim + k] += 
                        grad * states.data_float()[i * config_.state_dim + k] / batch_size;
                }
                
                // Update b1 gradients
                actor_grads_b1.data_float()[j] += grad / batch_size;
            }
        }
    }
    
    // Update hidden layer weights
    updateWithAdam(actor_.W1, actor_grads_W1, actor_optimizer_state_["W1"], "actor_W1");
    updateWithAdam(actor_.b1, actor_grads_b1, actor_optimizer_state_["b1"], "actor_b1");
}

void RLPolicyNet::updateCritic(const Tensor& states, const Tensor& returns) {
    size_t batch_size = states.shape()[0];
    
    // Forward pass through critic
    Tensor values = criticForward(states);
    
    // Compute MSE loss gradients
    Tensor value_grads({static_cast<int>(batch_size), 1});
    for (size_t i = 0; i < batch_size; ++i) {
        value_grads.data_float()[i] = 2.0f * (values.data_float()[i] - 
                                              returns.data_float()[i]) / batch_size;
    }
    
    // Compute hidden layer activations
    Tensor h1({static_cast<int>(batch_size), static_cast<int>(config_.hidden_dim)});
    optimizedMatMul(critic_.W1.data_float(), states.data_float(), h1.data_float(),
                   config_.hidden_dim, batch_size, config_.state_dim);
    
    // Add bias and apply ReLU
    float* h1_data = h1.data_float();
    const float* b1_data = critic_.b1.data_float();
    std::vector<bool> relu_mask(batch_size * config_.hidden_dim);
    
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            size_t idx = i * config_.hidden_dim + j;
            h1_data[idx] += b1_data[j];
            relu_mask[idx] = h1_data[idx] > 0;
            h1_data[idx] = std::max(0.0f, h1_data[idx]);
        }
    }
    
    // Compute gradients for output layer
    Tensor critic_grads_W2({1, static_cast<int>(config_.hidden_dim)});
    Tensor critic_grads_b2({1});
    std::memset(critic_grads_W2.data_float(), 0, critic_grads_W2.size() * sizeof(float));
    std::memset(critic_grads_b2.data_float(), 0, critic_grads_b2.size() * sizeof(float));
    
    for (size_t i = 0; i < batch_size; ++i) {
        float grad = value_grads.data_float()[i] * config_.value_loss_coef;
        
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            critic_grads_W2.data_float()[j] += grad * h1_data[i * config_.hidden_dim + j];
        }
        critic_grads_b2.data_float()[0] += grad;
    }
    
    // Apply gradient clipping
    float grad_norm = 0.0f;
    for (size_t i = 0; i < critic_grads_W2.size(); ++i) {
        grad_norm += critic_grads_W2.data_float()[i] * critic_grads_W2.data_float()[i];
    }
    grad_norm = std::sqrt(grad_norm + critic_grads_b2.data_float()[0] * 
                         critic_grads_b2.data_float()[0]);
    
    if (grad_norm > config_.max_grad_norm) {
        float scale = config_.max_grad_norm / grad_norm;
        for (size_t i = 0; i < critic_grads_W2.size(); ++i) {
            critic_grads_W2.data_float()[i] *= scale;
        }
        critic_grads_b2.data_float()[0] *= scale;
    }
    
    // Update with Adam
    auto updateWithAdam = [this](Tensor& weights, const Tensor& grads, 
                                AdamState& state, const std::string& name) {
        if (state.m.size() != weights.size()) {
            state.m = Tensor(weights.shape());
            state.v = Tensor(weights.shape());
            std::memset(state.m.data_float(), 0, state.m.size() * sizeof(float));
            std::memset(state.v.data_float(), 0, state.v.size() * sizeof(float));
        }
        
        state.t++;
        float lr_t = config_.learning_rate * std::sqrt(1.0f - std::pow(state.beta2, state.t)) / 
                     (1.0f - std::pow(state.beta1, state.t));
        
        float* w_data = weights.data_float();
        const float* g_data = grads.data_float();
        float* m_data = state.m.data_float();
        float* v_data = state.v.data_float();
        
        for (size_t i = 0; i < weights.size(); ++i) {
            m_data[i] = state.beta1 * m_data[i] + (1.0f - state.beta1) * g_data[i];
            v_data[i] = state.beta2 * v_data[i] + (1.0f - state.beta2) * g_data[i] * g_data[i];
            w_data[i] -= lr_t * m_data[i] / (std::sqrt(v_data[i]) + state.epsilon);
        }
    };
    
    // Update critic output layer
    updateWithAdam(critic_.W2, critic_grads_W2, critic_optimizer_state_["W2"], "critic_W2");
    updateWithAdam(critic_.b2, critic_grads_b2, critic_optimizer_state_["b2"], "critic_b2");
    
    // Compute gradients for hidden layer
    Tensor critic_grads_W1({static_cast<int>(config_.hidden_dim), 
                           static_cast<int>(config_.state_dim)});
    Tensor critic_grads_b1({static_cast<int>(config_.hidden_dim)});
    std::memset(critic_grads_W1.data_float(), 0, critic_grads_W1.size() * sizeof(float));
    std::memset(critic_grads_b1.data_float(), 0, critic_grads_b1.size() * sizeof(float));
    
    // Backpropagate through hidden layer
    for (size_t i = 0; i < batch_size; ++i) {
        float output_grad = value_grads.data_float()[i] * config_.value_loss_coef;
        
        for (size_t j = 0; j < config_.hidden_dim; ++j) {
            if (relu_mask[i * config_.hidden_dim + j]) {  // ReLU derivative
                float grad = output_grad * critic_.W2.data_float()[j];
                
                // Update W1 gradients
                for (size_t k = 0; k < config_.state_dim; ++k) {
                    critic_grads_W1.data_float()[j * config_.state_dim + k] += 
                        grad * states.data_float()[i * config_.state_dim + k];
                }
                
                // Update b1 gradients
                critic_grads_b1.data_float()[j] += grad;
            }
        }
    }
    
    // Update hidden layer weights
    updateWithAdam(critic_.W1, critic_grads_W1, critic_optimizer_state_["W1"], "critic_W1");
    updateWithAdam(critic_.b1, critic_grads_b1, critic_optimizer_state_["b1"], "critic_b1");
}

// StandardTrafficReward is already implemented in the header

} // namespace tacs