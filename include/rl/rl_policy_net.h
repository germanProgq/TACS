/*
 * Advantage Actor-Critic (A2C) implementation for traffic signal control
 * Production-ready RL agent with sub-3ms decision latency
 */

#ifndef RL_POLICY_NET_H
#define RL_POLICY_NET_H

#include <vector>
#include <memory>
#include <array>
#include <cmath>
#include <chrono>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <random>
#include <set>
#include <string>
#include "../core/tensor.h"
#include "../core/memory_manager.h"
#include "../utils/matrix_ops.h"

namespace tacs {

using core::Tensor;
using core::MemoryManager;

// Traffic signal phase enumeration
enum class SignalPhase {
    NS_GREEN_EW_RED = 0,     // North-South green, East-West red
    NS_YELLOW_EW_RED = 1,    // North-South yellow transition
    NS_RED_EW_GREEN = 2,     // North-South red, East-West green  
    NS_RED_EW_YELLOW = 3,    // East-West yellow transition
    ALL_RED = 4,             // All directions red (safety phase)
    EMERGENCY_OVERRIDE = 5    // Emergency vehicle override
};

// RL state representation
struct RLState {
    std::vector<float> queue_lengths;      // Vehicle queue lengths per lane
    std::vector<float> pedestrian_counts;  // Pedestrian waiting counts
    std::vector<float> cyclist_counts;     // Cyclist counts per direction
    float weather_condition;               // 0=clear, 1=rain, 2=fog, 3=snow
    float accident_indicator;              // 0=normal, >0 indicates accident severity
    float current_phase_duration;          // Time in current phase (seconds)
    float time_of_day;                     // Normalized time (0-1)
    
    // Flatten to feature vector for neural network
    std::vector<float> toFeatureVector() const;
    size_t getFeatureSize() const;
};

// Experience for replay buffer
struct Experience {
    RLState state;
    int action;
    float reward;
    RLState next_state;
    bool done;
    float priority;  // For prioritized experience replay
};

// Peer voting message for multi-agent coordination
struct PeerVoteMessage {
    std::string agent_id;
    SignalPhase proposed_phase;
    float confidence;
    std::string reason;  // "emergency", "congestion", "accident"
    std::chrono::system_clock::time_point timestamp;
};

class RLPolicyNet {
public:
    struct Config {
        // Network architecture
        size_t state_dim;        // Input state dimension
        size_t hidden_dim;      // Hidden layer size
        size_t num_actions;       // Number of signal phases
        
        // Training hyperparameters
        float learning_rate;
        float gamma;          // Discount factor
        float entropy_coef;   // Entropy regularization
        float value_loss_coef; // Value function loss weight
        float max_grad_norm;   // Gradient clipping
        
        // Experience replay
        size_t replay_buffer_size;
        size_t batch_size;
        float priority_alpha;   // Prioritization exponent
        float priority_beta;    // Importance sampling
        
        // Performance constraints
        float max_decision_time_ms;  // Maximum decision latency
        
        // Multi-agent coordination
        bool enable_peer_voting;
        float vote_threshold;   // Consensus threshold
        size_t min_voters;         // Minimum peers for override
        
        Config() : 
            state_dim(32),
            hidden_dim(128),
            num_actions(6),
            learning_rate(3e-4f),
            gamma(0.99f),
            entropy_coef(0.01f),
            value_loss_coef(0.5f),
            max_grad_norm(0.5f),
            replay_buffer_size(10000),
            batch_size(32),
            priority_alpha(0.6f),
            priority_beta(0.4f),
            max_decision_time_ms(3.0f),
            enable_peer_voting(true),
            vote_threshold(0.7f),
            min_voters(2) {}
    };
    
    explicit RLPolicyNet(const Config& config = Config());
    ~RLPolicyNet();
    
    // Core A2C methods
    SignalPhase selectAction(const RLState& state, bool training = false);
    void train(const std::vector<Experience>& batch);
    float computeAdvantage(const Experience& exp);
    
    // Network forward pass with SIMD optimization
    std::pair<Tensor, Tensor> forward(const Tensor& state_batch);
    
    // Loss computation
    struct A2CLoss {
        float policy_loss;
        float value_loss;
        float entropy_loss;
        float total_loss;
    };
    A2CLoss computeLoss(const std::vector<Experience>& batch);
    
    // Experience replay management
    void addExperience(const Experience& exp);
    std::vector<Experience> sampleBatch();
    void updatePriorities(const std::vector<size_t>& indices, 
                         const std::vector<float>& td_errors);
    
    // Multi-agent coordination
    void receivePeerVote(const PeerVoteMessage& vote);
    SignalPhase computeConsensusPhase(const RLState& state);
    void broadcastVote(SignalPhase phase, float confidence, 
                      const std::string& reason);
    
    // Emergency override system
    bool checkEmergencyOverride(const RLState& state);
    SignalPhase getEmergencyPhase(const RLState& state);
    
    // Catastrophic forgetting prevention
    void enableElasticWeightConsolidation(bool enable);
    void computeFisherInformation(const std::vector<RLState>& states);
    
    // Model persistence
    void saveWeights(const std::string& path) const;
    void loadWeights(const std::string& path);
    
    // Performance monitoring
    struct PerformanceStats {
        float avg_decision_time_ms;
        float max_decision_time_ms;
        size_t decisions_made;
        float avg_reward;
        float success_rate;
    };
    PerformanceStats getStats() const { return stats_; }
    
private:
    Config config_;
    MemoryManager* memory_manager_;  // Singleton instance
    
    // Actor-Critic networks
    struct ActorNetwork {
        Tensor W1, b1;  // Hidden layer
        Tensor W2, b2;  // Output layer (policy logits)
    };
    
    struct CriticNetwork {
        Tensor W1, b1;  // Hidden layer
        Tensor W2, b2;  // Output layer (value)
    };
    
    ActorNetwork actor_;
    CriticNetwork critic_;
    
    // Optimizers (Adam)
    struct AdamState {
        Tensor m, v;  // First and second moments
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;
        int t = 0;
    };
    
    std::unordered_map<std::string, AdamState> actor_optimizer_state_;
    std::unordered_map<std::string, AdamState> critic_optimizer_state_;
    
    // Experience replay buffer with priority
    struct PrioritizedReplayBuffer {
        std::vector<Experience> buffer;
        std::vector<float> priorities;
        size_t position = 0;
        size_t size = 0;
        std::mt19937 rng{std::random_device{}()};
    };
    PrioritizedReplayBuffer replay_buffer_;
    
    // Peer voting system
    std::vector<PeerVoteMessage> peer_votes_;
    std::mutex vote_mutex_;
    std::atomic<bool> emergency_active_{false};
    
    // Elastic Weight Consolidation for continual learning
    struct EWCData {
        bool enabled = false;
        std::unordered_map<std::string, Tensor> fisher_information_;
        std::unordered_map<std::string, Tensor> reference_weights_;
        float lambda = 100.0f;  // EWC penalty strength
    };
    EWCData ewc_;
    
    // Performance tracking
    mutable PerformanceStats stats_{};
    mutable std::chrono::high_resolution_clock::time_point last_decision_time_;
    
    // Internal methods
    void initializeWeights();
    Tensor actorForward(const Tensor& state);
    Tensor criticForward(const Tensor& state);
    void updateActor(const Tensor& states, const Tensor& actions, 
                    const Tensor& advantages);
    void updateCritic(const Tensor& states, const Tensor& returns);
    float computeEntropy(const Tensor& logits);
    
    // SIMD-optimized operations
    void optimizedMatMul(const float* A, const float* B, float* C,
                        size_t M, size_t N, size_t K);
};

// Traffic environment interface
class TrafficEnvironment {
public:
    virtual ~TrafficEnvironment() = default;
    
    // Get current intersection state
    virtual RLState getCurrentState() const = 0;
    
    // Apply signal phase change
    virtual void setSignalPhase(SignalPhase phase) = 0;
    
    // Compute reward based on current conditions
    virtual float computeReward(const RLState& state, 
                               SignalPhase action) const = 0;
    
    // Check if episode is complete
    virtual bool isEpisodeDone() const = 0;
    
    // Reset environment for new episode
    virtual void reset() = 0;
};

// Standard traffic reward function implementation
class StandardTrafficReward {
public:
    static float compute(const RLState& state, SignalPhase action) {
        // Base reward: negative mean queue length
        float queue_penalty = 0.0f;
        for (float q : state.queue_lengths) {
            queue_penalty += q;
        }
        queue_penalty /= state.queue_lengths.size();
        
        // Accident penalty
        float accident_penalty = 5.0f * state.accident_indicator;
        
        // Pedestrian/cyclist wait penalty
        float ped_penalty = 0.0f;
        for (float p : state.pedestrian_counts) {
            ped_penalty += p * 0.5f;  // Lower weight than vehicles
        }
        ped_penalty /= state.pedestrian_counts.size();
        
        // Phase duration penalty (discourage too frequent switching)
        float switch_penalty = 0.0f;
        if (state.current_phase_duration < 10.0f) {  // Minimum green time
            switch_penalty = 2.0f;
        }
        
        // Weather adjustment
        float weather_factor = 1.0f + 0.2f * state.weather_condition;
        
        // Total reward
        float reward = -(queue_penalty + accident_penalty + 
                        ped_penalty + switch_penalty) * weather_factor;
        
        return reward;
    }
};

} // namespace tacs

#endif // RL_POLICY_NET_H