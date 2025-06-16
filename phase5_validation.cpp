/*
 * Phase 5 Validation: Reinforcement Learning Engine
 * Tests A2C implementation, performance constraints, and multi-agent coordination
 */

#include <iostream>
#include <chrono>
#include <random>
#include <thread>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include "include/rl/rl_policy_net.h"

using namespace tacs;
using namespace std::chrono;

// Mock traffic environment for testing
class MockTrafficEnvironment : public TrafficEnvironment {
private:
    RLState current_state_;
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> queue_dist_{0.0f, 50.0f};
    std::uniform_real_distribution<float> ped_dist_{0.0f, 20.0f};
    std::uniform_real_distribution<float> cyclist_dist_{0.0f, 10.0f};
    std::uniform_int_distribution<int> weather_dist_{0, 3};
    std::bernoulli_distribution accident_dist_{0.05};  // 5% accident chance
    size_t step_count_ = 0;
    SignalPhase current_phase_ = SignalPhase::NS_GREEN_EW_RED;
    
public:
    MockTrafficEnvironment() {
        reset();
    }
    
    RLState getCurrentState() const override {
        return current_state_;
    }
    
    void setSignalPhase(SignalPhase phase) override {
        current_phase_ = phase;
        // Update phase duration
        current_state_.current_phase_duration += 1.0f;
        
        // Simulate traffic flow based on signal
        updateTrafficFlow();
    }
    
    float computeReward(const RLState& state, SignalPhase action) const override {
        return StandardTrafficReward::compute(state, action);
    }
    
    bool isEpisodeDone() const override {
        return step_count_ >= 1000;  // Episode length
    }
    
    void reset() override {
        // Initialize with random traffic conditions
        current_state_.queue_lengths.resize(8);  // 8 lanes
        current_state_.pedestrian_counts.resize(4);  // 4 crosswalks
        current_state_.cyclist_counts.resize(4);  // 4 directions
        
        for (auto& q : current_state_.queue_lengths) {
            q = queue_dist_(rng_);
        }
        for (auto& p : current_state_.pedestrian_counts) {
            p = ped_dist_(rng_);
        }
        for (auto& c : current_state_.cyclist_counts) {
            c = cyclist_dist_(rng_);
        }
        
        current_state_.weather_condition = static_cast<float>(weather_dist_(rng_));
        current_state_.accident_indicator = 0.0f;
        current_state_.current_phase_duration = 0.0f;
        current_state_.time_of_day = 0.5f;  // Noon
        
        step_count_ = 0;
        current_phase_ = SignalPhase::NS_GREEN_EW_RED;
    }
    
    void step() {
        step_count_++;
        
        // Update time of day
        current_state_.time_of_day += 1.0f / 86400.0f;  // 1 second per step
        if (current_state_.time_of_day > 1.0f) {
            current_state_.time_of_day -= 1.0f;
        }
        
        // Random accident occurrence
        if (accident_dist_(rng_)) {
            current_state_.accident_indicator = 1.0f;
        } else {
            current_state_.accident_indicator *= 0.95f;  // Decay
        }
        
        // Simulate queue changes
        updateTrafficFlow();
    }
    
private:
    void updateTrafficFlow() {
        // Simulate traffic dynamics based on current signal
        std::normal_distribution<float> flow_noise(0.0f, 2.0f);
        
        for (size_t i = 0; i < current_state_.queue_lengths.size(); ++i) {
            float& queue = current_state_.queue_lengths[i];
            
            // Determine if this lane has green
            bool has_green = false;
            if (current_phase_ == SignalPhase::NS_GREEN_EW_RED && i < 4) {
                has_green = true;
            } else if (current_phase_ == SignalPhase::NS_RED_EW_GREEN && i >= 4) {
                has_green = true;
            }
            
            if (has_green) {
                // Vehicles leaving
                queue = std::max(0.0f, queue - 3.0f + flow_noise(rng_));
            } else {
                // Vehicles accumulating
                queue = std::min(50.0f, queue + 2.0f + flow_noise(rng_));
            }
        }
        
        // Update pedestrian counts
        for (auto& p : current_state_.pedestrian_counts) {
            p = std::max(0.0f, p + flow_noise(rng_) * 0.5f);
            p = std::min(20.0f, p);
        }
        
        // Update cyclist counts
        for (auto& c : current_state_.cyclist_counts) {
            c = std::max(0.0f, c + flow_noise(rng_) * 0.3f);
            c = std::min(10.0f, c);
        }
    }
};

// Performance benchmark
void benchmarkDecisionLatency(RLPolicyNet& policy, MockTrafficEnvironment& env) {
    std::cout << "\n=== Decision Latency Benchmark ===" << std::endl;
    
    const int num_tests = 1000;
    std::vector<float> latencies;
    latencies.reserve(num_tests);
    
    // Warm up
    for (int i = 0; i < 10; ++i) {
        env.step();
        auto state = env.getCurrentState();
        policy.selectAction(state, false);
    }
    
    // Benchmark
    for (int i = 0; i < num_tests; ++i) {
        env.step();
        auto state = env.getCurrentState();
        
        auto start = high_resolution_clock::now();
        SignalPhase action = policy.selectAction(state, false);
        auto end = high_resolution_clock::now();
        
        float latency_ms = duration_cast<microseconds>(end - start).count() / 1000.0f;
        latencies.push_back(latency_ms);
    }
    
    // Calculate statistics
    std::sort(latencies.begin(), latencies.end());
    float avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / num_tests;
    float p50 = latencies[num_tests / 2];
    float p95 = latencies[static_cast<int>(num_tests * 0.95)];
    float p99 = latencies[static_cast<int>(num_tests * 0.99)];
    float max_latency = latencies.back();
    
    std::cout << "Average latency: " << std::fixed << std::setprecision(3) 
              << avg_latency << " ms" << std::endl;
    std::cout << "P50 latency: " << p50 << " ms" << std::endl;
    std::cout << "P95 latency: " << p95 << " ms" << std::endl;
    std::cout << "P99 latency: " << p99 << " ms" << std::endl;
    std::cout << "Max latency: " << max_latency << " ms" << std::endl;
    
    bool meets_constraint = max_latency <= 3.0f;
    std::cout << "Meets <3ms constraint: " << (meets_constraint ? "YES ✓" : "NO ✗") 
              << std::endl;
}

// Test basic A2C functionality
void testA2CTraining(RLPolicyNet& policy, MockTrafficEnvironment& env) {
    std::cout << "\n=== A2C Training Test ===" << std::endl;
    
    const int num_episodes = 10;
    const int steps_per_episode = 100;
    
    std::vector<float> episode_rewards;
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        env.reset();
        float total_reward = 0.0f;
        
        for (int step = 0; step < steps_per_episode; ++step) {
            // Get current state
            RLState state = env.getCurrentState();
            
            // Select action
            SignalPhase action = policy.selectAction(state, true);  // Training mode
            
            // Apply action
            env.setSignalPhase(action);
            env.step();
            
            // Get next state and reward
            RLState next_state = env.getCurrentState();
            float reward = env.computeReward(state, action);
            bool done = env.isEpisodeDone();
            
            // Store experience
            Experience exp{state, static_cast<int>(action), reward, next_state, done};
            policy.addExperience(exp);
            
            total_reward += reward;
            
            // Train every 32 steps
            if (step % 32 == 31) {
                auto batch = policy.sampleBatch();
                if (!batch.empty()) {
                    policy.train(batch);
                }
            }
        }
        
        episode_rewards.push_back(total_reward);
        std::cout << "Episode " << episode + 1 << " reward: " << total_reward << std::endl;
    }
    
    // Check if learning is happening (rewards should improve)
    float early_avg = std::accumulate(episode_rewards.begin(), 
                                     episode_rewards.begin() + 3, 0.0f) / 3.0f;
    float late_avg = std::accumulate(episode_rewards.end() - 3, 
                                    episode_rewards.end(), 0.0f) / 3.0f;
    
    std::cout << "Early episodes average: " << early_avg << std::endl;
    std::cout << "Late episodes average: " << late_avg << std::endl;
    std::cout << "Learning improvement: " << (late_avg > early_avg ? "YES ✓" : "NO ✗") 
              << std::endl;
}

// Test emergency override
void testEmergencyOverride(RLPolicyNet& policy) {
    std::cout << "\n=== Emergency Override Test ===" << std::endl;
    
    // Create state with high accident indicator
    RLState emergency_state;
    emergency_state.queue_lengths = {45.0f, 40.0f, 35.0f, 30.0f, 5.0f, 5.0f, 5.0f, 5.0f};
    emergency_state.pedestrian_counts = {10.0f, 10.0f, 10.0f, 10.0f};
    emergency_state.cyclist_counts = {5.0f, 5.0f, 5.0f, 5.0f};
    emergency_state.weather_condition = 0.0f;
    emergency_state.accident_indicator = 0.9f;  // High accident severity
    emergency_state.current_phase_duration = 30.0f;
    emergency_state.time_of_day = 0.5f;
    
    // Test override detection
    bool override_triggered = policy.checkEmergencyOverride(emergency_state);
    std::cout << "Emergency override triggered: " << (override_triggered ? "YES ✓" : "NO ✗") 
              << std::endl;
    
    // Test override action
    SignalPhase action = policy.selectAction(emergency_state, false);
    std::cout << "Emergency action: " << static_cast<int>(action) << std::endl;
    
    // Test with normal conditions
    emergency_state.accident_indicator = 0.0f;
    override_triggered = policy.checkEmergencyOverride(emergency_state);
    std::cout << "Override with normal conditions: " 
              << (override_triggered ? "YES ✗" : "NO ✓") << std::endl;
}

// Test peer voting system
void testPeerVoting(RLPolicyNet& policy) {
    std::cout << "\n=== Peer Voting System Test ===" << std::endl;
    
    // Simulate peer votes
    auto now = std::chrono::system_clock::now();
    
    PeerVoteMessage vote1{
        "intersection_1",
        SignalPhase::NS_GREEN_EW_RED,
        0.8f,
        "congestion",
        now
    };
    
    PeerVoteMessage vote2{
        "intersection_2",
        SignalPhase::NS_GREEN_EW_RED,
        0.9f,
        "emergency",
        now
    };
    
    PeerVoteMessage vote3{
        "intersection_3",
        SignalPhase::NS_RED_EW_GREEN,
        0.3f,
        "normal",
        now
    };
    
    // Add votes
    policy.receivePeerVote(vote1);
    policy.receivePeerVote(vote2);
    policy.receivePeerVote(vote3);
    
    // Create test state
    RLState state;
    state.queue_lengths = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    state.pedestrian_counts = {5.0f, 5.0f, 5.0f, 5.0f};
    state.cyclist_counts = {2.0f, 2.0f, 2.0f, 2.0f};
    state.weather_condition = 0.0f;
    state.accident_indicator = 0.0f;
    state.current_phase_duration = 20.0f;
    state.time_of_day = 0.5f;
    
    // Compute consensus
    SignalPhase consensus = policy.computeConsensusPhase(state);
    std::cout << "Consensus phase: " << static_cast<int>(consensus) << std::endl;
    std::cout << "Expected NS_GREEN (0): " 
              << (consensus == SignalPhase::NS_GREEN_EW_RED ? "YES ✓" : "NO ✗") 
              << std::endl;
}

// Test experience replay
void testExperienceReplay(RLPolicyNet& policy) {
    std::cout << "\n=== Experience Replay Test ===" << std::endl;
    
    // Create diverse experiences
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> reward_dist(-10.0f, 5.0f);
    
    for (int i = 0; i < 100; ++i) {
        RLState state;
        state.queue_lengths.resize(8);
        state.pedestrian_counts.resize(4);
        state.cyclist_counts.resize(4);
        
        // Random state
        for (auto& q : state.queue_lengths) q = i % 50;
        for (auto& p : state.pedestrian_counts) p = i % 20;
        for (auto& c : state.cyclist_counts) c = i % 10;
        state.weather_condition = i % 4;
        state.accident_indicator = 0.0f;
        state.current_phase_duration = i;
        state.time_of_day = (i % 100) / 100.0f;
        
        RLState next_state = state;
        next_state.queue_lengths[0] += 1.0f;
        
        Experience exp{
            state,
            i % 6,  // Random action
            reward_dist(rng),
            next_state,
            false
        };
        
        policy.addExperience(exp);
    }
    
    // Sample batch
    auto batch = policy.sampleBatch();
    std::cout << "Batch size: " << batch.size() << std::endl;
    std::cout << "Batch sampling works: " << (!batch.empty() ? "YES ✓" : "NO ✗") 
              << std::endl;
    
    // Check priority sampling diversity
    std::set<int> unique_actions;
    for (const auto& exp : batch) {
        unique_actions.insert(exp.action);
    }
    std::cout << "Action diversity in batch: " << unique_actions.size() << "/6" << std::endl;
}

// Test model persistence
void testModelPersistence(RLPolicyNet& policy) {
    std::cout << "\n=== Model Persistence Test ===" << std::endl;
    
    const std::string model_path = "test_rl_model.bin";
    
    // Save model
    try {
        policy.saveWeights(model_path);
        std::cout << "Model saved successfully ✓" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Save failed: " << e.what() << " ✗" << std::endl;
        return;
    }
    
    // Create new policy and load weights
    RLPolicyNet::Config config;
    config.state_dim = 32;
    config.hidden_dim = 128;
    config.num_actions = 6;
    
    RLPolicyNet loaded_policy(config);
    
    try {
        loaded_policy.loadWeights(model_path);
        std::cout << "Model loaded successfully ✓" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Load failed: " << e.what() << " ✗" << std::endl;
        return;
    }
    
    // Test that loaded model produces same output
    RLState test_state;
    test_state.queue_lengths = {10.0f, 20.0f, 30.0f, 40.0f, 5.0f, 15.0f, 25.0f, 35.0f};
    test_state.pedestrian_counts = {5.0f, 10.0f, 15.0f, 20.0f};
    test_state.cyclist_counts = {2.0f, 4.0f, 6.0f, 8.0f};
    test_state.weather_condition = 1.0f;
    test_state.accident_indicator = 0.0f;
    test_state.current_phase_duration = 45.0f;
    test_state.time_of_day = 0.75f;
    
    SignalPhase action1 = policy.selectAction(test_state, false);
    SignalPhase action2 = loaded_policy.selectAction(test_state, false);
    
    std::cout << "Actions match: " << (action1 == action2 ? "YES ✓" : "NO ✗") << std::endl;
    
    // Clean up
    std::remove(model_path.c_str());
}

// Test catastrophic forgetting prevention
void testElasticWeightConsolidation(RLPolicyNet& policy) {
    std::cout << "\n=== Elastic Weight Consolidation Test ===" << std::endl;
    
    // Enable EWC
    policy.enableElasticWeightConsolidation(true);
    std::cout << "EWC enabled ✓" << std::endl;
    
    // Initialize reference weights by enabling EWC with current weights
    policy.enableElasticWeightConsolidation(false);
    policy.enableElasticWeightConsolidation(true);
    
    // Generate important states for Fisher information
    std::vector<RLState> important_states;
    for (int i = 0; i < 50; ++i) {
        RLState state;
        state.queue_lengths = {20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f};
        state.pedestrian_counts = {10.0f, 10.0f, 10.0f, 10.0f};
        state.cyclist_counts = {5.0f, 5.0f, 5.0f, 5.0f};
        state.weather_condition = 0.0f;
        state.accident_indicator = 0.0f;
        state.current_phase_duration = 30.0f;
        state.time_of_day = 0.5f;
        important_states.push_back(state);
    }
    
    // Compute Fisher information
    policy.computeFisherInformation(important_states);
    std::cout << "Fisher information computed ✓" << std::endl;
    
    // Train on new task (different traffic pattern)
    std::vector<Experience> new_task_batch;
    for (int i = 0; i < 32; ++i) {
        RLState state;
        state.queue_lengths = {5.0f, 45.0f, 5.0f, 45.0f, 5.0f, 45.0f, 5.0f, 45.0f};
        state.pedestrian_counts = {2.0f, 18.0f, 2.0f, 18.0f};
        state.cyclist_counts = {1.0f, 9.0f, 1.0f, 9.0f};
        state.weather_condition = 2.0f;  // Fog
        state.accident_indicator = 0.0f;
        state.current_phase_duration = 60.0f;
        state.time_of_day = 0.9f;  // Evening
        
        Experience exp{state, i % 6, -5.0f, state, false};
        new_task_batch.push_back(exp);
    }
    
    // Compute loss with EWC
    auto loss = policy.computeLoss(new_task_batch);
    std::cout << "Loss with EWC penalty computed ✓" << std::endl;
    std::cout << "Total loss: " << loss.total_loss << std::endl;
}

// Main validation
int main() {
    std::cout << "PHASE 5 VALIDATION: Reinforcement Learning Engine" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    // Initialize RL policy
    RLPolicyNet::Config config;
    config.state_dim = 32;  // Based on state feature size
    config.hidden_dim = 128;
    config.num_actions = 6;
    config.learning_rate = 3e-4f;
    config.entropy_coef = 0.01f;
    config.max_decision_time_ms = 3.0f;
    
    RLPolicyNet policy(config);
    MockTrafficEnvironment env;
    
    // Run all tests
    benchmarkDecisionLatency(policy, env);
    testA2CTraining(policy, env);
    testEmergencyOverride(policy);
    testPeerVoting(policy);
    testExperienceReplay(policy);
    testModelPersistence(policy);
    testElasticWeightConsolidation(policy);
    
    // Final performance summary
    std::cout << "\n=== Performance Summary ===" << std::endl;
    auto stats = policy.getStats();
    std::cout << "Average decision time: " << stats.avg_decision_time_ms << " ms" << std::endl;
    std::cout << "Max decision time: " << stats.max_decision_time_ms << " ms" << std::endl;
    std::cout << "Total decisions made: " << stats.decisions_made << std::endl;
    
    bool latency_ok = stats.max_decision_time_ms <= 3.0f;
    std::cout << "\nMeets <3ms latency requirement: " 
              << (latency_ok ? "YES ✓" : "NO ✗") << std::endl;
    
    std::cout << "\nPHASE 5 VALIDATION: " 
              << (latency_ok ? "PASSED ✓" : "FAILED ✗") << std::endl;
    
    return latency_ok ? 0 : 1;
}