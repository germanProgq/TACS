/*
 * Federated Learning System for TACS
 * Distributed model training with V2X communication
 */

#ifndef FEDERATED_LEARNING_H
#define FEDERATED_LEARNING_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <functional>
#include <atomic>

#include "../include/core/tensor.h"
#include "../utils/config_manager.h"
#include "../include/core/memory_manager.h"

namespace TACS {

using tacs::core::Tensor;
using tacs::core::MemoryManager;

// Model snapshot with version control
struct ModelSnapshot {
    std::string version_hash;  // SHA256 hash of model parameters
    std::chrono::system_clock::time_point timestamp;
    std::unordered_map<std::string, Tensor> parameters;
    float performance_metrics;  // mAP or AUC for rollback decisions
    size_t training_samples;
    
    // Serialize to bytes for network transmission
    std::vector<uint8_t> serialize() const;
    static ModelSnapshot deserialize(const std::vector<uint8_t>& data);
};

// Federated learning configuration
struct FederatedConfig {
    // Network settings
    std::string server_address = "0.0.0.0";  // Configurable via config file
    int server_port = 8888;
    bool enable_encryption = true;
    
    // Learning parameters
    float aggregation_weight = 1.0f;  // Weight for this agent in global aggregation
    float distillation_temperature = 3.0f;  // Temperature for knowledge distillation
    float distillation_alpha = 0.7f;  // Balance between task loss and distillation
    
    // Performance monitoring
    float rollback_threshold = 0.05f;  // 5% performance drop triggers rollback
    size_t min_samples_for_update = 1000;  // Minimum samples before contributing
    
    // V2X settings
    bool enable_v2x = true;
    int v2x_channel = 178;  // DSRC channel for V2X
    
    // Load configuration from ConfigManager
    void loadFromConfig() {
        auto& config = tacs::utils::ConfigManager::getInstance();
        server_address = config.getString("network.server_address", server_address);
        server_port = config.getInt("network.server_port", server_port);
        enable_encryption = config.getBool("network.enable_encryption", enable_encryption);
        
        aggregation_weight = config.getFloat("federated.aggregation_weight", aggregation_weight);
        distillation_temperature = config.getFloat("federated.distillation_temperature", distillation_temperature);
        distillation_alpha = config.getFloat("federated.distillation_alpha", distillation_alpha);
        
        rollback_threshold = config.getFloat("federated.rollback_threshold", rollback_threshold);
        min_samples_for_update = config.getInt("federated.min_samples_for_update", min_samples_for_update);
        
        enable_v2x = config.getBool("v2x.enable", enable_v2x);
        v2x_channel = config.getInt("v2x.channel", v2x_channel);
    }
    std::string v2x_protocol = "DSRC";  // DSRC or C-V2X
    
    // Synchronization
    std::chrono::seconds sync_interval{300};  // 5 minutes
    size_t max_model_history = 10;  // Keep last 10 model snapshots
};

// V2X message types for federated learning
enum class FederatedMessageType : uint8_t {
    MODEL_UPDATE = 0x01,
    MODEL_REQUEST = 0x02,
    PERFORMANCE_REPORT = 0x03,
    ROLLBACK_SIGNAL = 0x04,
    DISTILLATION_REQUEST = 0x05,
    PEER_SYNC = 0x06,
    EMERGENCY_OVERRIDE = 0x07
};

// Main federated learning class
class FederatedLearning {
public:
    FederatedLearning(const FederatedConfig& config = FederatedConfig());
    ~FederatedLearning();
    
    // Initialize with model parameters
    void initialize(const std::unordered_map<std::string, Tensor>& initial_params);
    
    // Local training with parameter updates
    void updateLocalModel(const std::unordered_map<std::string, Tensor>& new_params,
                         float performance_metric,
                         size_t num_samples);
    
    // Server communication
    bool syncWithServer();
    bool requestGlobalModel();
    
    // Peer-to-peer synchronization
    void enablePeerSync(const std::vector<std::string>& peer_addresses);
    void broadcastUpdate();
    
    // Knowledge distillation
    std::unordered_map<std::string, Tensor> distillKnowledge(
        const std::unordered_map<std::string, Tensor>& teacher_params,
        const std::unordered_map<std::string, Tensor>& student_params);
    
    // Model versioning and rollback
    std::string computeModelHash(const std::unordered_map<std::string, Tensor>& params);
    bool rollbackToVersion(const std::string& version_hash);
    ModelSnapshot getCurrentSnapshot() const;
    
    // Performance monitoring
    void reportPerformance(float metric);
    bool shouldRollback() const;
    
    // V2X integration
    void handleV2XMessage(const std::vector<uint8_t>& message, const std::string& sender);
    void sendV2XBroadcast(FederatedMessageType type, const std::vector<uint8_t>& data);
    
    // Continual learning with EWC
    void updateFisherInformation(const std::unordered_map<std::string, Tensor>& grads);
    std::unordered_map<std::string, Tensor> applyEWCPenalty(
        const std::unordered_map<std::string, Tensor>& grads);
    
    // Get current model parameters
    std::unordered_map<std::string, Tensor> getModelParameters() const;
    
    // Emergency override for critical updates
    void applyEmergencyUpdate(const ModelSnapshot& snapshot);
    
private:
    FederatedConfig config_;
    
    // Model management
    std::unordered_map<std::string, Tensor> current_params_;
    std::vector<ModelSnapshot> model_history_;
    std::mutex params_mutex_;
    
    // Performance tracking
    std::vector<float> performance_history_;
    float baseline_performance_;
    std::atomic<bool> rollback_pending_{false};
    
    // Network communication
    int server_socket_;
    std::vector<int> peer_sockets_;
    std::thread sync_thread_;
    std::thread v2x_thread_;
    std::atomic<bool> running_{true};
    std::condition_variable sync_cv_;
    
    // EWC for catastrophic forgetting prevention
    std::unordered_map<std::string, Tensor> fisher_information_;
    std::unordered_map<std::string, Tensor> reference_params_;
    
    // V2X communication
    int v2x_socket_;
    std::mutex v2x_mutex_;
    
    // Helper methods
    void syncThreadWorker();
    void v2xThreadWorker();
    bool connectToServer();
    void disconnectFromServer();
    
    // Cryptography
    std::vector<uint8_t> encryptData(const std::vector<uint8_t>& data);
    std::vector<uint8_t> decryptData(const std::vector<uint8_t>& data);
    
    // Parameter aggregation
    std::unordered_map<std::string, Tensor> aggregateParameters(
        const std::vector<std::pair<ModelSnapshot, float>>& weighted_models);
    
    // Generate unique agent ID
    std::string generateAgentId();
    
public:
    // Compression for network efficiency (public for DistillationBank use)
    std::vector<uint8_t> compressParameters(const std::unordered_map<std::string, Tensor>& params);
    std::unordered_map<std::string, Tensor> decompressParameters(const std::vector<uint8_t>& data);
};

// Server-side distillation bank
class DistillationBank {
public:
    DistillationBank(int port = 8888);
    ~DistillationBank();
    
    // Start server
    void start();
    void stop();
    
    // Handle client connections
    void handleClient(int client_socket);
    
    // Model aggregation
    ModelSnapshot aggregateModels();
    
    // Distillation
    ModelSnapshot distillGlobalKnowledge();
    
private:
    int server_port_;
    int server_socket_;
    std::atomic<bool> running_{true};
    std::thread server_thread_;
    
    // Connected agents
    struct AgentInfo {
        std::string agent_id;
        std::string address;  // IP:port
        ModelSnapshot latest_model;
        float contribution_weight;
        std::chrono::system_clock::time_point last_update;
    };
    
    std::unordered_map<std::string, AgentInfo> agents_;
    std::mutex agents_mutex_;
    
    // Global model
    ModelSnapshot global_model_;
    std::mutex global_model_mutex_;
    
    void serverThreadWorker();
    void processModelUpdate(const std::string& agent_id, const ModelSnapshot& snapshot, 
                          const std::string& address = "");
    void broadcastGlobalModel();
};

} // namespace TACS

#endif // FEDERATED_LEARNING_H