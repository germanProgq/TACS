/*
 * Federated Learning Implementation
 * Production-ready distributed learning with V2X communication
 */

#include "../../include/federated/federated_learning.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <random>
#include "../../include/federated/sha256.h"
#include <zlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <fstream>
#include <sstream>
#ifdef __linux__
#include <ifaddrs.h>
#include <netpacket/packet.h>
#endif
#ifdef __APPLE__
#include <ifaddrs.h>
#include <net/if_dl.h>
#ifndef IFT_ETHER
#define IFT_ETHER 0x6  // Ethernet CSMA/CD
#endif
#endif

namespace TACS {

using tacs::core::Tensor;
using tacs::core::MemoryManager;

// ModelSnapshot implementation
std::vector<uint8_t> ModelSnapshot::serialize() const {
    std::vector<uint8_t> result;
    
    // Write header
    result.push_back(0xFE);  // Magic byte for federated learning
    result.push_back(0xD1);  // Version 1
    
    // Write version hash (32 bytes)
    result.insert(result.end(), version_hash.begin(), version_hash.end());
    
    // Write timestamp (8 bytes)
    auto time_since_epoch = timestamp.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch).count();
    for (int i = 7; i >= 0; --i) {
        result.push_back((seconds >> (i * 8)) & 0xFF);
    }
    
    // Write performance metrics (4 bytes)
    uint32_t perf_bits;
    memcpy(&perf_bits, &performance_metrics, sizeof(float));
    for (int i = 3; i >= 0; --i) {
        result.push_back((perf_bits >> (i * 8)) & 0xFF);
    }
    
    // Write training samples (8 bytes)
    for (int i = 7; i >= 0; --i) {
        result.push_back((training_samples >> (i * 8)) & 0xFF);
    }
    
    // Write number of parameters
    uint32_t num_params = parameters.size();
    for (int i = 3; i >= 0; --i) {
        result.push_back((num_params >> (i * 8)) & 0xFF);
    }
    
    // Write each parameter
    for (const auto& [name, tensor] : parameters) {
        // Write name length and name
        uint16_t name_len = name.length();
        result.push_back((name_len >> 8) & 0xFF);
        result.push_back(name_len & 0xFF);
        result.insert(result.end(), name.begin(), name.end());
        
        // Write tensor shape
        const auto& shape = tensor.shape();
        result.push_back(shape.size());
        for (int dim : shape) {
            for (int i = 3; i >= 0; --i) {
                result.push_back((dim >> (i * 8)) & 0xFF);
            }
        }
        
        // Write tensor data
        size_t data_size = tensor.size() * sizeof(float);
        const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(tensor.data());
        result.insert(result.end(), data_ptr, data_ptr + data_size);
    }
    
    return result;
}

ModelSnapshot ModelSnapshot::deserialize(const std::vector<uint8_t>& data) {
    ModelSnapshot snapshot;
    size_t offset = 0;
    
    // Check magic bytes
    if (data.size() < 2 || data[0] != 0xFE || data[1] != 0xD1) {
        throw std::runtime_error("Invalid federated learning snapshot format");
    }
    offset += 2;
    
    // Read version hash
    snapshot.version_hash = std::string(data.begin() + offset, data.begin() + offset + 32);
    offset += 32;
    
    // Read timestamp
    int64_t seconds = 0;
    for (int i = 0; i < 8; ++i) {
        seconds = (seconds << 8) | data[offset++];
    }
    snapshot.timestamp = std::chrono::system_clock::time_point(std::chrono::seconds(seconds));
    
    // Read performance metrics
    uint32_t perf_bits = 0;
    for (int i = 0; i < 4; ++i) {
        perf_bits = (perf_bits << 8) | data[offset++];
    }
    memcpy(&snapshot.performance_metrics, &perf_bits, sizeof(float));
    
    // Read training samples
    snapshot.training_samples = 0;
    for (int i = 0; i < 8; ++i) {
        snapshot.training_samples = (snapshot.training_samples << 8) | data[offset++];
    }
    
    // Read parameters
    uint32_t num_params = 0;
    for (int i = 0; i < 4; ++i) {
        num_params = (num_params << 8) | data[offset++];
    }
    
    for (uint32_t i = 0; i < num_params; ++i) {
        // Read name
        uint16_t name_len = (data[offset] << 8) | data[offset + 1];
        offset += 2;
        std::string name(data.begin() + offset, data.begin() + offset + name_len);
        offset += name_len;
        
        // Read tensor shape
        size_t num_dims = data[offset++];
        std::vector<int> shape;
        for (size_t j = 0; j < num_dims; ++j) {
            int dim = 0;
            for (int k = 0; k < 4; ++k) {
                dim = (dim << 8) | data[offset++];
            }
            shape.push_back(dim);
        }
        
        // Read tensor data
        size_t total_size = 1;
        for (size_t dim : shape) {
            total_size *= dim;
        }
        
        Tensor tensor(shape);
        size_t data_size = total_size * sizeof(float);
        memcpy(tensor.data(), data.data() + offset, data_size);
        offset += data_size;
        
        snapshot.parameters[name] = std::move(tensor);
    }
    
    return snapshot;
}

// FederatedLearning implementation
FederatedLearning::FederatedLearning(const FederatedConfig& config)
    : config_(config), baseline_performance_(0.0f) {
    // Memory manager initialized separately due to private destructor
    server_socket_ = -1;
    v2x_socket_ = -1;
}

FederatedLearning::~FederatedLearning() {
    running_ = false;
    sync_cv_.notify_all();
    
    if (sync_thread_.joinable()) {
        sync_thread_.join();
    }
    if (v2x_thread_.joinable()) {
        v2x_thread_.join();
    }
    
    disconnectFromServer();
    
    for (int socket : peer_sockets_) {
        close(socket);
    }
    
    if (v2x_socket_ >= 0) {
        close(v2x_socket_);
    }
}

void FederatedLearning::initialize(const std::unordered_map<std::string, Tensor>& initial_params) {
    std::lock_guard<std::mutex> lock(params_mutex_);
    current_params_ = initial_params;
    reference_params_ = initial_params;  // For EWC
    
    // Initialize Fisher information matrices
    for (const auto& [name, param] : initial_params) {
        fisher_information_[name] = Tensor(param.shape());
        fisher_information_[name].zero();
    }
    
    // Create initial snapshot
    ModelSnapshot initial_snapshot;
    initial_snapshot.version_hash = computeModelHash(initial_params);
    initial_snapshot.timestamp = std::chrono::system_clock::now();
    initial_snapshot.parameters = initial_params;
    initial_snapshot.performance_metrics = 0.0f;
    initial_snapshot.training_samples = 0;
    
    model_history_.push_back(initial_snapshot);
    
    // Start background threads
    sync_thread_ = std::thread(&FederatedLearning::syncThreadWorker, this);
    
    if (config_.enable_v2x) {
        v2x_thread_ = std::thread(&FederatedLearning::v2xThreadWorker, this);
    }
}

void FederatedLearning::updateLocalModel(const std::unordered_map<std::string, Tensor>& new_params,
                                       float performance_metric,
                                       size_t num_samples) {
    std::lock_guard<std::mutex> lock(params_mutex_);
    
    // Update parameters
    current_params_ = new_params;
    
    // Create new snapshot
    ModelSnapshot snapshot;
    snapshot.version_hash = computeModelHash(new_params);
    snapshot.timestamp = std::chrono::system_clock::now();
    snapshot.parameters = new_params;
    snapshot.performance_metrics = performance_metric;
    snapshot.training_samples = num_samples;
    
    // Add to history
    model_history_.push_back(snapshot);
    
    // Maintain history size
    if (model_history_.size() > config_.max_model_history) {
        model_history_.erase(model_history_.begin());
    }
    
    // Update performance tracking
    performance_history_.push_back(performance_metric);
    
    // Check for performance degradation
    if (performance_history_.size() > 1) {
        size_t window_size = std::min(size_t(5), performance_history_.size());
        float avg_recent = std::accumulate(performance_history_.end() - window_size,
                                         performance_history_.end(), 0.0f) / window_size;
        
        float degradation = (baseline_performance_ - avg_recent) / baseline_performance_;
        if (baseline_performance_ > 0 && degradation > config_.rollback_threshold) {
            rollback_pending_ = true;
            std::cout << "Performance degradation detected: " << degradation * 100 << "%. Rollback pending." << std::endl;
        }
    }
    
    // Trigger sync
    sync_cv_.notify_one();
}

bool FederatedLearning::syncWithServer() {
    if (!connectToServer()) {
        return false;
    }
    
    try {
        // Prepare model update message
        ModelSnapshot snapshot = getCurrentSnapshot();
        auto serialized = snapshot.serialize();
        
        // Add message header
        std::vector<uint8_t> message;
        message.push_back(static_cast<uint8_t>(FederatedMessageType::MODEL_UPDATE));
        
        // Generate unique agent ID from MAC address and process ID
        std::string agent_id = generateAgentId();
        message.insert(message.end(), agent_id.begin(), agent_id.end());
        
        // Add compressed model data
        auto compressed = compressParameters(snapshot.parameters);
        uint32_t compressed_size = compressed.size();
        for (int i = 3; i >= 0; --i) {
            message.push_back((compressed_size >> (i * 8)) & 0xFF);
        }
        
        // Add performance metrics
        uint32_t perf_bits;
        memcpy(&perf_bits, &snapshot.performance_metrics, sizeof(float));
        for (int i = 3; i >= 0; --i) {
            message.push_back((perf_bits >> (i * 8)) & 0xFF);
        }
        
        // Add training samples
        for (int i = 7; i >= 0; --i) {
            message.push_back((snapshot.training_samples >> (i * 8)) & 0xFF);
        }
        
        message.insert(message.end(), compressed.begin(), compressed.end());
        
        // Encrypt if enabled
        if (config_.enable_encryption) {
            message = encryptData(message);
        }
        
        // Send to server
        ssize_t sent = send(server_socket_, message.data(), message.size(), 0);
        if (sent < 0) {
            std::cerr << "Failed to send model update to server" << std::endl;
            return false;
        }
        
        // Wait for response
        std::vector<uint8_t> response(65536);
        ssize_t received = recv(server_socket_, response.data(), response.size(), 0);
        if (received > 0) {
            response.resize(received);
            
            // Decrypt if needed
            if (config_.enable_encryption) {
                response = decryptData(response);
            }
            
            // Process response
            if (response[0] == static_cast<uint8_t>(FederatedMessageType::MODEL_UPDATE)) {
                // Extract global model
                std::vector<uint8_t> model_data(response.begin() + 1, response.end());
                ModelSnapshot global_model = ModelSnapshot::deserialize(model_data);
                
                // Apply distillation if configured
                if (config_.distillation_alpha > 0) {
                    auto distilled = distillKnowledge(global_model.parameters, current_params_);
                    updateLocalModel(distilled, snapshot.performance_metrics, snapshot.training_samples);
                }
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during server sync: " << e.what() << std::endl;
        return false;
    }
}

void FederatedLearning::enablePeerSync(const std::vector<std::string>& peer_addresses) {
    for (const auto& addr : peer_addresses) {
        // Parse address (format: "ip:port")
        size_t colon_pos = addr.find(':');
        if (colon_pos == std::string::npos) continue;
        
        std::string ip = addr.substr(0, colon_pos);
        int port = std::stoi(addr.substr(colon_pos + 1));
        
        // Create socket
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) continue;
        
        // Set non-blocking
        int flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, flags | O_NONBLOCK);
        
        // Connect to peer
        struct sockaddr_in peer_addr;
        peer_addr.sin_family = AF_INET;
        peer_addr.sin_port = htons(port);
        inet_pton(AF_INET, ip.c_str(), &peer_addr.sin_addr);
        
        connect(sock, (struct sockaddr*)&peer_addr, sizeof(peer_addr));
        peer_sockets_.push_back(sock);
    }
}

std::unordered_map<std::string, Tensor> FederatedLearning::distillKnowledge(
    const std::unordered_map<std::string, Tensor>& teacher_params,
    const std::unordered_map<std::string, Tensor>& student_params) {
    
    std::unordered_map<std::string, Tensor> distilled_params;
    
    for (const auto& [name, teacher_tensor] : teacher_params) {
        auto student_it = student_params.find(name);
        if (student_it == student_params.end()) {
            distilled_params[name] = teacher_tensor;
            continue;
        }
        
        const Tensor& student_tensor = student_it->second;
        
        // Create distilled tensor
        Tensor distilled(teacher_tensor.shape());
        
        // Apply temperature-scaled softmax and combine
        float T = config_.distillation_temperature;
        float alpha = config_.distillation_alpha;
        
        const float* teacher_data = teacher_tensor.data_float();
        const float* student_data = student_tensor.data_float();
        float* distilled_data = distilled.data_float();
        
        for (size_t i = 0; i < teacher_tensor.size(); ++i) {
            float teacher_val = teacher_data[i] / T;
            float student_val = student_data[i] / T;
            
            // Weighted combination
            distilled_data[i] = alpha * teacher_val + (1 - alpha) * student_val;
        }
        
        distilled_params[name] = std::move(distilled);
    }
    
    return distilled_params;
}

std::string FederatedLearning::computeModelHash(const std::unordered_map<std::string, Tensor>& params) {
    SHA256 sha256;
    
    // Sort parameters by name for consistent hashing
    std::vector<std::string> names;
    for (const auto& [name, _] : params) {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    
    // Hash each parameter
    for (const auto& name : names) {
        const Tensor& tensor = params.at(name);
        
        // Hash name
        sha256.update(name.c_str(), name.length());
        
        // Hash shape
        const auto& shape = tensor.shape();
        for (int dim : shape) {
            sha256.update(&dim, sizeof(int));
        }
        
        // Hash data
        sha256.update(tensor.data(), tensor.size() * sizeof(float));
    }
    
    // Get final hash
    return sha256.final();
}

bool FederatedLearning::rollbackToVersion(const std::string& version_hash) {
    std::lock_guard<std::mutex> lock(params_mutex_);
    
    // Find snapshot with matching hash
    for (const auto& snapshot : model_history_) {
        if (snapshot.version_hash == version_hash) {
            current_params_ = snapshot.parameters;
            rollback_pending_ = false;
            
            // Reset performance baseline
            baseline_performance_ = snapshot.performance_metrics;
            
            std::cout << "Rolled back to version: " << version_hash.substr(0, 8) << std::endl;
            return true;
        }
    }
    
    return false;
}

void FederatedLearning::updateFisherInformation(const std::unordered_map<std::string, Tensor>& grads) {
    std::lock_guard<std::mutex> lock(params_mutex_);
    
    float decay = 0.99f;  // Exponential moving average
    
    for (const auto& [name, grad] : grads) {
        auto fisher_it = fisher_information_.find(name);
        if (fisher_it == fisher_information_.end()) continue;
        
        Tensor& fisher = fisher_it->second;
        
        // Update Fisher information: F = decay * F + (1-decay) * grad^2
        const float* grad_data = grad.data_float();
        float* fisher_data = fisher.data_float();
        
        for (size_t i = 0; i < grad.size(); ++i) {
            float g = grad_data[i];
            fisher_data[i] = decay * fisher_data[i] + (1 - decay) * g * g;
        }
    }
}

std::unordered_map<std::string, Tensor> FederatedLearning::applyEWCPenalty(
    const std::unordered_map<std::string, Tensor>& grads) {
    
    std::lock_guard<std::mutex> lock(params_mutex_);
    std::unordered_map<std::string, Tensor> penalized_grads;
    
    float ewc_lambda = 1000.0f;  // EWC strength
    
    for (const auto& [name, grad] : grads) {
        auto param_it = current_params_.find(name);
        auto ref_it = reference_params_.find(name);
        auto fisher_it = fisher_information_.find(name);
        
        if (param_it == current_params_.end() || 
            ref_it == reference_params_.end() ||
            fisher_it == fisher_information_.end()) {
            penalized_grads[name] = grad;
            continue;
        }
        
        Tensor penalized(grad.shape());
        const Tensor& param = param_it->second;
        const Tensor& ref = ref_it->second;
        const Tensor& fisher = fisher_it->second;
        
        const float* grad_data = grad.data_float();
        const float* param_data = param.data_float();
        const float* ref_data = ref.data_float();
        const float* fisher_data = fisher.data_float();
        float* penalized_data = penalized.data_float();
        
        // Add EWC penalty: grad + lambda * F * (theta - theta_ref)
        for (size_t i = 0; i < grad.size(); ++i) {
            float penalty = ewc_lambda * fisher_data[i] * (param_data[i] - ref_data[i]);
            penalized_data[i] = grad_data[i] + penalty;
        }
        
        penalized_grads[name] = std::move(penalized);
    }
    
    return penalized_grads;
}

std::string FederatedLearning::generateAgentId() {
    // Generate unique ID from MAC address and process ID
    std::stringstream ss;
    
    // Get MAC address
    std::string mac_addr = "000000000000";  // Default
    
#ifdef __linux__
    struct ifaddrs *ifap, *ifa;
    if (getifaddrs(&ifap) == 0) {
        for (ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_PACKET) {
                struct sockaddr_ll *s = (struct sockaddr_ll*)ifa->ifa_addr;
                if (s->sll_hatype == 1) {  // Ethernet
                    char mac[18];
                    snprintf(mac, sizeof(mac), "%02x%02x%02x%02x%02x%02x",
                            s->sll_addr[0], s->sll_addr[1], s->sll_addr[2],
                            s->sll_addr[3], s->sll_addr[4], s->sll_addr[5]);
                    mac_addr = mac;
                    break;
                }
            }
        }
        freeifaddrs(ifap);
    }
#elif defined(__APPLE__)
    struct ifaddrs *ifap, *ifa;
    if (getifaddrs(&ifap) == 0) {
        for (ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_LINK) {
                struct sockaddr_dl *sdl = (struct sockaddr_dl*)ifa->ifa_addr;
                if (sdl->sdl_type == IFT_ETHER) {
                    unsigned char *ptr = (unsigned char *)LLADDR(sdl);
                    char mac[18];
                    snprintf(mac, sizeof(mac), "%02x%02x%02x%02x%02x%02x",
                            ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5]);
                    mac_addr = mac;
                    break;
                }
            }
        }
        freeifaddrs(ifap);
    }
#endif
    
    // Combine with process ID for uniqueness
    ss << mac_addr << "_" << std::hex << getpid();
    
    // Take first 16 chars for agent ID
    std::string agent_id = ss.str();
    if (agent_id.length() > 16) {
        agent_id = agent_id.substr(0, 16);
    }
    
    return agent_id;
}

void FederatedLearning::handleV2XMessage(const std::vector<uint8_t>& message, const std::string& sender) {
    if (message.empty()) return;
    
    FederatedMessageType type = static_cast<FederatedMessageType>(message[0]);
    
    switch (type) {
        case FederatedMessageType::MODEL_UPDATE: {
            // Peer is sharing their model update
            try {
                std::vector<uint8_t> model_data(message.begin() + 1, message.end());
                ModelSnapshot peer_model = ModelSnapshot::deserialize(model_data);
                
                // Apply distillation if performance is better
                if (peer_model.performance_metrics > getCurrentSnapshot().performance_metrics) {
                    auto distilled = distillKnowledge(peer_model.parameters, current_params_);
                    updateLocalModel(distilled, getCurrentSnapshot().performance_metrics, 
                                   getCurrentSnapshot().training_samples);
                }
            } catch (const std::exception& e) {
                std::cerr << "Failed to process peer model update: " << e.what() << std::endl;
            }
            break;
        }
        
        case FederatedMessageType::EMERGENCY_OVERRIDE: {
            // Critical update that must be applied immediately
            try {
                std::vector<uint8_t> model_data(message.begin() + 1, message.end());
                ModelSnapshot emergency_model = ModelSnapshot::deserialize(model_data);
                applyEmergencyUpdate(emergency_model);
            } catch (const std::exception& e) {
                std::cerr << "Failed to apply emergency update: " << e.what() << std::endl;
            }
            break;
        }
        
        case FederatedMessageType::PERFORMANCE_REPORT: {
            // Peer reporting their performance metrics
            if (message.size() >= 5) {
                float peer_performance;
                memcpy(&peer_performance, message.data() + 1, sizeof(float));
                std::cout << "Peer " << sender << " reports performance: " << peer_performance << std::endl;
            }
            break;
        }
        
        default:
            break;
    }
}

void FederatedLearning::sendV2XBroadcast(FederatedMessageType type, const std::vector<uint8_t>& data) {
    std::lock_guard<std::mutex> lock(v2x_mutex_);
    
    if (v2x_socket_ < 0) return;
    
    // Prepare V2X message
    std::vector<uint8_t> message;
    message.push_back(static_cast<uint8_t>(type));
    message.insert(message.end(), data.begin(), data.end());
    
    // Broadcast to all peers
    struct sockaddr_in broadcast_addr;
    broadcast_addr.sin_family = AF_INET;
    broadcast_addr.sin_port = htons(config_.server_port + 100);  // V2X port offset
    broadcast_addr.sin_addr.s_addr = htonl(INADDR_BROADCAST);
    
    sendto(v2x_socket_, message.data(), message.size(), 0,
           (struct sockaddr*)&broadcast_addr, sizeof(broadcast_addr));
}

void FederatedLearning::broadcastUpdate() {
    // Send current model to all peers
    ModelSnapshot snapshot = getCurrentSnapshot();
    auto data = snapshot.serialize();
    sendV2XBroadcast(FederatedMessageType::MODEL_UPDATE, data);
}

void FederatedLearning::reportPerformance(float metric) {
    performance_history_.push_back(metric);
    
    // Set baseline if first report
    if (performance_history_.size() == 1) {
        baseline_performance_ = metric;
    }
    
    // Check for performance degradation immediately
    if (performance_history_.size() > 1 && baseline_performance_ > 0) {
        float degradation = (baseline_performance_ - metric) / baseline_performance_;
        if (degradation > config_.rollback_threshold) {
            rollback_pending_ = true;
            std::cout << "Performance degradation in report: " << degradation * 100 << "%. Rollback pending." << std::endl;
        }
    }
}

bool FederatedLearning::shouldRollback() const {
    return rollback_pending_;
}

std::unordered_map<std::string, Tensor> FederatedLearning::getModelParameters() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(params_mutex_));
    return current_params_;
}

ModelSnapshot FederatedLearning::getCurrentSnapshot() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(params_mutex_));
    
    if (!model_history_.empty()) {
        return model_history_.back();
    }
    
    // Create snapshot from current parameters
    ModelSnapshot snapshot;
    snapshot.version_hash = const_cast<FederatedLearning*>(this)->computeModelHash(current_params_);
    snapshot.timestamp = std::chrono::system_clock::now();
    snapshot.parameters = current_params_;
    snapshot.performance_metrics = performance_history_.empty() ? 0.0f : performance_history_.back();
    snapshot.training_samples = 0;
    
    return snapshot;
}

void FederatedLearning::applyEmergencyUpdate(const ModelSnapshot& snapshot) {
    std::lock_guard<std::mutex> lock(params_mutex_);
    
    current_params_ = snapshot.parameters;
    model_history_.push_back(snapshot);
    
    // Maintain history size
    if (model_history_.size() > config_.max_model_history) {
        model_history_.erase(model_history_.begin());
    }
    
    std::cout << "Emergency update applied: " << snapshot.version_hash.substr(0, 8) << std::endl;
}

// Private helper methods
void FederatedLearning::syncThreadWorker() {
    while (running_) {
        std::unique_lock<std::mutex> lock(params_mutex_);
        sync_cv_.wait_for(lock, config_.sync_interval, [this] { return !running_; });
        
        if (!running_) break;
        
        // Check if we should sync
        if (getCurrentSnapshot().training_samples >= config_.min_samples_for_update) {
            lock.unlock();
            syncWithServer();
        }
    }
}

void FederatedLearning::v2xThreadWorker() {
    // Create V2X socket
    v2x_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (v2x_socket_ < 0) {
        std::cerr << "Failed to create V2X socket" << std::endl;
        return;
    }
    
    // Enable broadcast
    int broadcast_enable = 1;
    setsockopt(v2x_socket_, SOL_SOCKET, SO_BROADCAST, &broadcast_enable, sizeof(broadcast_enable));
    
    // Bind to V2X port
    struct sockaddr_in v2x_addr;
    v2x_addr.sin_family = AF_INET;
    v2x_addr.sin_port = htons(config_.server_port + 100);
    v2x_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    
    if (bind(v2x_socket_, (struct sockaddr*)&v2x_addr, sizeof(v2x_addr)) < 0) {
        std::cerr << "Failed to bind V2X socket" << std::endl;
        close(v2x_socket_);
        v2x_socket_ = -1;
        return;
    }
    
    // Set non-blocking
    int flags = fcntl(v2x_socket_, F_GETFL, 0);
    fcntl(v2x_socket_, F_SETFL, flags | O_NONBLOCK);
    
    // Listen for V2X messages
    std::vector<uint8_t> buffer(65536);
    struct sockaddr_in sender_addr;
    socklen_t sender_len = sizeof(sender_addr);
    
    while (running_) {
        ssize_t received = recvfrom(v2x_socket_, buffer.data(), buffer.size(), 0,
                                  (struct sockaddr*)&sender_addr, &sender_len);
        
        if (received > 0) {
            std::vector<uint8_t> message(buffer.begin(), buffer.begin() + received);
            std::string sender_ip = inet_ntoa(sender_addr.sin_addr);
            
            // Decrypt if needed
            if (config_.enable_encryption) {
                try {
                    message = decryptData(message);
                } catch (const std::exception& e) {
                    continue;  // Skip invalid messages
                }
            }
            
            handleV2XMessage(message, sender_ip);
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

bool FederatedLearning::connectToServer() {
    if (server_socket_ >= 0) {
        return true;  // Already connected
    }
    
    server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket_ < 0) {
        return false;
    }
    
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(config_.server_port);
    inet_pton(AF_INET, config_.server_address.c_str(), &server_addr.sin_addr);
    
    if (connect(server_socket_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(server_socket_);
        server_socket_ = -1;
        return false;
    }
    
    return true;
}

void FederatedLearning::disconnectFromServer() {
    if (server_socket_ >= 0) {
        close(server_socket_);
        server_socket_ = -1;
    }
}

std::vector<uint8_t> FederatedLearning::compressParameters(const std::unordered_map<std::string, Tensor>& params) {
    // Serialize parameters
    std::vector<uint8_t> uncompressed;
    
    // Write number of parameters
    uint32_t num_params = params.size();
    for (int i = 3; i >= 0; --i) {
        uncompressed.push_back((num_params >> (i * 8)) & 0xFF);
    }
    
    // Write each parameter
    for (const auto& [name, tensor] : params) {
        // Name length and name
        uint16_t name_len = name.length();
        uncompressed.push_back((name_len >> 8) & 0xFF);
        uncompressed.push_back(name_len & 0xFF);
        uncompressed.insert(uncompressed.end(), name.begin(), name.end());
        
        // Tensor data
        const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(tensor.data());
        size_t data_size = tensor.size() * sizeof(float);
        uncompressed.insert(uncompressed.end(), data_ptr, data_ptr + data_size);
    }
    
    // Compress using zlib
    uLongf compressed_size = compressBound(uncompressed.size());
    std::vector<uint8_t> compressed(compressed_size);
    
    if (compress2(compressed.data(), &compressed_size, uncompressed.data(), 
                  uncompressed.size(), Z_BEST_COMPRESSION) != Z_OK) {
        return uncompressed;  // Return uncompressed on failure
    }
    
    compressed.resize(compressed_size);
    return compressed;
}

std::unordered_map<std::string, Tensor> FederatedLearning::decompressParameters(const std::vector<uint8_t>& data) {
    // Decompress using zlib
    std::vector<uint8_t> uncompressed;
    uLongf uncompressed_size = data.size() * 10;  // Estimate
    uncompressed.resize(uncompressed_size);
    
    int result = uncompress(uncompressed.data(), &uncompressed_size, 
                           data.data(), data.size());
    
    if (result != Z_OK) {
        throw std::runtime_error("Failed to decompress parameters");
    }
    
    uncompressed.resize(uncompressed_size);
    
    // Deserialize parameters
    std::unordered_map<std::string, Tensor> params;
    size_t offset = 0;
    
    // Read number of parameters
    uint32_t num_params = 0;
    for (int i = 0; i < 4; ++i) {
        num_params = (num_params << 8) | uncompressed[offset++];
    }
    
    // Read each parameter
    for (uint32_t i = 0; i < num_params; ++i) {
        // Name length and name
        uint16_t name_len = (uncompressed[offset] << 8) | uncompressed[offset + 1];
        offset += 2;
        std::string name(uncompressed.begin() + offset, 
                        uncompressed.begin() + offset + name_len);
        offset += name_len;
        
        // Read tensor shape
        size_t num_dims = uncompressed[offset++];
        std::vector<int> shape;
        for (size_t j = 0; j < num_dims; ++j) {
            int dim = 0;
            for (int k = 0; k < 4; ++k) {
                dim = (dim << 8) | uncompressed[offset++];
            }
            shape.push_back(dim);
        }
        
        // Read tensor data
        Tensor tensor(shape);
        size_t data_size = tensor.size() * sizeof(float);
        memcpy(tensor.data(), uncompressed.data() + offset, data_size);
        offset += data_size;
        
        params[name] = std::move(tensor);
    }
    
    return params;
}

std::vector<uint8_t> FederatedLearning::encryptData(const std::vector<uint8_t>& data) {
    // ChaCha20 stream cipher implementation
    std::vector<uint8_t> encrypted(data.size() + 12);  // +12 for nonce
    
    // Generate random nonce
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    
    uint8_t nonce[12];
    for (int i = 0; i < 12; ++i) {
        nonce[i] = dist(gen);
        encrypted[i] = nonce[i];
    }
    
    // ChaCha20 key schedule and encryption
    uint32_t key[8] = {
        0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
        0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c
    };
    
    uint32_t state[16];
    uint32_t working_state[16];
    uint64_t counter = 0;
    
    for (size_t pos = 0; pos < data.size(); pos += 64) {
        // Initialize state
        state[0] = 0x61707865; state[1] = 0x3320646e;
        state[2] = 0x79622d32; state[3] = 0x6b206574;
        
        // Copy key
        for (int i = 0; i < 8; ++i) {
            state[4 + i] = key[i];
        }
        
        // Counter and nonce
        state[12] = counter & 0xFFFFFFFF;
        state[13] = (counter >> 32) & 0xFFFFFFFF;
        state[14] = ((uint32_t)nonce[0]) | ((uint32_t)nonce[1] << 8) | 
                    ((uint32_t)nonce[2] << 16) | ((uint32_t)nonce[3] << 24);
        state[15] = ((uint32_t)nonce[4]) | ((uint32_t)nonce[5] << 8) | 
                    ((uint32_t)nonce[6] << 16) | ((uint32_t)nonce[7] << 24);
        
        // Copy state to working state
        for (int i = 0; i < 16; ++i) {
            working_state[i] = state[i];
        }
        
        // ChaCha20 quarter round
        auto quarter_round = [](uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
            a += b; d ^= a; d = (d << 16) | (d >> 16);
            c += d; b ^= c; b = (b << 12) | (b >> 20);
            a += b; d ^= a; d = (d << 8) | (d >> 24);
            c += d; b ^= c; b = (b << 7) | (b >> 25);
        };
        
        // 20 rounds (10 double rounds)
        for (int i = 0; i < 10; ++i) {
            // Column rounds
            quarter_round(working_state[0], working_state[4], working_state[8], working_state[12]);
            quarter_round(working_state[1], working_state[5], working_state[9], working_state[13]);
            quarter_round(working_state[2], working_state[6], working_state[10], working_state[14]);
            quarter_round(working_state[3], working_state[7], working_state[11], working_state[15]);
            // Diagonal rounds
            quarter_round(working_state[0], working_state[5], working_state[10], working_state[15]);
            quarter_round(working_state[1], working_state[6], working_state[11], working_state[12]);
            quarter_round(working_state[2], working_state[7], working_state[8], working_state[13]);
            quarter_round(working_state[3], working_state[4], working_state[9], working_state[14]);
        }
        
        // Add original state
        for (int i = 0; i < 16; ++i) {
            working_state[i] += state[i];
        }
        
        // XOR with data
        size_t block_size = std::min<size_t>(64, data.size() - pos);
        for (size_t i = 0; i < block_size; ++i) {
            encrypted[12 + pos + i] = data[pos + i] ^ ((uint8_t*)working_state)[i];
        }
        
        counter++;
    }
    
    return encrypted;
}

std::vector<uint8_t> FederatedLearning::decryptData(const std::vector<uint8_t>& data) {
    if (data.size() < 12) {
        throw std::runtime_error("Invalid encrypted data: too short for nonce");
    }
    
    // Extract nonce
    uint8_t nonce[12];
    for (int i = 0; i < 12; ++i) {
        nonce[i] = data[i];
    }
    
    std::vector<uint8_t> decrypted(data.size() - 12);
    
    // ChaCha20 key schedule and decryption (same as encryption)
    uint32_t key[8] = {
        0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
        0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c
    };
    
    uint32_t state[16];
    uint32_t working_state[16];
    uint64_t counter = 0;
    
    for (size_t pos = 0; pos < decrypted.size(); pos += 64) {
        // Initialize state
        state[0] = 0x61707865; state[1] = 0x3320646e;
        state[2] = 0x79622d32; state[3] = 0x6b206574;
        
        // Copy key
        for (int i = 0; i < 8; ++i) {
            state[4 + i] = key[i];
        }
        
        // Counter and nonce
        state[12] = counter & 0xFFFFFFFF;
        state[13] = (counter >> 32) & 0xFFFFFFFF;
        state[14] = ((uint32_t)nonce[0]) | ((uint32_t)nonce[1] << 8) | 
                    ((uint32_t)nonce[2] << 16) | ((uint32_t)nonce[3] << 24);
        state[15] = ((uint32_t)nonce[4]) | ((uint32_t)nonce[5] << 8) | 
                    ((uint32_t)nonce[6] << 16) | ((uint32_t)nonce[7] << 24);
        
        // Copy state to working state
        for (int i = 0; i < 16; ++i) {
            working_state[i] = state[i];
        }
        
        // ChaCha20 quarter round
        auto quarter_round = [](uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
            a += b; d ^= a; d = (d << 16) | (d >> 16);
            c += d; b ^= c; b = (b << 12) | (b >> 20);
            a += b; d ^= a; d = (d << 8) | (d >> 24);
            c += d; b ^= c; b = (b << 7) | (b >> 25);
        };
        
        // 20 rounds (10 double rounds)
        for (int i = 0; i < 10; ++i) {
            // Column rounds
            quarter_round(working_state[0], working_state[4], working_state[8], working_state[12]);
            quarter_round(working_state[1], working_state[5], working_state[9], working_state[13]);
            quarter_round(working_state[2], working_state[6], working_state[10], working_state[14]);
            quarter_round(working_state[3], working_state[7], working_state[11], working_state[15]);
            // Diagonal rounds
            quarter_round(working_state[0], working_state[5], working_state[10], working_state[15]);
            quarter_round(working_state[1], working_state[6], working_state[11], working_state[12]);
            quarter_round(working_state[2], working_state[7], working_state[8], working_state[13]);
            quarter_round(working_state[3], working_state[4], working_state[9], working_state[14]);
        }
        
        // Add original state
        for (int i = 0; i < 16; ++i) {
            working_state[i] += state[i];
        }
        
        // XOR with data
        size_t block_size = std::min<size_t>(64, decrypted.size() - pos);
        for (size_t i = 0; i < block_size; ++i) {
            decrypted[pos + i] = data[12 + pos + i] ^ ((uint8_t*)working_state)[i];
        }
        
        counter++;
    }
    
    return decrypted;
}

std::unordered_map<std::string, Tensor> FederatedLearning::aggregateParameters(
    const std::vector<std::pair<ModelSnapshot, float>>& weighted_models) {
    
    std::unordered_map<std::string, Tensor> aggregated;
    
    if (weighted_models.empty()) {
        return aggregated;
    }
    
    // Calculate total weight
    float total_weight = 0.0f;
    for (const auto& [model, weight] : weighted_models) {
        total_weight += weight;
    }
    
    if (total_weight <= 0.0f) {
        return aggregated;
    }
    
    // Initialize aggregated parameters with zeros
    const auto& first_params = weighted_models[0].first.parameters;
    for (const auto& [name, tensor] : first_params) {
        aggregated[name] = Tensor(tensor.shape());
        aggregated[name].zero();
    }
    
    // Weighted average of all model parameters
    for (const auto& [model, weight] : weighted_models) {
        float normalized_weight = weight / total_weight;
        
        for (const auto& [name, tensor] : model.parameters) {
            auto it = aggregated.find(name);
            if (it != aggregated.end()) {
                Tensor& agg_tensor = it->second;
                const float* src_data = tensor.data_float();
                float* dst_data = agg_tensor.data_float();
                
                // Add weighted contribution
                for (size_t i = 0; i < tensor.size(); ++i) {
                    dst_data[i] += normalized_weight * src_data[i];
                }
            }
        }
    }
    
    return aggregated;
}

// DistillationBank implementation
DistillationBank::DistillationBank(int port) : server_port_(port) {
    server_socket_ = -1;
}

DistillationBank::~DistillationBank() {
    stop();
}

void DistillationBank::start() {
    // Create server socket
    server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket_ < 0) {
        throw std::runtime_error("Failed to create server socket");
    }
    
    // Allow reuse
    int opt = 1;
    setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    // Bind
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(server_port_);
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    
    if (bind(server_socket_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(server_socket_);
        throw std::runtime_error("Failed to bind server socket");
    }
    
    // Listen
    if (listen(server_socket_, 10) < 0) {
        close(server_socket_);
        throw std::runtime_error("Failed to listen on server socket");
    }
    
    // Start server thread
    running_ = true;
    server_thread_ = std::thread(&DistillationBank::serverThreadWorker, this);
}

void DistillationBank::stop() {
    running_ = false;
    
    if (server_socket_ >= 0) {
        close(server_socket_);
        server_socket_ = -1;
    }
    
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

void DistillationBank::serverThreadWorker() {
    while (running_) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_socket = accept(server_socket_, (struct sockaddr*)&client_addr, &client_len);
        if (client_socket < 0) {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                std::cerr << "Failed to accept client connection" << std::endl;
            }
            continue;
        }
        
        // Handle client in separate thread
        std::thread client_thread(&DistillationBank::handleClient, this, client_socket);
        client_thread.detach();
    }
}

void DistillationBank::handleClient(int client_socket) {
    std::vector<uint8_t> buffer(1048576);  // 1MB buffer
    
    // Get client address
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);
    getpeername(client_socket, (struct sockaddr*)&client_addr, &addr_len);
    std::string client_address = std::string(inet_ntoa(client_addr.sin_addr)) + ":" + 
                                std::to_string(ntohs(client_addr.sin_port));
    
    ssize_t received = recv(client_socket, buffer.data(), buffer.size(), 0);
    if (received <= 0) {
        close(client_socket);
        return;
    }
    
    buffer.resize(received);
    
    // Process message
    if (buffer[0] == static_cast<uint8_t>(FederatedMessageType::MODEL_UPDATE)) {
        // Extract agent ID
        std::string agent_id(buffer.begin() + 1, buffer.begin() + 9);
        
        // Extract compressed size
        uint32_t compressed_size = 0;
        for (int i = 0; i < 4; ++i) {
            compressed_size = (compressed_size << 8) | buffer[9 + i];
        }
        
        // Extract performance metrics and training samples from message
        float performance_metrics = 0.0f;
        size_t training_samples = 0;
        
        if (buffer.size() >= 25) {  // Ensure we have enough data
            // Extract performance metrics (4 bytes after compressed size)
            uint32_t perf_bits = 0;
            for (int i = 0; i < 4; ++i) {
                perf_bits = (perf_bits << 8) | buffer[13 + i];
            }
            memcpy(&performance_metrics, &perf_bits, sizeof(float));
            
            // Extract training samples (8 bytes after performance)
            for (int i = 0; i < 8; ++i) {
                training_samples = (training_samples << 8) | buffer[21 + i];
            }
        }
        
        // Extract model data (after performance and samples)
        size_t model_data_offset = 13;
        if (buffer.size() >= 25) {
            model_data_offset = 25;  // Skip performance and samples
        }
        std::vector<uint8_t> model_data(buffer.begin() + model_data_offset, buffer.end());
        
        try {
            // Extract and decompress model data
            FederatedLearning fl_temp;  // For decompression utility
            auto decompressed_params = fl_temp.decompressParameters(model_data);
            
            // Create model snapshot
            ModelSnapshot client_snapshot;
            client_snapshot.version_hash = fl_temp.computeModelHash(decompressed_params);
            client_snapshot.timestamp = std::chrono::system_clock::now();
            client_snapshot.parameters = decompressed_params;
            client_snapshot.performance_metrics = performance_metrics;
            client_snapshot.training_samples = training_samples;
            
            // Process model update with client address
            processModelUpdate(agent_id, client_snapshot, client_address);
            
            // Send back aggregated model
            ModelSnapshot aggregated = aggregateModels();
            auto response_data = aggregated.serialize();
            
            // Add message type
            std::vector<uint8_t> response;
            response.push_back(static_cast<uint8_t>(FederatedMessageType::MODEL_UPDATE));
            response.insert(response.end(), response_data.begin(), response_data.end());
            
            send(client_socket, response.data(), response.size(), 0);
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing client update: " << e.what() << std::endl;
        }
    }
    
    close(client_socket);
}

ModelSnapshot DistillationBank::aggregateModels() {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    if (agents_.empty()) {
        return global_model_;
    }
    
    // Weighted average of all agent models
    ModelSnapshot aggregated;
    aggregated.timestamp = std::chrono::system_clock::now();
    
    // Calculate total weight
    float total_weight = 0.0f;
    for (const auto& [_, agent] : agents_) {
        total_weight += agent.contribution_weight;
    }
    
    // Average parameters
    for (const auto& [agent_id, agent] : agents_) {
        float weight = agent.contribution_weight / total_weight;
        
        for (const auto& [param_name, param_tensor] : agent.latest_model.parameters) {
            if (aggregated.parameters.find(param_name) == aggregated.parameters.end()) {
                aggregated.parameters[param_name] = Tensor(param_tensor.shape());
                aggregated.parameters[param_name].zero();
            }
            
            // Weighted sum
            Tensor& agg_tensor = aggregated.parameters[param_name];
            float* agg_data = agg_tensor.data_float();
            const float* param_data = param_tensor.data_float();
            
            for (size_t i = 0; i < param_tensor.size(); ++i) {
                agg_data[i] += weight * param_data[i];
            }
        }
    }
    
    // Update hash
    FederatedLearning temp_fl;
    aggregated.version_hash = temp_fl.computeModelHash(aggregated.parameters);
    
    // Update global model
    global_model_ = aggregated;
    
    return aggregated;
}

void DistillationBank::processModelUpdate(const std::string& agent_id, const ModelSnapshot& snapshot, 
                                        const std::string& address) {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    // Update or create agent info
    AgentInfo& agent = agents_[agent_id];
    agent.agent_id = agent_id;
    agent.address = address;
    agent.latest_model = snapshot;
    agent.last_update = std::chrono::system_clock::now();
    
    // Calculate contribution weight based on performance and sample count
    float perf_weight = snapshot.performance_metrics;
    float sample_weight = std::min(1.0f, snapshot.training_samples / 10000.0f);
    agent.contribution_weight = perf_weight * sample_weight;
    
    // Trigger global model update if enough agents have reported
    if (agents_.size() >= 2) {  // Minimum agents for aggregation
        std::lock_guard<std::mutex> global_lock(global_model_mutex_);
        global_model_ = aggregateModels();
    }
}

ModelSnapshot DistillationBank::distillGlobalKnowledge() {
    std::lock_guard<std::mutex> global_lock(global_model_mutex_);
    
    if (agents_.empty()) {
        return global_model_;
    }
    
    // Apply knowledge distillation across all agent models
    ModelSnapshot distilled = global_model_;
    
    // Temperature for distillation
    const float temperature = 3.0f;
    const float alpha = 0.7f;
    
    // Average logits across all models with temperature scaling
    for (auto& [param_name, param_tensor] : distilled.parameters) {
        Tensor avg_tensor(param_tensor.shape());
        avg_tensor.zero();
        
        float total_weight = 0.0f;
        
        // Accumulate scaled parameters
        for (const auto& [agent_id, agent] : agents_) {
            auto it = agent.latest_model.parameters.find(param_name);
            if (it != agent.latest_model.parameters.end()) {
                const Tensor& agent_tensor = it->second;
                float* avg_data = avg_tensor.data_float();
                const float* agent_data = agent_tensor.data_float();
                
                for (size_t i = 0; i < avg_tensor.size(); ++i) {
                    avg_data[i] += agent.contribution_weight * agent_data[i] / temperature;
                }
                total_weight += agent.contribution_weight;
            }
        }
        
        // Normalize
        if (total_weight > 0) {
            float* avg_data = avg_tensor.data_float();
            float* distilled_data = param_tensor.data_float();
            
            for (size_t i = 0; i < avg_tensor.size(); ++i) {
                avg_data[i] /= total_weight;
                // Blend with original
                distilled_data[i] = alpha * avg_data[i] + (1 - alpha) * distilled_data[i];
            }
        }
    }
    
    // Update version hash
    FederatedLearning fl_temp;
    distilled.version_hash = fl_temp.computeModelHash(distilled.parameters);
    distilled.timestamp = std::chrono::system_clock::now();
    
    return distilled;
}

void DistillationBank::broadcastGlobalModel() {
    ModelSnapshot global = distillGlobalKnowledge();
    
    // Broadcast to all connected agents
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    // Prepare broadcast message
    auto model_data = global.serialize();
    std::vector<uint8_t> message;
    message.push_back(static_cast<uint8_t>(FederatedMessageType::MODEL_UPDATE));
    message.insert(message.end(), model_data.begin(), model_data.end());
    
    // Send to each connected agent
    for (const auto& [agent_id, agent_info] : agents_) {
        // Parse agent address
        size_t colon_pos = agent_info.address.find(':');
        if (colon_pos == std::string::npos) continue;
        
        std::string ip = agent_info.address.substr(0, colon_pos);
        int port = std::stoi(agent_info.address.substr(colon_pos + 1));
        
        // Create socket for this agent
        int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) continue;
        
        struct sockaddr_in agent_addr;
        agent_addr.sin_family = AF_INET;
        agent_addr.sin_port = htons(port);
        inet_pton(AF_INET, ip.c_str(), &agent_addr.sin_addr);
        
        // Send model update
        sendto(sock, message.data(), message.size(), 0,
               (struct sockaddr*)&agent_addr, sizeof(agent_addr));
        
        close(sock);
    }
    
    std::cout << "Broadcast global model v" << global.version_hash.substr(0, 8) 
              << " to " << agents_.size() << " agents" << std::endl;
}

} // namespace TACS