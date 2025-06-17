/*
 * V2X Protocol Implementation for TACS
 * DSRC and C-V2X compliant communication
 */

#ifndef V2X_PROTOCOL_H
#define V2X_PROTOCOL_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <queue>
#include <functional>
#include <condition_variable>

#include "../include/core/tensor.h"

namespace TACS {

using tacs::core::Tensor;

// V2X message types according to SAE J2735 standard
enum class V2XMessageType : uint16_t {
    // Core messages
    BSM = 0x0014,              // Basic Safety Message
    SPaT = 0x0013,             // Signal Phase and Timing
    MAP = 0x0012,              // MAP data
    TIM = 0x001F,              // Traveler Information Message
    
    // TACS-specific extensions
    MODEL_SYNC = 0x1001,       // Federated model synchronization
    PERFORMANCE_REPORT = 0x1002, // Performance metrics sharing
    EMERGENCY_OVERRIDE = 0x1003, // Emergency traffic control
    DRONE_TELEMETRY = 0x1004,   // Drone position and status
    TRAFFIC_FLOW = 0x1005,      // Traffic flow statistics
    WEATHER_UPDATE = 0x1006,    // Weather condition updates
    ACCIDENT_ALERT = 0x1007     // Accident detection alert
};

// V2X communication modes
enum class V2XMode {
    DSRC,      // Dedicated Short Range Communications (802.11p)
    CV2X_PC5,  // Cellular V2X PC5 (direct)
    CV2X_UU    // Cellular V2X Uu (network)
};

// Basic Safety Message structure compliant with SAE J2735
struct BasicSafetyMessage {
    uint32_t msg_count;
    uint32_t id;  // Temporary ID
    double latitude;
    double longitude;
    double elevation;
    double speed;  // m/s
    double heading;  // degrees
    double acceleration;
    uint8_t brake_status;
    uint32_t vehicle_size_length;  // cm
    uint32_t vehicle_size_width;   // cm
    std::chrono::system_clock::time_point timestamp;
    
    std::vector<uint8_t> serialize() const;
    static BasicSafetyMessage deserialize(const std::vector<uint8_t>& data);
};

// Signal Phase and Timing message
struct SPaTMessage {
    uint32_t intersection_id;
    uint8_t signal_group;
    uint8_t event_state;  // red, yellow, green, etc.
    uint16_t min_end_time;  // deciseconds
    uint16_t max_end_time;  // deciseconds
    uint16_t likely_time;   // deciseconds
    uint8_t confidence;
    
    std::vector<uint8_t> serialize() const;
    static SPaTMessage deserialize(const std::vector<uint8_t>& data);
};

// TACS model sync message
struct ModelSyncMessage {
    std::string agent_id;
    std::string model_version;
    float performance_metric;
    size_t training_samples;
    std::vector<uint8_t> compressed_params;  // Compressed model parameters
    
    std::vector<uint8_t> serialize() const;
    static ModelSyncMessage deserialize(const std::vector<uint8_t>& data);
};

// V2X configuration
struct V2XConfig {
    V2XMode mode = V2XMode::DSRC;
    int channel = 178;  // DSRC channel 178 (5.89 GHz)
    int tx_power = 20;  // dBm
    std::string interface = "wlan0";  // Network interface
    
    // DSRC parameters
    uint16_t psid = 0x20;  // Provider Service Identifier
    uint8_t priority = 6;   // 802.11p priority
    
    // Security
    bool enable_security = true;
    std::string cert_path = "/etc/tacs/v2x_cert.pem";
    std::string key_path = "/etc/tacs/v2x_key.pem";
    
    // Timing
    std::chrono::milliseconds bsm_interval{100};  // 10 Hz
    std::chrono::milliseconds spat_interval{1000}; // 1 Hz
    
    // Geo-networking
    double comm_range = 300.0;  // meters
    bool enable_multi_hop = true;
    uint8_t max_hops = 3;
};

// V2X communication handler
class V2XProtocol {
public:
    V2XProtocol(const V2XConfig& config = V2XConfig());
    ~V2XProtocol();
    
    // Initialize V2X communication
    bool initialize();
    void shutdown();
    
    // Message transmission
    void sendBSM(const BasicSafetyMessage& bsm);
    void sendSPaT(const SPaTMessage& spat);
    void sendModelSync(const ModelSyncMessage& sync);
    void broadcastMessage(V2XMessageType type, const std::vector<uint8_t>& data);
    
    // Message reception
    void registerHandler(V2XMessageType type, 
                        std::function<void(const std::vector<uint8_t>&, const std::string&)> handler);
    
    // Geo-networking
    void enableGeoNetworking(double latitude, double longitude);
    void setGeocastArea(double center_lat, double center_lon, double radius);
    
    // Security
    bool verifyMessage(const std::vector<uint8_t>& message);
    std::vector<uint8_t> signMessage(const std::vector<uint8_t>& message);
    
    // Statistics
    struct V2XStats {
        uint64_t messages_sent;
        uint64_t messages_received;
        uint64_t messages_dropped;
        double avg_latency_ms;
        double packet_delivery_ratio;
        std::unordered_map<V2XMessageType, uint64_t> message_counts;
    };
    
    V2XStats getStatistics() const;
    
private:
    V2XConfig config_;
    std::atomic<bool> running_{false};
    
    // Network sockets
    int dsrc_socket_;
    int cv2x_socket_;
    
    // Message handling
    std::unordered_map<V2XMessageType, 
                      std::function<void(const std::vector<uint8_t>&, const std::string&)>> handlers_;
    std::mutex handlers_mutex_;
    
    // Threads
    std::thread rx_thread_;
    std::thread tx_thread_;
    std::thread maintenance_thread_;
    
    // Message queues
    struct QueuedMessage {
        V2XMessageType type;
        std::vector<uint8_t> data;
        std::chrono::system_clock::time_point timestamp;
        uint8_t priority;
        double target_lat;
        double target_lon;
        double target_radius;
        
        // Comparison operator for priority queue (higher priority first)
        bool operator<(const QueuedMessage& other) const {
            return priority < other.priority;  // Note: reversed for max heap
        }
    };
    
    std::priority_queue<QueuedMessage> tx_queue_;
    std::mutex tx_queue_mutex_;
    std::condition_variable tx_cv_;
    
    // Geo-networking
    double current_lat_;
    double current_lon_;
    std::unordered_map<std::string, std::chrono::system_clock::time_point> neighbor_table_;
    
    // Security context
    void* security_context_;  // OpenSSL context
    
    // Statistics
    mutable std::mutex stats_mutex_;
    V2XStats stats_;
    
    // Helper methods
    void rxThreadWorker();
    void txThreadWorker();
    void maintenanceThreadWorker();
    
    bool initializeDSRC();
    bool initializeCV2X();
    
    void processDSRCMessage(const std::vector<uint8_t>& message, const std::string& sender);
    void processCV2XMessage(const std::vector<uint8_t>& message, const std::string& sender);
    
    bool isInGeocastArea(double lat, double lon) const;
    double calculateDistance(double lat1, double lon1, double lat2, double lon2) const;
    
    // IEEE 1609 WAVE protocol stack
    std::vector<uint8_t> encapsulateWAVE(V2XMessageType type, const std::vector<uint8_t>& data);
    std::pair<V2XMessageType, std::vector<uint8_t>> decapsulateWAVE(const std::vector<uint8_t>& frame);
    
    // Cryptographic functions
    void computeHMAC(const std::vector<uint8_t>& data, uint8_t* output);
    void computeSHA256(const uint8_t* data, size_t len, uint8_t* output);
};

// Inter-intersection synchronization protocol
class IntersectionSync {
public:
    IntersectionSync(const std::string& intersection_id);
    ~IntersectionSync();
    
    // Register with neighboring intersections
    void registerNeighbor(const std::string& neighbor_id, const std::string& address);
    
    // Synchronize traffic state
    struct TrafficState {
        std::string intersection_id;
        std::unordered_map<std::string, uint32_t> queue_lengths;  // approach -> queue length
        std::unordered_map<std::string, double> flow_rates;       // approach -> vehicles/min
        uint8_t current_phase;
        uint16_t phase_remaining_time;  // seconds
        bool emergency_active;
        
        std::vector<uint8_t> serialize() const;
        static TrafficState deserialize(const std::vector<uint8_t>& data);
    };
    
    void broadcastState(const TrafficState& state);
    void requestNeighborStates();
    
    // Coordinated control
    struct CoordinationRequest {
        std::string requesting_intersection;
        std::string coordination_type;  // "green_wave", "emergency", "congestion"
        std::unordered_map<std::string, double> parameters;
        
        std::vector<uint8_t> serialize() const;
        static CoordinationRequest deserialize(const std::vector<uint8_t>& data);
    };
    
    void requestCoordination(const CoordinationRequest& request);
    void registerCoordinationHandler(std::function<void(const CoordinationRequest&)> handler);
    
    // Get neighbor states
    std::unordered_map<std::string, TrafficState> getNeighborStates() const;
    
private:
    std::string intersection_id_;
    std::unique_ptr<V2XProtocol> v2x_;
    
    // Neighbor management
    struct NeighborInfo {
        std::string id;
        std::string address;
        TrafficState latest_state;
        std::chrono::system_clock::time_point last_update;
        double reliability_score;
    };
    
    std::unordered_map<std::string, NeighborInfo> neighbors_;
    mutable std::mutex neighbors_mutex_;
    
    // Coordination
    std::function<void(const CoordinationRequest&)> coordination_handler_;
    std::mutex coordination_mutex_;
    
    // Message handlers
    void handleTrafficState(const std::vector<uint8_t>& data, const std::string& sender);
    void handleCoordinationRequest(const std::vector<uint8_t>& data, const std::string& sender);
};

// Vehicle-to-Infrastructure data exchange
class V2IExchange {
public:
    V2IExchange();
    ~V2IExchange();
    
    // Vehicle data collection
    struct VehicleData {
        std::string vehicle_id;
        std::string vehicle_type;  // "car", "truck", "bus", "emergency"
        double position_lat;
        double position_lon;
        double speed;  // m/s
        double heading;  // degrees
        std::string destination;  // optional
        std::vector<std::string> route_preferences;  // "fastest", "shortest", "eco"
        
        std::vector<uint8_t> serialize() const;
        static VehicleData deserialize(const std::vector<uint8_t>& data);
    };
    
    void collectVehicleData(const VehicleData& data);
    
    // Infrastructure guidance
    struct GuidanceMessage {
        std::string vehicle_id;
        std::string message_type;  // "route", "speed", "lane", "warning"
        std::unordered_map<std::string, std::string> parameters;
        uint32_t validity_duration;  // seconds
        
        std::vector<uint8_t> serialize() const;
        static GuidanceMessage deserialize(const std::vector<uint8_t>& data);
    };
    
    void sendGuidance(const GuidanceMessage& guidance);
    
    // Aggregate statistics
    struct V2IStatistics {
        uint32_t connected_vehicles;
        double avg_speed;
        double avg_travel_time;
        std::unordered_map<std::string, uint32_t> vehicle_types;
        std::unordered_map<std::string, double> route_utilization;
    };
    
    V2IStatistics getStatistics() const;
    
private:
    std::unique_ptr<V2XProtocol> v2x_;
    
    // Vehicle tracking
    std::unordered_map<std::string, VehicleData> vehicles_;
    mutable std::mutex vehicles_mutex_;
    
    // Statistics collection
    V2IStatistics stats_;
    mutable std::mutex stats_mutex_;
    
    void updateStatistics();
};

} // namespace TACS

#endif // V2X_PROTOCOL_H