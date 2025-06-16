#pragma once

#include <vector>
#include <memory>
#include <chrono>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <cmath>

namespace tacs {

struct Point2D {
    float x;
    float y;
    
    float distance(const Point2D& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }
};

struct VoronoiCell {
    Point2D center;
    std::vector<Point2D> vertices;
    float area;
    int droneId;
};

struct DroneState {
    int id;
    Point2D position;
    Point2D targetPosition;
    float altitude;
    float batteryLevel;
    float speed;
    float maxSpeed;
    float batteryDrainRate;
    bool isActive;
    bool needsReplacement;
    std::chrono::steady_clock::time_point lastUpdate;
    
    float remainingFlightTime() const {
        return batteryLevel / batteryDrainRate;
    }
};

struct CoverageMetrics {
    float totalArea;
    float coveredArea;
    float coverageRatio;
    int activeDrones;
    int chargingDrones;
    float averageBatteryLevel;
};

struct DroneSwarmConfig {
    float minBatteryThreshold = 0.2f;
    float criticalBatteryThreshold = 0.1f;
    float idealAltitude = 50.0f;
    float communicationRange = 500.0f;
    float batteryDrainRate = 0.001f;
    float maxDroneSpeed = 15.0f;
    double referenceLatitude = 37.7749;   // San Francisco default
    double referenceLongitude = -122.4194; // San Francisco default
    float groundPlaneHeight = 0.0f;
    bool enableV2XLocalProcessing = true;
    std::string droneNetworkBase = "192.168.100.0";
    int v2xPort = 47347;
};

class DroneSwarm {
public:
    DroneSwarm(float coverageAreaWidth, float coverageAreaHeight, int initialDroneCount);
    DroneSwarm(float coverageAreaWidth, float coverageAreaHeight, int initialDroneCount, 
               const DroneSwarmConfig& config);
    ~DroneSwarm();
    
    void updateSwarm(float deltaTime);
    
    void addDrone(const Point2D& position);
    void removeDrone(int droneId);
    void replaceDrone(int droneId);
    
    std::vector<VoronoiCell> computeVoronoiDiagram();
    
    void optimizeCoverage();
    
    CoverageMetrics getCoverageMetrics() const;
    
    const std::vector<DroneState>& getDrones() const { return drones_; }
    
    void handleDroneMessage(int droneId, const std::vector<uint8_t>& message);
    
    void sendMessageToDrone(int droneId, const std::vector<uint8_t>& message);
    void broadcastMessage(const std::vector<uint8_t>& message);
    
    void setChargingStationPosition(const Point2D& position) { chargingStation_ = position; }
    
    bool requestCoverageReassignment(int droneId, const Point2D& newPosition);
    
    std::vector<uint8_t> serializeSwarmState() const;
    void deserializeSwarmState(const std::vector<uint8_t>& data);
    
    // Testing support - simulate battery drain
    void simulateBatteryDrain(int droneId, float drainAmount);
    
private:
    float coverageAreaWidth_;
    float coverageAreaHeight_;
    std::vector<DroneState> drones_;
    std::unordered_map<int, size_t> droneIdToIndex_;
    Point2D chargingStation_;
    
    mutable std::mutex swarmMutex_;
    std::atomic<int> nextDroneId_{0};
    std::atomic<bool> isRunning_{false};
    std::thread updateThread_;
    
    DroneSwarmConfig config_;
    
    // V2X communication
    int v2xSocket_ = 0;
    std::atomic<uint64_t> v2xMessagesSent_{0};
    std::atomic<uint64_t> v2xTransmissionErrors_{0};
    
    void updateDronePositions(float deltaTime);
    void monitorBatteryLevels();
    void handleLowBattery(DroneState& drone);
    
    Point2D computeOptimalPosition(const DroneState& drone, const std::vector<VoronoiCell>& cells);
    
    std::vector<Point2D> computeVoronoiVertices(int droneIndex, const std::vector<DroneState>& allDrones);
    
    bool isPointInCoverageArea(const Point2D& point) const;
    
    void rebalanceCoverage();
    
    std::vector<int> findNeighborDrones(int droneId, float maxDistance);
    
    void updateThreadFunc();
    
    void initializeV2XSocket();
    void cleanupV2XSocket();
};

class DroneMessaging {
public:
    enum MessageType : uint8_t {
        POSITION_UPDATE = 0x01,
        BATTERY_STATUS = 0x02,
        COVERAGE_REQUEST = 0x03,
        EMERGENCY_LANDING = 0x04,
        DETECTION_REPORT = 0x05,
        SWARM_REBALANCE = 0x06,
        HEARTBEAT = 0x07
    };
    
    struct Message {
        MessageType type;
        int senderId;
        int receiverId;
        std::vector<uint8_t> payload;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    static std::vector<uint8_t> encodeMessage(const Message& msg);
    static Message decodeMessage(const std::vector<uint8_t>& data);
    
    static std::vector<uint8_t> encodePositionUpdate(const Point2D& position, float altitude);
    static std::tuple<Point2D, float> decodePositionUpdate(const std::vector<uint8_t>& payload);
    
    static std::vector<uint8_t> encodeBatteryStatus(float batteryLevel, float estimatedFlightTime);
    static std::pair<float, float> decodeBatteryStatus(const std::vector<uint8_t>& payload);
    
    static std::vector<uint8_t> encodeDetectionReport(int objectType, const Point2D& location, float confidence);
    static std::tuple<int, Point2D, float> decodeDetectionReport(const std::vector<uint8_t>& payload);
};

} // namespace tacs