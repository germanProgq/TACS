#pragma once

#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include "core/tensor.h"
#include "models/tacsnet.h"
#include "drone/drone_swarm.h"

namespace tacs {

using core::Tensor;
using models::TACSNet;
using models::TACSNetUltra;
using models::Detection;

struct IMUData {
    float roll;
    float pitch;
    float yaw;
    float accelX;
    float accelY;
    float accelZ;
    float gyroX;
    float gyroY;
    float gyroZ;
    std::chrono::steady_clock::time_point timestamp;
};

struct StabilizationParams {
    float horizonCorrection;
    float verticalCorrection;
    float rotationCorrection;
    float scaleFactor;
    float cropOffsetX;
    float cropOffsetY;
};

struct AerialDetection {
    float x, y, w, h;
    float confidence;
    int classId;
    Point2D worldCoordinate;
    float altitude;
};

class AerialInference {
public:
    AerialInference(int droneImageWidth, int droneImageHeight, float droneAltitude);
    ~AerialInference();
    
    void processFrame(const uint8_t* imageData, const IMUData& imuData);
    
    std::vector<AerialDetection> getDetections() const;
    
    void updateCalibration(float focalLength, float sensorWidth, float sensorHeight);
    
    StabilizationParams computeStabilization(const IMUData& imuData);
    
    Tensor stabilizeImage(const Tensor& inputImage, const StabilizationParams& params);
    
    Tensor downscaleForDrone(const Tensor& inputImage, int targetWidth, int targetHeight);
    
    Point2D pixelToWorldCoordinate(float pixelX, float pixelY, float altitude);
    
    void setGroundPlaneHeight(float height) { groundPlaneHeight_ = height; }
    
    float getProcessingLatency() const { return lastProcessingTime_; }
    
    void enableAdaptiveDownscaling(bool enable) { adaptiveDownscaling_ = enable; }
    
    void setCompressionQuality(float quality);
    
    // Fast mode for testing - uses simplified detection
    void enableFastMode(bool enable) { fastMode_ = enable; }
    
private:
    std::unique_ptr<TACSNet> tacsNet_;
    std::unique_ptr<TACSNetUltra> tacsNetUltra_;  // Ultra-lightweight model for fast mode
    int droneImageWidth_;
    int droneImageHeight_;
    float droneAltitude_;
    float groundPlaneHeight_;
    
    float focalLength_;
    float sensorWidth_;
    float sensorHeight_;
    
    mutable std::mutex detectionMutex_;
    std::vector<AerialDetection> latestDetections_;
    
    std::queue<IMUData> imuHistory_;
    static constexpr size_t MAX_IMU_HISTORY = 10;
    
    float lastProcessingTime_;
    std::chrono::steady_clock::time_point lastFrameTime_;
    
    bool adaptiveDownscaling_;
    float compressionQuality_;
    bool fastMode_;
    
    // Pre-allocated workspace for performance
    mutable Tensor workspaceImage_;
    mutable Tensor workspaceBatch_;
    mutable bool workspaceInitialized_;
    
    Tensor applyPerspectiveCorrection(const Tensor& image, const IMUData& imu);
    
    Tensor applyMotionBlurCompensation(const Tensor& image, const IMUData& imu);
    
    float calculateGroundSampleDistance(float altitude);
    
    void adjustDetectionsForAltitude(std::vector<Detection>& detections, float altitude);
    
    Tensor preprocessForAerial(const Tensor& image);
    
    float estimateMotionBlur(const IMUData& current, const IMUData& previous);
    
    void filterSmallObjects(std::vector<Detection>& detections, float minSize);
    
    Tensor enhanceContrast(const Tensor& image);
    
    std::pair<float, float> calculateFieldOfView();
    
    std::vector<Detection> extractUltraDetections(const models::DetectionOutput& output, float conf_threshold);
    
    void processFastMode(const uint8_t* imageData, const IMUData& imuData);
};

class DroneToGroundComm {
public:
    struct GroundMessage {
        enum Type {
            DETECTION_BATCH,
            COVERAGE_UPDATE,
            BATTERY_ALERT,
            IMAGE_STREAM,
            TELEMETRY
        };
        
        Type type;
        int droneId;
        std::vector<uint8_t> payload;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    DroneToGroundComm(int droneId, const std::string& groundStationAddress);
    ~DroneToGroundComm();
    
    void sendDetections(const std::vector<AerialDetection>& detections);
    
    void sendCoverageUpdate(const VoronoiCell& coverage);
    
    void sendBatteryAlert(float batteryLevel, float estimatedFlightTime);
    
    void streamCompressedImage(const Tensor& image, float quality);
    
    void sendTelemetry(const IMUData& imu, const Point2D& position, float altitude);
    
    bool isConnected() const { return isConnected_; }
    
    void setCompressionLevel(int level) { compressionLevel_ = level; }
    
    void enableEncryption(bool enable) { encryptionEnabled_ = enable; }
    
    float getBandwidthUsage() const { return bandwidthUsage_; }
    
private:
    int droneId_;
    std::string groundStationAddress_;
    std::atomic<bool> isConnected_{false};
    
    int compressionLevel_;
    bool encryptionEnabled_;
    float bandwidthUsage_;
    
    std::queue<GroundMessage> messageQueue_;
    mutable std::mutex queueMutex_;
    std::thread commThread_;
    std::atomic<bool> shouldStop_{false};
    
    void communicationLoop();
    
    std::vector<uint8_t> compressData(const std::vector<uint8_t>& data);
    
    std::vector<uint8_t> encryptData(const std::vector<uint8_t>& data);
    
    void updateBandwidthMetrics(size_t bytesSent);
    
    std::vector<uint8_t> serializeDetections(const std::vector<AerialDetection>& detections);
    
    std::vector<uint8_t> compressImage(const Tensor& image, float quality);
};

class AdaptiveResolution {
public:
    AdaptiveResolution(int baseWidth, int baseHeight);
    
    std::pair<int, int> computeOptimalResolution(float processingLoad, float batteryLevel, float bandwidth);
    
    void updateMetrics(float latency, float accuracy);
    
    float getQualityScore() const { return qualityScore_; }
    
    void setMinResolution(int width, int height);
    void setMaxResolution(int width, int height);
    
private:
    int baseWidth_;
    int baseHeight_;
    int minWidth_;
    int minHeight_;
    int maxWidth_;
    int maxHeight_;
    
    float qualityScore_;
    float targetLatency_;
    
    struct ResolutionProfile {
        int width;
        int height;
        float expectedLatency;
        float expectedAccuracy;
    };
    
    std::vector<ResolutionProfile> profiles_;
    
    void initializeProfiles();
    
    float predictLatency(int width, int height);
    
    float predictAccuracy(int width, int height);
};

} // namespace tacs