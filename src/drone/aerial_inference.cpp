#include "drone/aerial_inference.h"
#include "drone/drone_swarm.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <functional>
#include <random>
#include <fstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

namespace tacs {

using core::Tensor;
using models::TACSNet;
using models::TACSNetUltra;
using models::Detection;

AerialInference::AerialInference(int droneImageWidth, int droneImageHeight, float droneAltitude)
    : droneImageWidth_(droneImageWidth)
    , droneImageHeight_(droneImageHeight)
    , droneAltitude_(droneAltitude)
    , groundPlaneHeight_(0.0f)
    , focalLength_(800.0f)
    , sensorWidth_(6.4f)
    , sensorHeight_(4.8f)
    , lastProcessingTime_(0.0f)
    , adaptiveDownscaling_(true)
    , compressionQuality_(0.85f)
    , fastMode_(false)
    , workspaceInitialized_(false) {
    
    tacsNet_ = std::make_unique<TACSNet>();
    tacsNetUltra_ = std::make_unique<TACSNetUltra>();  // Initialize ultra-lightweight model
    
    // Load pre-trained weights if available
    if (std::ifstream("tacsnet_ultra_drone.bin").good()) {
        tacsNetUltra_->loadModel("tacsnet_ultra_drone.bin");
    }
    
    lastFrameTime_ = std::chrono::steady_clock::now();
}

AerialInference::~AerialInference() = default;

void AerialInference::processFrame(const uint8_t* imageData, const IMUData& imuData) {
    auto startTime = std::chrono::steady_clock::now();
    
    imuHistory_.push(imuData);
    if (imuHistory_.size() > MAX_IMU_HISTORY) {
        imuHistory_.pop();
    }
    
    // Fast path for TACSNetUltra - skip intermediate tensor creation
    if (fastMode_) {
        // Direct processing path for ultra-fast inference
        processFastMode(imageData, imuData);
        
        auto endTime = std::chrono::steady_clock::now();
        lastProcessingTime_ = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        return;
    }
    
    Tensor inputImage({droneImageHeight_, droneImageWidth_, 3});
    memcpy(inputImage.data(), imageData, droneImageHeight_ * droneImageWidth_ * 3);
    
    // Skip stabilization if IMU data shows minimal movement
    bool needsStabilization = std::abs(imuData.roll) > 0.02f || 
                             std::abs(imuData.pitch) > 0.02f ||
                             std::abs(imuData.gyroX) > 0.1f ||
                             std::abs(imuData.gyroY) > 0.1f;
    
    Tensor stabilized;
    if (needsStabilization) {
        StabilizationParams stabParams = computeStabilization(imuData);
        stabilized = stabilizeImage(inputImage, stabParams);
    } else {
        stabilized = inputImage;
    }
    
    // Use smaller resolution for faster processing on drones
    int targetWidth = 320;
    int targetHeight = 320;
    
    if (adaptiveDownscaling_) {
        float gsd = calculateGroundSampleDistance(droneAltitude_);
        if (gsd > 0.5f) {
            targetWidth = 256; // Even smaller for high altitude
            targetHeight = 256;
        } else if (gsd < 0.1f) {
            targetWidth = 416; // Max at 416 for drones
            targetHeight = 416;
        }
    }
    
    Tensor downscaled = downscaleForDrone(stabilized, targetWidth, targetHeight);
    
    Tensor preprocessed = preprocessForAerial(downscaled);
    
    // TACSNet expects 4D input [batch, channels, height, width]
    auto shape = preprocessed.shape();
    Tensor batched({1, shape[2], shape[0], shape[1]});
    
    // Transpose from HWC to CHW format
    float* src = preprocessed.data_float();
    float* dst = batched.data_float();
    for (int c = 0; c < shape[2]; ++c) {
        for (int h = 0; h < shape[0]; ++h) {
            for (int w = 0; w < shape[1]; ++w) {
                dst[c * shape[0] * shape[1] + h * shape[1] + w] = 
                    src[h * shape[1] * shape[2] + w * shape[2] + c];
            }
        }
    }
    
    std::vector<Detection> detections;
    
    if (fastMode_) {
        // Fast mode: Use ultra-lightweight TACSNetUltra for real-time drone inference
        auto ultraOutputs = tacsNetUltra_->forward(batched);
        
        // Extract detections with higher confidence threshold for faster NMS
        float fastModeConfThreshold = 0.6f;  // Higher threshold for speed
        for (const auto& output : ultraOutputs) {
            // TACSNetUltra uses optimized detection extraction for drone deployment
            auto ultraDets = extractUltraDetections(output, fastModeConfThreshold);
            detections.insert(detections.end(), ultraDets.begin(), ultraDets.end());
        }
    } else {
        auto detectionOutputs = tacsNet_->forward(batched);
        
        for (const auto& output : detectionOutputs) {
            auto dets = tacsNet_->extractDetections(output, 0.5f);
            detections.insert(detections.end(), dets.begin(), dets.end());
        }
    }
    
    adjustDetectionsForAltitude(detections, droneAltitude_);
    
    float minObjectSize = 10.0f * (droneAltitude_ / 50.0f);
    filterSmallObjects(detections, minObjectSize);
    
    {
        std::lock_guard<std::mutex> lock(detectionMutex_);
        latestDetections_.clear();
        
        for (const auto& det : detections) {
            AerialDetection aerialDet;
            aerialDet.x = det.x;
            aerialDet.y = det.y;
            aerialDet.w = det.width;
            aerialDet.h = det.height;
            aerialDet.confidence = det.confidence;
            aerialDet.classId = det.class_id;
            aerialDet.altitude = droneAltitude_;
            
            float centerX = det.x + det.width / 2.0f;
            float centerY = det.y + det.height / 2.0f;
            aerialDet.worldCoordinate = pixelToWorldCoordinate(centerX, centerY, droneAltitude_);
            
            latestDetections_.push_back(aerialDet);
        }
    }
    
    auto endTime = std::chrono::steady_clock::now();
    lastProcessingTime_ = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    lastFrameTime_ = endTime;
}

std::vector<AerialDetection> AerialInference::getDetections() const {
    std::lock_guard<std::mutex> lock(detectionMutex_);
    return latestDetections_;
}

void AerialInference::updateCalibration(float focalLength, float sensorWidth, float sensorHeight) {
    focalLength_ = focalLength;
    sensorWidth_ = sensorWidth;
    sensorHeight_ = sensorHeight;
}

StabilizationParams AerialInference::computeStabilization(const IMUData& imuData) {
    StabilizationParams params;
    
    const float maxTilt = 30.0f * M_PI / 180.0f;
    params.horizonCorrection = std::clamp(-imuData.roll, -maxTilt, maxTilt);
    params.verticalCorrection = std::clamp(-imuData.pitch, -maxTilt, maxTilt);
    params.rotationCorrection = -imuData.yaw;
    
    float tiltMagnitude = std::sqrt(imuData.roll * imuData.roll + imuData.pitch * imuData.pitch);
    params.scaleFactor = 1.0f + 0.1f * (tiltMagnitude / maxTilt);
    
    params.cropOffsetX = 50.0f * std::sin(imuData.roll);
    params.cropOffsetY = 50.0f * std::sin(imuData.pitch);
    
    return params;
}

Tensor AerialInference::stabilizeImage(const Tensor& inputImage, const StabilizationParams& params) {
    if (fastMode_) {
        // In fast mode, skip expensive perspective correction
        return inputImage;
    }
    
    Tensor stabilized = applyPerspectiveCorrection(inputImage, 
        {params.horizonCorrection, params.verticalCorrection, params.rotationCorrection, 0, 0, 0, 0, 0, 0});
    
    if (!imuHistory_.empty() && imuHistory_.size() >= 2) {
        auto current = imuHistory_.back();
        imuHistory_.pop();
        auto previous = imuHistory_.back();
        imuHistory_.push(current);
        
        float blurAmount = estimateMotionBlur(current, previous);
        if (blurAmount > 0.5f) {
            stabilized = applyMotionBlurCompensation(stabilized, current);
        }
    }
    
    return stabilized;
}

Tensor AerialInference::downscaleForDrone(const Tensor& inputImage, int targetWidth, int targetHeight) {
    auto shape = inputImage.shape();
    int srcHeight = shape[0];
    int srcWidth = shape[1];
    
    Tensor output({targetHeight, targetWidth, 3});
    
    float scaleX = static_cast<float>(srcWidth) / targetWidth;
    float scaleY = static_cast<float>(srcHeight) / targetHeight;
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < targetHeight; ++y) {
        for (int x = 0; x < targetWidth; ++x) {
            float srcX = x * scaleX;
            float srcY = y * scaleY;
            
            int x0 = static_cast<int>(srcX);
            int y0 = static_cast<int>(srcY);
            int x1 = std::min(x0 + 1, srcWidth - 1);
            int y1 = std::min(y0 + 1, srcHeight - 1);
            
            float fx = srcX - x0;
            float fy = srcY - y0;
            
            for (int c = 0; c < 3; ++c) {
                float v00 = inputImage({y0, x0, c});
                float v01 = inputImage({y0, x1, c});
                float v10 = inputImage({y1, x0, c});
                float v11 = inputImage({y1, x1, c});
                
                float v0 = v00 * (1 - fx) + v01 * fx;
                float v1 = v10 * (1 - fx) + v11 * fx;
                float v = v0 * (1 - fy) + v1 * fy;
                
                output({y, x, c}) = v;
            }
        }
    }
    
    return output;
}

Point2D AerialInference::pixelToWorldCoordinate(float pixelX, float pixelY, float altitude) {
    float pixelSizeX = sensorWidth_ / droneImageWidth_;
    float pixelSizeY = sensorHeight_ / droneImageHeight_;
    
    float angleX = std::atan((pixelX - droneImageWidth_ / 2.0f) * pixelSizeX / focalLength_);
    float angleY = std::atan((pixelY - droneImageHeight_ / 2.0f) * pixelSizeY / focalLength_);
    
    float groundDistance = altitude - groundPlaneHeight_;
    
    Point2D worldCoord;
    worldCoord.x = groundDistance * std::tan(angleX);
    worldCoord.y = groundDistance * std::tan(angleY);
    
    return worldCoord;
}

void AerialInference::setCompressionQuality(float quality) {
    compressionQuality_ = std::clamp(quality, 0.1f, 1.0f);
}

Tensor AerialInference::applyPerspectiveCorrection(const Tensor& image, const IMUData& imu) {
    auto shape = image.shape();
    int height = shape[0];
    int width = shape[1];
    
    Tensor corrected(shape);
    
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    
    float cosRoll = std::cos(imu.roll);
    float sinRoll = std::sin(imu.roll);
    float cosPitch = std::cos(imu.pitch);
    float sinPitch = std::sin(imu.pitch);
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float dx = x - cx;
            float dy = y - cy;
            
            float x_rot = dx * cosRoll - dy * sinRoll;
            float y_rot = dx * sinRoll + dy * cosRoll;
            
            float x_persp = x_rot;
            float y_persp = y_rot * cosPitch - focalLength_ * sinPitch;
            float z_persp = y_rot * sinPitch + focalLength_ * cosPitch;
            
            if (z_persp > 0) {
                float srcX = (x_persp * focalLength_ / z_persp) + cx;
                float srcY = (y_persp * focalLength_ / z_persp) + cy;
                
                if (srcX >= 0 && srcX < width - 1 && srcY >= 0 && srcY < height - 1) {
                    int x0 = static_cast<int>(srcX);
                    int y0 = static_cast<int>(srcY);
                    float fx = srcX - x0;
                    float fy = srcY - y0;
                    
                    for (int c = 0; c < 3; ++c) {
                        float v00 = image({y0, x0, c});
                        float v01 = image({y0, x0 + 1, c});
                        float v10 = image({y0 + 1, x0, c});
                        float v11 = image({y0 + 1, x0 + 1, c});
                        
                        float v = (1 - fx) * (1 - fy) * v00 +
                                  fx * (1 - fy) * v01 +
                                  (1 - fx) * fy * v10 +
                                  fx * fy * v11;
                        
                        corrected({y, x, c}) = v;
                    }
                }
            }
        }
    }
    
    return corrected;
}

Tensor AerialInference::applyMotionBlurCompensation(const Tensor& image, const IMUData& imu) {
    float blurKernelSize = std::min(5.0f, std::abs(imu.gyroX) + std::abs(imu.gyroY));
    
    if (blurKernelSize < 1.0f) {
        return image;
    }
    
    return enhanceContrast(image);
}

float AerialInference::calculateGroundSampleDistance(float altitude) {
    float sensorPixelSize = sensorWidth_ / droneImageWidth_;
    return (altitude * sensorPixelSize) / focalLength_;
}

void AerialInference::adjustDetectionsForAltitude(std::vector<Detection>& detections, float altitude) {
    float scaleFactor = altitude / 50.0f;
    
    for (auto& det : detections) {
        det.width *= scaleFactor;
        det.height *= scaleFactor;
        
        det.confidence *= std::exp(-0.01f * (altitude - 50.0f));
        det.confidence = std::clamp(det.confidence, 0.0f, 1.0f);
    }
}

Tensor AerialInference::preprocessForAerial(const Tensor& image) {
    // Skip contrast enhancement for speed - normalize directly
    auto shape = image.shape();
    Tensor normalized(shape);
    const float* src = image.data_float();
    float* dst = normalized.data_float();
    size_t total = shape[0] * shape[1] * shape[2];
    
    // Vectorized normalization
    #pragma omp parallel for simd
    for (size_t i = 0; i < total; ++i) {
        dst[i] = src[i] / 255.0f;
    }
    
    return normalized;
}

float AerialInference::estimateMotionBlur(const IMUData& current, const IMUData& previous) {
    float dt = std::chrono::duration<float>(current.timestamp - previous.timestamp).count();
    
    float angularVelocity = std::sqrt(
        std::pow(current.gyroX - previous.gyroX, 2) +
        std::pow(current.gyroY - previous.gyroY, 2) +
        std::pow(current.gyroZ - previous.gyroZ, 2)
    ) / dt;
    
    float linearAccel = std::sqrt(
        std::pow(current.accelX, 2) +
        std::pow(current.accelY, 2) +
        std::pow(current.accelZ, 2)
    );
    
    return angularVelocity * 0.5f + linearAccel * 0.1f;
}

void AerialInference::filterSmallObjects(std::vector<Detection>& detections, float minSize) {
    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
            [minSize](const Detection& det) {
                return det.width < minSize || det.height < minSize;
            }),
        detections.end()
    );
}

Tensor AerialInference::enhanceContrast(const Tensor& image) {
    auto shape = image.shape();
    Tensor enhanced(shape);
    
    std::vector<float> histogram(256, 0);
    int totalPixels = shape[0] * shape[1];
    
    for (int y = 0; y < shape[0]; ++y) {
        for (int x = 0; x < shape[1]; ++x) {
            float gray = 0.299f * image({y, x, 0}) +
                        0.587f * image({y, x, 1}) +
                        0.114f * image({y, x, 2});
            int bin = std::clamp(static_cast<int>(gray), 0, 255);
            histogram[bin]++;
        }
    }
    
    std::vector<float> cumulative(256);
    cumulative[0] = histogram[0] / totalPixels;
    for (int i = 1; i < 256; ++i) {
        cumulative[i] = cumulative[i-1] + histogram[i] / totalPixels;
    }
    
    for (int y = 0; y < shape[0]; ++y) {
        for (int x = 0; x < shape[1]; ++x) {
            for (int c = 0; c < 3; ++c) {
                float value = image({y, x, c});
                int bin = std::clamp(static_cast<int>(value), 0, 255);
                enhanced({y, x, c}) = cumulative[bin] * 255.0f;
            }
        }
    }
    
    return enhanced;
}

std::pair<float, float> AerialInference::calculateFieldOfView() {
    float fovX = 2.0f * std::atan(sensorWidth_ / (2.0f * focalLength_)) * 180.0f / M_PI;
    float fovY = 2.0f * std::atan(sensorHeight_ / (2.0f * focalLength_)) * 180.0f / M_PI;
    return {fovX, fovY};
}

std::vector<Detection> AerialInference::extractUltraDetections(
    const models::DetectionOutput& output, float conf_threshold) {
    std::vector<Detection> detections;
    
    // TACSNetUltra outputs are optimized for speed with efficient format
    const auto& bbox_preds = output.bbox_predictions;
    const auto& obj_scores = output.objectness_scores;
    const auto& class_preds = output.class_predictions;
    
    auto bbox_shape = bbox_preds.shape();
    int num_anchors = bbox_shape[1];
    int grid_h = bbox_shape[2];
    int grid_w = bbox_shape[3];
    
    // Ultra-fast detection extraction with SIMD-friendly loop structure
    for (int a = 0; a < num_anchors; ++a) {
        for (int y = 0; y < grid_h; ++y) {
            for (int x = 0; x < grid_w; ++x) {
                float objectness = obj_scores({0, a, y, x, 0});
                
                // Early exit for low confidence
                if (objectness < conf_threshold) continue;
                
                // Get class probabilities (3 classes: car, pedestrian, cyclist)
                float max_class_prob = 0.0f;
                int best_class = 0;
                for (int c = 0; c < 3; ++c) {
                    float prob = class_preds({0, a, y, x, c});
                    if (prob > max_class_prob) {
                        max_class_prob = prob;
                        best_class = c;
                    }
                }
                
                float combined_conf = objectness * max_class_prob;
                if (combined_conf < conf_threshold) continue;
                
                // Extract bounding box (already in pixel coordinates for TACSNetUltra)
                Detection det;
                det.x = bbox_preds({0, a, y, x, 0});
                det.y = bbox_preds({0, a, y, x, 1});
                det.width = bbox_preds({0, a, y, x, 2});
                det.height = bbox_preds({0, a, y, x, 3});
                det.confidence = combined_conf;
                det.class_id = best_class;
                det.class_prob = max_class_prob;
                
                detections.push_back(det);
            }
        }
    }
    
    return detections;
}

void AerialInference::processFastMode(const uint8_t* imageData, const IMUData& imuData) {
    // Ultra-optimized fast path for TACSNetUltra
    const int targetSize = 256;  // Fixed size for ultra-fast processing
    
    // Initialize workspace if needed
    if (!workspaceInitialized_) {
        workspaceImage_ = Tensor({targetSize, targetSize, 3});
        workspaceBatch_ = Tensor({1, 3, targetSize, targetSize});
        workspaceInitialized_ = true;
    }
    
    // Direct downscale and normalize in one pass
    float scaleX = static_cast<float>(droneImageWidth_) / targetSize;
    float scaleY = static_cast<float>(droneImageHeight_) / targetSize;
    
    float* dstBatch = workspaceBatch_.data_float();
    
    // Combined downscale, normalize, and transpose in single loop
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < targetSize; ++y) {
        for (int x = 0; x < targetSize; ++x) {
            int srcX = static_cast<int>(x * scaleX);
            int srcY = static_cast<int>(y * scaleY);
            int srcIdx = (srcY * droneImageWidth_ + srcX) * 3;
            
            // Direct placement in CHW format with normalization
            dstBatch[0 * targetSize * targetSize + y * targetSize + x] = imageData[srcIdx + 0] / 255.0f;
            dstBatch[1 * targetSize * targetSize + y * targetSize + x] = imageData[srcIdx + 1] / 255.0f;
            dstBatch[2 * targetSize * targetSize + y * targetSize + x] = imageData[srcIdx + 2] / 255.0f;
        }
    }
    
    // Ultra-fast inference with TACSNetUltra
    auto ultraOutputs = tacsNetUltra_->forward(workspaceBatch_);
    
    std::vector<Detection> detections;
    float fastModeConfThreshold = 0.7f;  // Higher threshold for fewer detections
    
    for (const auto& output : ultraOutputs) {
        auto ultraDets = extractUltraDetections(output, fastModeConfThreshold);
        
        // Scale detections back to original image size
        for (auto& det : ultraDets) {
            det.x *= scaleX;
            det.y *= scaleY;
            det.width *= scaleX;
            det.height *= scaleY;
        }
        
        detections.insert(detections.end(), ultraDets.begin(), ultraDets.end());
    }
    
    // Minimal post-processing for speed
    adjustDetectionsForAltitude(detections, droneAltitude_);
    
    // Update latest detections
    {
        std::lock_guard<std::mutex> lock(detectionMutex_);
        latestDetections_.clear();
        
        for (const auto& det : detections) {
            AerialDetection aerialDet;
            aerialDet.x = det.x;
            aerialDet.y = det.y;
            aerialDet.w = det.width;
            aerialDet.h = det.height;
            aerialDet.confidence = det.confidence;
            aerialDet.classId = det.class_id;
            aerialDet.worldCoordinate = pixelToWorldCoordinate(det.x + det.width/2, 
                                                              det.y + det.height/2, 
                                                              droneAltitude_);
            aerialDet.altitude = droneAltitude_;
            latestDetections_.push_back(aerialDet);
        }
    }
}

// DroneToGroundComm implementation

DroneToGroundComm::DroneToGroundComm(int droneId, const std::string& groundStationAddress)
    : droneId_(droneId)
    , groundStationAddress_(groundStationAddress)
    , compressionLevel_(6)
    , encryptionEnabled_(false)
    , bandwidthUsage_(0.0f) {
    
    isConnected_ = true;
    shouldStop_ = false;
    commThread_ = std::thread(&DroneToGroundComm::communicationLoop, this);
}

DroneToGroundComm::~DroneToGroundComm() {
    shouldStop_ = true;
    if (commThread_.joinable()) {
        commThread_.join();
    }
}

void DroneToGroundComm::sendDetections(const std::vector<AerialDetection>& detections) {
    GroundMessage msg;
    msg.type = GroundMessage::DETECTION_BATCH;
    msg.droneId = droneId_;
    msg.payload = serializeDetections(detections);
    msg.timestamp = std::chrono::steady_clock::now();
    
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        messageQueue_.push(msg);
    }
}

void DroneToGroundComm::sendCoverageUpdate(const VoronoiCell& coverage) {
    GroundMessage msg;
    msg.type = GroundMessage::COVERAGE_UPDATE;
    msg.droneId = droneId_;
    
    msg.payload.resize(sizeof(float) * (2 + coverage.vertices.size() * 2 + 1));
    size_t offset = 0;
    
    memcpy(msg.payload.data() + offset, &coverage.center.x, sizeof(float));
    offset += sizeof(float);
    memcpy(msg.payload.data() + offset, &coverage.center.y, sizeof(float));
    offset += sizeof(float);
    
    size_t vertexCount = coverage.vertices.size();
    memcpy(msg.payload.data() + offset, &vertexCount, sizeof(size_t));
    offset += sizeof(size_t);
    
    for (const auto& vertex : coverage.vertices) {
        memcpy(msg.payload.data() + offset, &vertex.x, sizeof(float));
        offset += sizeof(float);
        memcpy(msg.payload.data() + offset, &vertex.y, sizeof(float));
        offset += sizeof(float);
    }
    
    memcpy(msg.payload.data() + offset, &coverage.area, sizeof(float));
    
    msg.timestamp = std::chrono::steady_clock::now();
    
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        messageQueue_.push(msg);
    }
}

void DroneToGroundComm::sendBatteryAlert(float batteryLevel, float estimatedFlightTime) {
    GroundMessage msg;
    msg.type = GroundMessage::BATTERY_ALERT;
    msg.droneId = droneId_;
    msg.payload.resize(2 * sizeof(float));
    
    memcpy(msg.payload.data(), &batteryLevel, sizeof(float));
    memcpy(msg.payload.data() + sizeof(float), &estimatedFlightTime, sizeof(float));
    
    msg.timestamp = std::chrono::steady_clock::now();
    
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        messageQueue_.push(msg);
    }
}

void DroneToGroundComm::streamCompressedImage(const Tensor& image, float quality) {
    GroundMessage msg;
    msg.type = GroundMessage::IMAGE_STREAM;
    msg.droneId = droneId_;
    msg.payload = compressImage(image, quality);
    msg.timestamp = std::chrono::steady_clock::now();
    
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        messageQueue_.push(msg);
    }
}

void DroneToGroundComm::sendTelemetry(const IMUData& imu, const Point2D& position, float altitude) {
    GroundMessage msg;
    msg.type = GroundMessage::TELEMETRY;
    msg.droneId = droneId_;
    
    size_t dataSize = sizeof(IMUData) + sizeof(Point2D) + sizeof(float);
    msg.payload.resize(dataSize);
    
    size_t offset = 0;
    memcpy(msg.payload.data() + offset, &imu, sizeof(IMUData));
    offset += sizeof(IMUData);
    memcpy(msg.payload.data() + offset, &position, sizeof(Point2D));
    offset += sizeof(Point2D);
    memcpy(msg.payload.data() + offset, &altitude, sizeof(float));
    
    msg.timestamp = std::chrono::steady_clock::now();
    
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        messageQueue_.push(msg);
    }
}

void DroneToGroundComm::communicationLoop() {
    // Initialize V2X communication socket for production deployment
    int sockfd = -1;
    struct sockaddr_in groundAddr;
    
    // Parse ground station address
    size_t colonPos = groundStationAddress_.find(':');
    std::string ipAddr = groundStationAddress_.substr(0, colonPos);
    int port = 8080;
    if (colonPos != std::string::npos) {
        port = std::stoi(groundStationAddress_.substr(colonPos + 1));
    }
    
    // Create UDP socket for V2X communication (DSRC/C-V2X compatible)
    sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sockfd < 0) {
        isConnected_ = false;
    } else {
        // Configure socket for low-latency V2X communication
        int optval = 1;
        setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
        
        // Set socket to non-blocking for real-time requirements
        int flags = fcntl(sockfd, F_GETFL, 0);
        fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);
        
        // Configure ground station address
        memset(&groundAddr, 0, sizeof(groundAddr));
        groundAddr.sin_family = AF_INET;
        groundAddr.sin_port = htons(port);
        inet_pton(AF_INET, ipAddr.c_str(), &groundAddr.sin_addr);
        
        isConnected_ = true;
    }
    
    while (!shouldStop_) {
        std::vector<GroundMessage> messagesToSend;
        
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            while (!messageQueue_.empty()) {
                messagesToSend.push_back(messageQueue_.front());
                messageQueue_.pop();
            }
        }
        
        for (const auto& msg : messagesToSend) {
            std::vector<uint8_t> data = msg.payload;
            
            if (compressionLevel_ > 0) {
                data = compressData(data);
            }
            
            if (encryptionEnabled_) {
                data = encryptData(data);
            }
            
            // V2X-compliant message structure
            std::vector<uint8_t> v2xPacket;
            v2xPacket.reserve(data.size() + 32);
            
            // V2X message header (WAVE/DSRC BSM-compatible)
            struct V2XMessageHeader {
                uint8_t protocolVersion = 0x03;  // V2X protocol version
                uint8_t messageType;              // Message type from GroundMessage::Type
                uint32_t sourceID;                // Drone ID
                uint32_t sequenceNumber;          // Message sequence
                uint64_t timestamp;               // Microsecond timestamp
                uint16_t payloadLength;           // Payload size
                uint16_t checksum;                // Header checksum
            } header;
            
            static std::atomic<uint32_t> seqNum{0};
            header.messageType = static_cast<uint8_t>(msg.type);
            header.sourceID = static_cast<uint32_t>(droneId_);
            header.sequenceNumber = seqNum.fetch_add(1);
            header.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                msg.timestamp.time_since_epoch()).count();
            header.payloadLength = static_cast<uint16_t>(data.size());
            
            // Calculate header checksum
            uint16_t checksum = 0;
            const uint8_t* headerBytes = reinterpret_cast<const uint8_t*>(&header);
            for (size_t i = 0; i < sizeof(header) - sizeof(header.checksum); ++i) {
                checksum += headerBytes[i];
            }
            header.checksum = checksum;
            
            // Assemble V2X packet
            v2xPacket.insert(v2xPacket.end(), headerBytes, headerBytes + sizeof(header));
            v2xPacket.insert(v2xPacket.end(), data.begin(), data.end());
            
            // Transmit via V2X protocol
            if (isConnected_ && sockfd >= 0) {
                ssize_t bytesSent = sendto(sockfd, v2xPacket.data(), v2xPacket.size(), 
                                          MSG_DONTWAIT, 
                                          reinterpret_cast<struct sockaddr*>(&groundAddr), 
                                          sizeof(groundAddr));
                
                if (bytesSent > 0) {
                    updateBandwidthMetrics(bytesSent);
                } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    // Handle transmission error
                    isConnected_ = false;
                }
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Clean up socket
    if (sockfd >= 0) {
        close(sockfd);
    }
}

std::vector<uint8_t> DroneToGroundComm::compressData(const std::vector<uint8_t>& data) {
    // LZ77-based compression for efficient data transmission
    std::vector<uint8_t> compressed;
    compressed.reserve(data.size() * 0.8f); // Expect ~20% compression
    
    const int windowSize = 4096;
    const int lookAheadSize = 18;
    
    // Add header with original size
    uint32_t originalSize = data.size();
    compressed.resize(4);
    memcpy(compressed.data(), &originalSize, 4);
    
    size_t pos = 0;
    while (pos < data.size()) {
        int bestMatchDist = 0;
        int bestMatchLen = 0;
        
        // Search for matching pattern in sliding window
        int searchStart = std::max(0, static_cast<int>(pos) - windowSize);
        for (int i = searchStart; i < static_cast<int>(pos); ++i) {
            int matchLen = 0;
            while (matchLen < lookAheadSize && 
                   pos + matchLen < data.size() && 
                   data[i + matchLen] == data[pos + matchLen]) {
                matchLen++;
            }
            
            if (matchLen > bestMatchLen) {
                bestMatchLen = matchLen;
                bestMatchDist = pos - i;
            }
        }
        
        if (bestMatchLen >= 3) {
            // Encode as (distance, length) pair
            compressed.push_back(0x80 | (bestMatchDist >> 8));
            compressed.push_back(bestMatchDist & 0xFF);
            compressed.push_back(bestMatchLen - 3);
            pos += bestMatchLen;
        } else {
            // Encode as literal
            compressed.push_back(data[pos]);
            pos++;
        }
    }
    
    return compressed;
}

std::vector<uint8_t> DroneToGroundComm::encryptData(const std::vector<uint8_t>& data) {
    // ChaCha20-style stream cipher for secure communication
    std::vector<uint8_t> encrypted;
    encrypted.reserve(data.size() + 12); // IV + data
    
    // Generate IV using high-resolution time and additional entropy
    uint8_t iv[12] = {0};
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    
    // Mix multiple entropy sources for IV
    uint64_t ivData = nanos ^ (static_cast<uint64_t>(droneId_) << 32);
    ivData ^= reinterpret_cast<uintptr_t>(&data[0]);  // Stack address entropy
    
    memcpy(iv, &ivData, sizeof(ivData));
    
    // Add message counter for uniqueness
    static std::atomic<uint32_t> msgCounter{0};
    uint32_t counter = msgCounter.fetch_add(1);
    memcpy(iv + 8, &counter, sizeof(counter));
    
    // Add IV to encrypted data
    encrypted.insert(encrypted.end(), iv, iv + 12);
    
    // Initialize ChaCha20 state
    uint32_t state[16] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,  // Constants
        0, 0, 0, 0,  // Key (will be derived below)
        0, 0, 0, 0,  // Key continued
        0, 0, 0, 0  // Counter + IV
    };
    
    // Derive key from drone ID and ground station address hash
    std::hash<std::string> hasher;
    uint64_t addrHash = hasher(groundStationAddress_);
    uint64_t idHash = static_cast<uint64_t>(droneId_) * 0x9E3779B97F4A7C15ULL;
    
    // Mix drone ID and address hash for key material
    state[4] = static_cast<uint32_t>(addrHash);
    state[5] = static_cast<uint32_t>(addrHash >> 32);
    state[6] = static_cast<uint32_t>(idHash);
    state[7] = static_cast<uint32_t>(idHash >> 32);
    
    // Additional key material from timestamp
    auto keyTime = std::chrono::steady_clock::now().time_since_epoch().count();
    state[8] = static_cast<uint32_t>(keyTime);
    state[9] = static_cast<uint32_t>(keyTime >> 32);
    state[10] = static_cast<uint32_t>(~keyTime);
    state[11] = static_cast<uint32_t>(~keyTime >> 32);
    
    // Copy IV to state
    memcpy(&state[13], iv, 12);
    
    // Generate keystream and encrypt
    for (size_t i = 0; i < data.size(); i += 64) {
        uint32_t workingState[16];
        memcpy(workingState, state, sizeof(state));
        
        // ChaCha20 quarter round
        auto quarterRound = [](uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
            a += b; d ^= a; d = (d << 16) | (d >> 16);
            c += d; b ^= c; b = (b << 12) | (b >> 20);
            a += b; d ^= a; d = (d << 8) | (d >> 24);
            c += d; b ^= c; b = (b << 7) | (b >> 25);
        };
        
        // 20 rounds (10 double rounds)
        for (int round = 0; round < 10; ++round) {
            // Column rounds
            quarterRound(workingState[0], workingState[4], workingState[8], workingState[12]);
            quarterRound(workingState[1], workingState[5], workingState[9], workingState[13]);
            quarterRound(workingState[2], workingState[6], workingState[10], workingState[14]);
            quarterRound(workingState[3], workingState[7], workingState[11], workingState[15]);
            // Diagonal rounds
            quarterRound(workingState[0], workingState[5], workingState[10], workingState[15]);
            quarterRound(workingState[1], workingState[6], workingState[11], workingState[12]);
            quarterRound(workingState[2], workingState[7], workingState[8], workingState[13]);
            quarterRound(workingState[3], workingState[4], workingState[9], workingState[14]);
        }
        
        // Add original state
        for (int j = 0; j < 16; ++j) {
            workingState[j] += state[j];
        }
        
        // XOR with data
        uint8_t* keystream = reinterpret_cast<uint8_t*>(workingState);
        for (size_t j = 0; j < 64 && i + j < data.size(); ++j) {
            encrypted.push_back(data[i + j] ^ keystream[j]);
        }
        
        // Increment counter
        state[12]++;
    }
    
    return encrypted;
}

void DroneToGroundComm::updateBandwidthMetrics(size_t bytesSent) {
    static auto lastUpdate = std::chrono::steady_clock::now();
    static size_t totalBytes = 0;
    
    totalBytes += bytesSent;
    
    auto now = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(now - lastUpdate).count();
    
    if (elapsed >= 1.0f) {
        bandwidthUsage_ = totalBytes / elapsed;
        totalBytes = 0;
        lastUpdate = now;
    }
}

std::vector<uint8_t> DroneToGroundComm::serializeDetections(const std::vector<AerialDetection>& detections) {
    size_t detectionSize = sizeof(AerialDetection);
    std::vector<uint8_t> data(sizeof(size_t) + detections.size() * detectionSize);
    
    size_t count = detections.size();
    memcpy(data.data(), &count, sizeof(size_t));
    
    size_t offset = sizeof(size_t);
    for (const auto& det : detections) {
        memcpy(data.data() + offset, &det, detectionSize);
        offset += detectionSize;
    }
    
    return data;
}

std::vector<uint8_t> DroneToGroundComm::compressImage(const Tensor& image, float quality) {
    // DCT-based image compression (JPEG-style)
    auto shape = image.shape();
    int height = shape[0];
    int width = shape[1];
    
    std::vector<uint8_t> compressed;
    compressed.reserve(height * width); // Conservative estimate
    
    // Header: dimensions and quality
    compressed.resize(12);
    memcpy(compressed.data(), &width, 4);
    memcpy(compressed.data() + 4, &height, 4);
    memcpy(compressed.data() + 8, &quality, 4);
    
    // Quantization matrix scaled by quality
    const int baseQuant[64] = {
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    };
    
    int quantScale = static_cast<int>((1.0f - quality) * 50.0f + 1.0f);
    
    // Process 8x8 blocks
    for (int by = 0; by < height; by += 8) {
        for (int bx = 0; bx < width; bx += 8) {
            // For each color channel
            for (int ch = 0; ch < 3; ++ch) {
                float block[64];
                
                // Extract 8x8 block
                for (int y = 0; y < 8; ++y) {
                    for (int x = 0; x < 8; ++x) {
                        int py = std::min(by + y, height - 1);
                        int px = std::min(bx + x, width - 1);
                        block[y * 8 + x] = image({py, px, ch}) - 128.0f;
                    }
                }
                
                // Apply 2D DCT
                float dctBlock[64];
                for (int u = 0; u < 8; ++u) {
                    for (int v = 0; v < 8; ++v) {
                        float sum = 0.0f;
                        for (int y = 0; y < 8; ++y) {
                            for (int x = 0; x < 8; ++x) {
                                sum += block[y * 8 + x] * 
                                      std::cos((2 * x + 1) * u * M_PI / 16.0f) *
                                      std::cos((2 * y + 1) * v * M_PI / 16.0f);
                            }
                        }
                        float cu = (u == 0) ? 1.0f / std::sqrt(2.0f) : 1.0f;
                        float cv = (v == 0) ? 1.0f / std::sqrt(2.0f) : 1.0f;
                        dctBlock[v * 8 + u] = 0.25f * cu * cv * sum;
                    }
                }
                
                // Quantize and zigzag scan
                const int zigzag[64] = {
                    0,  1,  8, 16,  9,  2,  3, 10,
                    17, 24, 32, 25, 18, 11,  4,  5,
                    12, 19, 26, 33, 40, 48, 41, 34,
                    27, 20, 13,  6,  7, 14, 21, 28,
                    35, 42, 49, 56, 57, 50, 43, 36,
                    29, 22, 15, 23, 30, 37, 44, 51,
                    58, 59, 52, 45, 38, 31, 39, 46,
                    53, 60, 61, 54, 47, 55, 62, 63
                };
                
                std::vector<int16_t> quantized;
                int lastNonZero = -1;
                for (int i = 0; i < 64; ++i) {
                    int pos = zigzag[i];
                    int quant = baseQuant[pos] * quantScale;
                    int value = static_cast<int>(std::round(dctBlock[pos] / quant));
                    if (value != 0) lastNonZero = i;
                    quantized.push_back(value);
                }
                
                // Encode using run-length encoding
                compressed.push_back(lastNonZero + 1); // Number of coefficients
                for (int i = 0; i <= lastNonZero; ++i) {
                    int16_t val = quantized[i];
                    compressed.push_back((val >> 8) & 0xFF);
                    compressed.push_back(val & 0xFF);
                }
            }
        }
    }
    
    return compressed;
}

// AdaptiveResolution implementation

AdaptiveResolution::AdaptiveResolution(int baseWidth, int baseHeight)
    : baseWidth_(baseWidth)
    , baseHeight_(baseHeight)
    , minWidth_(160)
    , minHeight_(120)
    , maxWidth_(1920)
    , maxHeight_(1080)
    , qualityScore_(1.0f)
    , targetLatency_(50.0f) {
    
    initializeProfiles();
}

std::pair<int, int> AdaptiveResolution::computeOptimalResolution(
    float processingLoad, float batteryLevel, float bandwidth) {
    
    float score = processingLoad * 0.4f + (1.0f - batteryLevel) * 0.3f + 
                  (1.0f - bandwidth / 10000000.0f) * 0.3f;
    
    int selectedWidth = baseWidth_;
    int selectedHeight = baseHeight_;
    float bestScore = std::numeric_limits<float>::max();
    
    for (const auto& profile : profiles_) {
        float latency = predictLatency(profile.width, profile.height);
        float accuracy = predictAccuracy(profile.width, profile.height);
        
        // Higher score means we need lower resolution
        float resolutionPenalty = (profile.width * profile.height) / (1920.0f * 1080.0f);
        
        float profileScore = std::abs(latency - targetLatency_) / targetLatency_ +
                           (1.0f - accuracy) * 2.0f +
                           score * resolutionPenalty;
        
        if (profileScore < bestScore) {
            bestScore = profileScore;
            selectedWidth = profile.width;
            selectedHeight = profile.height;
        }
    }
    
    return {selectedWidth, selectedHeight};
}

void AdaptiveResolution::updateMetrics(float latency, float accuracy) {
    qualityScore_ = qualityScore_ * 0.9f + (accuracy * (targetLatency_ / latency)) * 0.1f;
    qualityScore_ = std::clamp(qualityScore_, 0.0f, 1.0f);
}

void AdaptiveResolution::setMinResolution(int width, int height) {
    minWidth_ = width;
    minHeight_ = height;
    initializeProfiles();
}

void AdaptiveResolution::setMaxResolution(int width, int height) {
    maxWidth_ = width;
    maxHeight_ = height;
    initializeProfiles();
}

void AdaptiveResolution::initializeProfiles() {
    profiles_.clear();
    
    std::vector<std::pair<int, int>> resolutions = {
        {160, 120}, {320, 240}, {416, 416}, {608, 608},
        {640, 480}, {800, 600}, {1024, 768}, {1280, 720},
        {1920, 1080}
    };
    
    for (const auto& [w, h] : resolutions) {
        if (w >= minWidth_ && h >= minHeight_ && w <= maxWidth_ && h <= maxHeight_) {
            ResolutionProfile profile;
            profile.width = w;
            profile.height = h;
            profile.expectedLatency = predictLatency(w, h);
            profile.expectedAccuracy = predictAccuracy(w, h);
            profiles_.push_back(profile);
        }
    }
}

float AdaptiveResolution::predictLatency(int width, int height) {
    float pixels = width * height;
    float baseLatency = 10.0f;
    return baseLatency + (pixels / 1000000.0f) * 25.0f;
}

float AdaptiveResolution::predictAccuracy(int width, int height) {
    float minPixels = minWidth_ * minHeight_;
    float maxPixels = maxWidth_ * maxHeight_;
    float pixels = width * height;
    
    float normalized = (pixels - minPixels) / (maxPixels - minPixels);
    return 0.7f + 0.25f * std::sqrt(normalized);
}

} // namespace tacs