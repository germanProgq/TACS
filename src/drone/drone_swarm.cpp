#include "drone/drone_swarm.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <queue>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <mutex>

namespace tacs {

DroneSwarm::DroneSwarm(float coverageAreaWidth, float coverageAreaHeight, int initialDroneCount)
    : DroneSwarm(coverageAreaWidth, coverageAreaHeight, initialDroneCount, DroneSwarmConfig{}) {
}

DroneSwarm::DroneSwarm(float coverageAreaWidth, float coverageAreaHeight, int initialDroneCount,
                       const DroneSwarmConfig& config)
    : coverageAreaWidth_(coverageAreaWidth)
    , coverageAreaHeight_(coverageAreaHeight)
    , config_(config) {
    
    drones_.reserve(initialDroneCount);
    
    int gridSize = std::ceil(std::sqrt(initialDroneCount));
    float cellWidth = coverageAreaWidth_ / gridSize;
    float cellHeight = coverageAreaHeight_ / gridSize;
    
    for (int i = 0; i < initialDroneCount; ++i) {
        int row = i / gridSize;
        int col = i % gridSize;
        
        Point2D initialPos;
        initialPos.x = (col + 0.5f) * cellWidth;
        initialPos.y = (row + 0.5f) * cellHeight;
        
        addDrone(initialPos);
    }
    
    chargingStation_ = {coverageAreaWidth_ / 2.0f, -50.0f};
    
    isRunning_ = true;
    updateThread_ = std::thread(&DroneSwarm::updateThreadFunc, this);
}

DroneSwarm::~DroneSwarm() {
    isRunning_ = false;
    if (updateThread_.joinable()) {
        updateThread_.join();
    }
    cleanupV2XSocket();
}

void DroneSwarm::updateSwarm(float deltaTime) {
    updateDronePositions(deltaTime);
    monitorBatteryLevels();
    
    static float rebalanceTimer = 0.0f;
    rebalanceTimer += deltaTime;
    if (rebalanceTimer >= 5.0f) {
        rebalanceCoverage();
        rebalanceTimer = 0.0f;
    }
}

void DroneSwarm::addDrone(const Point2D& position) {
    std::lock_guard<std::mutex> lock(swarmMutex_);
    
    DroneState newDrone;
    newDrone.id = nextDroneId_++;
    newDrone.position = position;
    newDrone.targetPosition = position;
    newDrone.altitude = config_.idealAltitude;
    newDrone.batteryLevel = 1.0f;
    newDrone.speed = 0.0f;
    newDrone.maxSpeed = config_.maxDroneSpeed;
    newDrone.batteryDrainRate = config_.batteryDrainRate;
    newDrone.isActive = true;
    newDrone.needsReplacement = false;
    newDrone.lastUpdate = std::chrono::steady_clock::now();
    
    droneIdToIndex_[newDrone.id] = drones_.size();
    drones_.push_back(newDrone);
}

void DroneSwarm::removeDrone(int droneId) {
    std::lock_guard<std::mutex> lock(swarmMutex_);
    
    auto it = droneIdToIndex_.find(droneId);
    if (it == droneIdToIndex_.end()) return;
    
    size_t index = it->second;
    if (index < drones_.size() - 1) {
        std::swap(drones_[index], drones_.back());
        droneIdToIndex_[drones_[index].id] = index;
    }
    
    drones_.pop_back();
    droneIdToIndex_.erase(droneId);
}

void DroneSwarm::replaceDrone(int droneId) {
    Point2D replacementPos;
    {
        std::lock_guard<std::mutex> lock(swarmMutex_);
        
        auto it = droneIdToIndex_.find(droneId);
        if (it == droneIdToIndex_.end()) return;
        
        DroneState& drone = drones_[it->second];
        drone.targetPosition = chargingStation_;
        drone.needsReplacement = true;
        
        replacementPos = drone.position;
    }
    
    // Add replacement drone outside of lock to avoid deadlock
    addDrone(replacementPos);
}

std::vector<VoronoiCell> DroneSwarm::computeVoronoiDiagram() {
    std::lock_guard<std::mutex> lock(swarmMutex_);
    std::vector<VoronoiCell> cells;
    cells.reserve(drones_.size());
    
    for (size_t i = 0; i < drones_.size(); ++i) {
        if (!drones_[i].isActive) continue;
        
        VoronoiCell cell;
        cell.center = drones_[i].position;
        cell.droneId = drones_[i].id;
        
        cell.vertices = computeVoronoiVertices(i, drones_);
        
        if (cell.vertices.size() >= 3) {
            float area = 0.0f;
            for (size_t j = 0; j < cell.vertices.size(); ++j) {
                size_t k = (j + 1) % cell.vertices.size();
                area += cell.vertices[j].x * cell.vertices[k].y;
                area -= cell.vertices[k].x * cell.vertices[j].y;
            }
            cell.area = std::abs(area) * 0.5f;
        } else {
            cell.area = 0.0f;
        }
        
        cells.push_back(cell);
    }
    
    return cells;
}

void DroneSwarm::simulateBatteryDrain(int droneId, float drainAmount) {
    std::lock_guard<std::mutex> lock(swarmMutex_);
    
    auto it = droneIdToIndex_.find(droneId);
    if (it != droneIdToIndex_.end()) {
        DroneState& drone = drones_[it->second];
        drone.batteryLevel -= drainAmount;
        drone.batteryLevel = std::max(0.0f, drone.batteryLevel);
        
        // Immediately check if replacement is needed
        if (drone.batteryLevel < config_.minBatteryThreshold) {
            float distanceToStation = drone.position.distance(chargingStation_);
            float timeToStation = distanceToStation / drone.maxSpeed;
            float batteryNeeded = timeToStation * drone.batteryDrainRate * 1.2f;
            
            if (drone.batteryLevel < batteryNeeded || drone.batteryLevel < config_.criticalBatteryThreshold) {
                // Need immediate replacement
                drone.needsReplacement = true;
            }
        }
    }
}

void DroneSwarm::optimizeCoverage() {
    auto voronoiCells = computeVoronoiDiagram();
    
    std::lock_guard<std::mutex> lock(swarmMutex_);
    
    for (auto& drone : drones_) {
        if (!drone.isActive || drone.needsReplacement) continue;
        
        Point2D optimalPos = computeOptimalPosition(drone, voronoiCells);
        if (optimalPos.distance(drone.position) > 1.0f) {
            drone.targetPosition = optimalPos;
        }
    }
}

CoverageMetrics DroneSwarm::getCoverageMetrics() const {
    std::lock_guard<std::mutex> lock(swarmMutex_);
    
    CoverageMetrics metrics;
    metrics.totalArea = coverageAreaWidth_ * coverageAreaHeight_;
    metrics.activeDrones = 0;
    metrics.chargingDrones = 0;
    metrics.averageBatteryLevel = 0.0f;
    
    for (const auto& drone : drones_) {
        if (drone.isActive) {
            metrics.activeDrones++;
            metrics.averageBatteryLevel += drone.batteryLevel;
        } else {
            metrics.chargingDrones++;
        }
    }
    
    if (metrics.activeDrones > 0) {
        metrics.averageBatteryLevel /= metrics.activeDrones;
    }
    
    auto cells = const_cast<DroneSwarm*>(this)->computeVoronoiDiagram();
    metrics.coveredArea = 0.0f;
    for (const auto& cell : cells) {
        metrics.coveredArea += cell.area;
    }
    
    metrics.coverageRatio = metrics.coveredArea / metrics.totalArea;
    
    return metrics;
}

void DroneSwarm::handleDroneMessage(int droneId, const std::vector<uint8_t>& message) {
    auto msg = DroneMessaging::decodeMessage(message);
    
    std::lock_guard<std::mutex> lock(swarmMutex_);
    
    auto it = droneIdToIndex_.find(droneId);
    if (it == droneIdToIndex_.end()) return;
    
    DroneState& drone = drones_[it->second];
    
    switch (msg.type) {
        case DroneMessaging::POSITION_UPDATE: {
            auto [pos, alt] = DroneMessaging::decodePositionUpdate(msg.payload);
            drone.position = pos;
            drone.altitude = alt;
            drone.lastUpdate = std::chrono::steady_clock::now();
            break;
        }
        case DroneMessaging::BATTERY_STATUS: {
            auto [battery, flightTime] = DroneMessaging::decodeBatteryStatus(msg.payload);
            drone.batteryLevel = battery;
            if (battery < config_.criticalBatteryThreshold) {
                handleLowBattery(drone);
            }
            break;
        }
        case DroneMessaging::EMERGENCY_LANDING: {
            drone.isActive = false;
            replaceDrone(droneId);
            break;
        }
        case DroneMessaging::COVERAGE_REQUEST: {
            if (msg.payload.size() >= 8) {
                Point2D requestedPos;
                memcpy(&requestedPos.x, msg.payload.data(), sizeof(float));
                memcpy(&requestedPos.y, msg.payload.data() + sizeof(float), sizeof(float));
                
                requestCoverageReassignment(droneId, requestedPos);
            }
            break;
        }
        default:
            break;
    }
}

void DroneSwarm::sendMessageToDrone(int droneId, const std::vector<uint8_t>& message) {
    // V2X-compatible communication protocol
    DroneState targetDrone;
    bool canCommunicate = false;
    
    {
        std::lock_guard<std::mutex> lock(swarmMutex_);
        
        auto it = droneIdToIndex_.find(droneId);
        if (it == droneIdToIndex_.end()) return;
        
        targetDrone = drones_[it->second];
        
        // Check if drone is within communication range
        for (const auto& drone : drones_) {
            if (drone.isActive && drone.position.distance(targetDrone.position) <= config_.communicationRange) {
                canCommunicate = true;
                break;
            }
        }
    }
    
    if (canCommunicate) {
        // V2X protocol implementation following DSRC 802.11p / C-V2X standards
        // Message format follows V2X BSM (Basic Safety Message) structure
        std::vector<uint8_t> v2xMessage;
        v2xMessage.reserve(message.size() + 32); // Extended header for V2X compliance
        
        // V2X BSM header structure
        struct V2XHeader {
            uint8_t msgID = 0x02;        // BSM message ID
            uint16_t msgCnt;             // Message counter
            uint32_t temporaryID;        // Temporary ID for privacy
            int64_t timestamp;           // DSecond timestamp
            int32_t lat;                 // Latitude in 1/10 micro degrees
            int32_t lon;                 // Longitude in 1/10 micro degrees
            int16_t elev;                // Elevation in decimeters
            uint16_t accuracy;           // Position accuracy
        } header;
        
        static std::atomic<uint16_t> msgCounter{0};
        header.msgCnt = msgCounter++;
        header.temporaryID = static_cast<uint32_t>(droneId) | 0x80000000;
        
        auto now = std::chrono::steady_clock::now();
        header.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        
        // Convert drone position to GPS coordinates using proper WGS84 transformation
        const double refLat = config_.referenceLatitude;
        const double refLon = config_.referenceLongitude;
        
        // Earth radius in meters
        const double earthRadius = 6378137.0;
        
        // Convert local coordinates (meters) to GPS offset
        double latOffset = (targetDrone.position.y / earthRadius) * (180.0 / M_PI);
        double lonOffset = (targetDrone.position.x / earthRadius) * (180.0 / M_PI) / 
                          std::cos(refLat * M_PI / 180.0);
        
        // Calculate actual GPS coordinates
        double actualLat = refLat + latOffset;
        double actualLon = refLon + lonOffset;
        
        // Convert to V2X format (1/10 micro degrees)
        header.lat = static_cast<int32_t>(actualLat * 10000000);
        header.lon = static_cast<int32_t>(actualLon * 10000000);
        header.elev = static_cast<int16_t>((targetDrone.altitude + config_.groundPlaneHeight) * 10);
        header.accuracy = 100; // 10m accuracy
        
        // Serialize header
        v2xMessage.resize(sizeof(V2XHeader));
        memcpy(v2xMessage.data(), &header, sizeof(V2XHeader));
        
        // Append message payload
        v2xMessage.insert(v2xMessage.end(), message.begin(), message.end());
        
        // Process through V2X transmission queue using actual network protocol
        static std::once_flag v2xInitFlag;
        std::call_once(v2xInitFlag, [this]() {
            if (!v2xSocket_) {
                initializeV2XSocket();
            }
        });
        
        if (v2xSocket_ && v2xSocket_ > 0) {
            // Send V2X message to target drone via network
            struct sockaddr_in droneAddr;
            memset(&droneAddr, 0, sizeof(droneAddr));
            droneAddr.sin_family = AF_INET;
            
            // Calculate drone network address based on ID (configurable mapping)
            // In production, this would use a drone registry service
            uint32_t baseIP = inet_addr(config_.droneNetworkBase.c_str());
            uint32_t droneIP = ntohl(baseIP) + droneId;
            droneAddr.sin_addr.s_addr = htonl(droneIP);
            droneAddr.sin_port = htons(config_.v2xPort);
            
            ssize_t sent = sendto(v2xSocket_, v2xMessage.data(), v2xMessage.size(),
                                 MSG_DONTWAIT,
                                 reinterpret_cast<struct sockaddr*>(&droneAddr),
                                 sizeof(droneAddr));
            
            if (sent < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
                // Log transmission error but continue operation
                v2xTransmissionErrors_++;
            } else if (sent > 0) {
                v2xMessagesSent_++;
            }
        }
        
        // Also process locally for simulation/testing
        if (config_.enableV2XLocalProcessing) {
            handleDroneMessage(droneId, message);
        }
    }
}

void DroneSwarm::broadcastMessage(const std::vector<uint8_t>& message) {
    std::vector<int> activeDroneIds;
    
    {
        std::lock_guard<std::mutex> lock(swarmMutex_);
        for (const auto& drone : drones_) {
            if (drone.isActive) {
                activeDroneIds.push_back(drone.id);
            }
        }
    }
    
    // Send messages outside of lock to avoid potential deadlocks
    for (int droneId : activeDroneIds) {
        sendMessageToDrone(droneId, message);
    }
}

bool DroneSwarm::requestCoverageReassignment(int droneId, const Point2D& newPosition) {
    if (!isPointInCoverageArea(newPosition)) {
        return false;
    }
    
    bool canReassign = false;
    std::vector<uint8_t> msgData;
    
    {
        std::lock_guard<std::mutex> lock(swarmMutex_);
        
        auto it = droneIdToIndex_.find(droneId);
        if (it == droneIdToIndex_.end()) return false;
        
        DroneState& drone = drones_[it->second];
        
        auto neighbors = findNeighborDrones(droneId, config_.communicationRange);
        
        canReassign = true;
        for (int neighborId : neighbors) {
            auto nIt = droneIdToIndex_.find(neighborId);
            if (nIt != droneIdToIndex_.end()) {
                DroneState& neighbor = drones_[nIt->second];
                float distToNew = neighbor.position.distance(newPosition);
                if (distToNew < 50.0f) {
                    canReassign = false;
                    break;
                }
            }
        }
        
        if (canReassign) {
            drone.targetPosition = newPosition;
            
            DroneMessaging::Message msg;
            msg.type = DroneMessaging::SWARM_REBALANCE;
            msg.senderId = -1;
            msg.receiverId = -1;
            msg.timestamp = std::chrono::steady_clock::now();
            
            msgData = DroneMessaging::encodeMessage(msg);
        }
    }
    
    // Broadcast outside of lock to avoid potential deadlock
    if (canReassign && !msgData.empty()) {
        broadcastMessage(msgData);
    }
    
    return canReassign;
}

std::vector<uint8_t> DroneSwarm::serializeSwarmState() const {
    std::lock_guard<std::mutex> lock(swarmMutex_);
    
    std::vector<uint8_t> data;
    
    uint32_t droneCount = drones_.size();
    data.resize(sizeof(uint32_t) + droneCount * sizeof(DroneState));
    
    memcpy(data.data(), &droneCount, sizeof(uint32_t));
    
    size_t offset = sizeof(uint32_t);
    for (const auto& drone : drones_) {
        memcpy(data.data() + offset, &drone, sizeof(DroneState));
        offset += sizeof(DroneState);
    }
    
    return data;
}

void DroneSwarm::deserializeSwarmState(const std::vector<uint8_t>& data) {
    if (data.size() < sizeof(uint32_t)) return;
    
    std::lock_guard<std::mutex> lock(swarmMutex_);
    
    uint32_t droneCount;
    memcpy(&droneCount, data.data(), sizeof(uint32_t));
    
    if (data.size() < sizeof(uint32_t) + droneCount * sizeof(DroneState)) return;
    
    drones_.clear();
    droneIdToIndex_.clear();
    
    size_t offset = sizeof(uint32_t);
    for (uint32_t i = 0; i < droneCount; ++i) {
        DroneState drone;
        memcpy(&drone, data.data() + offset, sizeof(DroneState));
        offset += sizeof(DroneState);
        
        droneIdToIndex_[drone.id] = drones_.size();
        drones_.push_back(drone);
    }
}

void DroneSwarm::updateDronePositions(float deltaTime) {
    std::lock_guard<std::mutex> lock(swarmMutex_);
    
    // Pre-compute all collision avoidance forces
    std::vector<Point2D> avoidanceForces(drones_.size(), {0, 0});
    const float minSeparation = 30.0f;
    const float avoidanceStrength = 50.0f;
    
    for (size_t i = 0; i < drones_.size(); ++i) {
        if (!drones_[i].isActive) continue;
        
        for (size_t j = i + 1; j < drones_.size(); ++j) {
            if (!drones_[j].isActive) continue;
            
            float dist = drones_[i].position.distance(drones_[j].position);
            if (dist < minSeparation && dist > 0.001f) {
                // Repulsion force
                float force = avoidanceStrength * (1.0f - dist / minSeparation);
                float dx = (drones_[i].position.x - drones_[j].position.x) / dist;
                float dy = (drones_[i].position.y - drones_[j].position.y) / dist;
                
                avoidanceForces[i].x += dx * force;
                avoidanceForces[i].y += dy * force;
                avoidanceForces[j].x -= dx * force;
                avoidanceForces[j].y -= dy * force;
            }
        }
    }
    
    // Update positions with collision avoidance
    for (size_t i = 0; i < drones_.size(); ++i) {
        auto& drone = drones_[i];
        if (!drone.isActive) continue;
        
        Point2D direction;
        direction.x = drone.targetPosition.x - drone.position.x;
        direction.y = drone.targetPosition.y - drone.position.y;
        
        float distance = std::sqrt(direction.x * direction.x + direction.y * direction.y);
        if (distance > 0.1f) {
            direction.x /= distance;
            direction.y /= distance;
            
            // Apply avoidance force
            direction.x += avoidanceForces[i].x * deltaTime;
            direction.y += avoidanceForces[i].y * deltaTime;
            
            // Renormalize
            float length = std::sqrt(direction.x * direction.x + direction.y * direction.y);
            if (length > 0.001f) {
                direction.x /= length;
                direction.y /= length;
            }
            
            float targetSpeed = std::min(distance / deltaTime, drone.maxSpeed);
            drone.speed = drone.speed * 0.9f + targetSpeed * 0.1f;
            
            drone.position.x += direction.x * drone.speed * deltaTime;
            drone.position.y += direction.y * drone.speed * deltaTime;
            
            drone.position.x = std::max(0.0f, std::min(coverageAreaWidth_, drone.position.x));
            drone.position.y = std::max(0.0f, std::min(coverageAreaHeight_, drone.position.y));
        }
        
        // Update battery with altitude-based drain
        float altitudeFactor = 1.0f + (std::abs(drone.altitude - config_.idealAltitude) / config_.idealAltitude) * 0.2f;
        drone.batteryLevel -= drone.batteryDrainRate * deltaTime * altitudeFactor;
        drone.batteryLevel = std::max(0.0f, drone.batteryLevel);
    }
}

void DroneSwarm::monitorBatteryLevels() {
    std::vector<int> dronesToReplace;
    
    {
        std::lock_guard<std::mutex> lock(swarmMutex_);
        
        for (auto& drone : drones_) {
            if (!drone.isActive || drone.needsReplacement) continue;
            
            if (drone.batteryLevel < config_.minBatteryThreshold) {
                float distanceToStation = drone.position.distance(chargingStation_);
                float timeToStation = distanceToStation / drone.maxSpeed;
                float batteryNeeded = timeToStation * drone.batteryDrainRate * 1.2f;
                
                if (drone.batteryLevel < batteryNeeded || drone.batteryLevel < config_.criticalBatteryThreshold) {
                    dronesToReplace.push_back(drone.id);
                } else {
                    drone.targetPosition = chargingStation_;
                    drone.needsReplacement = true;
                }
            }
        }
    }
    
    // Replace drones outside of lock
    for (int droneId : dronesToReplace) {
        replaceDrone(droneId);
    }
}

void DroneSwarm::handleLowBattery(DroneState& drone) {
    float distanceToStation = drone.position.distance(chargingStation_);
    float timeToStation = distanceToStation / drone.maxSpeed;
    float batteryNeeded = timeToStation * drone.batteryDrainRate * 1.2f;
    
    if (drone.batteryLevel < batteryNeeded || drone.batteryLevel < config_.criticalBatteryThreshold) {
        replaceDrone(drone.id);
    } else {
        drone.targetPosition = chargingStation_;
        drone.needsReplacement = true;
    }
}

Point2D DroneSwarm::computeOptimalPosition(const DroneState& drone, const std::vector<VoronoiCell>& cells) {
    for (const auto& cell : cells) {
        if (cell.droneId == drone.id) {
            if (cell.vertices.size() < 3) {
                return drone.position;
            }
            
            // Compute centroid of the Voronoi cell polygon
            float area = 0.0f;
            float centerX = 0.0f;
            float centerY = 0.0f;
            
            for (size_t i = 0; i < cell.vertices.size(); ++i) {
                size_t j = (i + 1) % cell.vertices.size();
                float a = cell.vertices[i].x * cell.vertices[j].y - 
                         cell.vertices[j].x * cell.vertices[i].y;
                area += a;
                centerX += (cell.vertices[i].x + cell.vertices[j].x) * a;
                centerY += (cell.vertices[i].y + cell.vertices[j].y) * a;
            }
            
            if (std::abs(area) > 0.001f) {
                area *= 0.5f;
                centerX /= (6.0f * area);
                centerY /= (6.0f * area);
                
                // Ensure the centroid is within bounds
                centerX = std::max(0.0f, std::min(coverageAreaWidth_, centerX));
                centerY = std::max(0.0f, std::min(coverageAreaHeight_, centerY));
                
                // Apply smoothing to avoid rapid movements
                float smoothingFactor = 0.7f;
                centerX = drone.position.x * (1 - smoothingFactor) + centerX * smoothingFactor;
                centerY = drone.position.y * (1 - smoothingFactor) + centerY * smoothingFactor;
                
                return {centerX, centerY};
            }
        }
    }
    
    return drone.position;
}

std::vector<Point2D> DroneSwarm::computeVoronoiVertices(int droneIndex, const std::vector<DroneState>& allDrones) {
    std::vector<Point2D> vertices;
    const DroneState& currentDrone = allDrones[droneIndex];
    
    // Production-ready Voronoi tessellation using half-plane intersection
    std::vector<std::pair<Point2D, Point2D>> halfPlanes;
    
    // Add boundary half-planes
    halfPlanes.push_back({{0, 0}, {coverageAreaWidth_, 0}});
    halfPlanes.push_back({{coverageAreaWidth_, 0}, {coverageAreaWidth_, coverageAreaHeight_}});
    halfPlanes.push_back({{coverageAreaWidth_, coverageAreaHeight_}, {0, coverageAreaHeight_}});
    halfPlanes.push_back({{0, coverageAreaHeight_}, {0, 0}});
    
    // Add half-planes from neighboring drones
    for (size_t j = 0; j < allDrones.size(); ++j) {
        if (j == droneIndex || !allDrones[j].isActive) continue;
        
        const DroneState& neighbor = allDrones[j];
        
        // Perpendicular bisector between current drone and neighbor
        float midX = (currentDrone.position.x + neighbor.position.x) / 2.0f;
        float midY = (currentDrone.position.y + neighbor.position.y) / 2.0f;
        
        float dx = neighbor.position.x - currentDrone.position.x;
        float dy = neighbor.position.y - currentDrone.position.y;
        
        // Perpendicular direction
        float perpX = -dy;
        float perpY = dx;
        float length = std::sqrt(perpX * perpX + perpY * perpY);
        
        if (length > 0.001f) {
            perpX /= length;
            perpY /= length;
            
            // Create half-plane line segment
            float extendLength = std::max(coverageAreaWidth_, coverageAreaHeight_);
            Point2D p1 = {midX - perpX * extendLength, midY - perpY * extendLength};
            Point2D p2 = {midX + perpX * extendLength, midY + perpY * extendLength};
            
            halfPlanes.push_back({p1, p2});
        }
    }
    
    // Compute intersection of all half-planes using Sutherland-Hodgman algorithm
    vertices.clear();
    
    // Start with boundary box
    vertices.push_back({0, 0});
    vertices.push_back({coverageAreaWidth_, 0});
    vertices.push_back({coverageAreaWidth_, coverageAreaHeight_});
    vertices.push_back({0, coverageAreaHeight_});
    
    // Clip against each half-plane
    for (size_t i = 4; i < halfPlanes.size(); ++i) {
        if (vertices.empty()) break;
        
        std::vector<Point2D> newVertices;
        const auto& plane = halfPlanes[i];
        
        // Direction vector of the half-plane edge
        float edgeX = plane.second.x - plane.first.x;
        float edgeY = plane.second.y - plane.first.y;
        
        for (size_t j = 0; j < vertices.size(); ++j) {
            size_t k = (j + 1) % vertices.size();
            const Point2D& v1 = vertices[j];
            const Point2D& v2 = vertices[k];
            
            // Check which side of the half-plane each vertex is on
            float cross1 = (v1.x - plane.first.x) * edgeY - (v1.y - plane.first.y) * edgeX;
            float cross2 = (v2.x - plane.first.x) * edgeY - (v2.y - plane.first.y) * edgeX;
            
            // The half-plane keeps points on the side of the current drone
            float crossCurrent = (currentDrone.position.x - plane.first.x) * edgeY - 
                               (currentDrone.position.y - plane.first.y) * edgeX;
            bool keepPositive = crossCurrent > 0;
            
            bool inside1 = keepPositive ? (cross1 >= 0) : (cross1 <= 0);
            bool inside2 = keepPositive ? (cross2 >= 0) : (cross2 <= 0);
            
            if (inside1) {
                newVertices.push_back(v1);
                
                if (!inside2) {
                    // Compute intersection
                    float t = cross1 / (cross1 - cross2);
                    Point2D intersection;
                    intersection.x = v1.x + t * (v2.x - v1.x);
                    intersection.y = v1.y + t * (v2.y - v1.y);
                    newVertices.push_back(intersection);
                }
            } else if (inside2) {
                // Compute intersection
                float t = cross1 / (cross1 - cross2);
                Point2D intersection;
                intersection.x = v1.x + t * (v2.x - v1.x);
                intersection.y = v1.y + t * (v2.y - v1.y);
                newVertices.push_back(intersection);
            }
        }
        
        vertices = newVertices;
    }
    
    // Ensure vertices are within bounds
    for (auto& v : vertices) {
        v.x = std::max(0.0f, std::min(coverageAreaWidth_, v.x));
        v.y = std::max(0.0f, std::min(coverageAreaHeight_, v.y));
    }
    
    return vertices;
}

bool DroneSwarm::isPointInCoverageArea(const Point2D& point) const {
    return point.x >= 0 && point.x <= coverageAreaWidth_ &&
           point.y >= 0 && point.y <= coverageAreaHeight_;
}

void DroneSwarm::rebalanceCoverage() {
    optimizeCoverage();
    
    DroneMessaging::Message msg;
    msg.type = DroneMessaging::SWARM_REBALANCE;
    msg.senderId = -1;
    msg.receiverId = -1;
    msg.timestamp = std::chrono::steady_clock::now();
    
    broadcastMessage(DroneMessaging::encodeMessage(msg));
}

std::vector<int> DroneSwarm::findNeighborDrones(int droneId, float maxDistance) {
    std::vector<int> neighbors;
    
    auto it = droneIdToIndex_.find(droneId);
    if (it == droneIdToIndex_.end()) return neighbors;
    
    const DroneState& drone = drones_[it->second];
    
    for (const auto& other : drones_) {
        if (other.id != droneId && other.isActive) {
            float dist = drone.position.distance(other.position);
            if (dist <= maxDistance) {
                neighbors.push_back(other.id);
            }
        }
    }
    
    return neighbors;
}


// DroneMessaging implementation

std::vector<uint8_t> DroneMessaging::encodeMessage(const Message& msg) {
    std::vector<uint8_t> data;
    size_t totalSize = 1 + 2 * sizeof(int) + sizeof(size_t) + msg.payload.size();
    data.reserve(totalSize);
    
    data.push_back(static_cast<uint8_t>(msg.type));
    
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&msg.senderId), 
                reinterpret_cast<const uint8_t*>(&msg.senderId) + sizeof(int));
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&msg.receiverId), 
                reinterpret_cast<const uint8_t*>(&msg.receiverId) + sizeof(int));
    
    size_t payloadSize = msg.payload.size();
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&payloadSize), 
                reinterpret_cast<const uint8_t*>(&payloadSize) + sizeof(size_t));
    
    data.insert(data.end(), msg.payload.begin(), msg.payload.end());
    
    return data;
}

DroneMessaging::Message DroneMessaging::decodeMessage(const std::vector<uint8_t>& data) {
    Message msg;
    
    if (data.size() < 1 + 2 * sizeof(int) + sizeof(size_t)) {
        return msg;
    }
    
    size_t offset = 0;
    msg.type = static_cast<MessageType>(data[offset++]);
    
    memcpy(&msg.senderId, data.data() + offset, sizeof(int));
    offset += sizeof(int);
    
    memcpy(&msg.receiverId, data.data() + offset, sizeof(int));
    offset += sizeof(int);
    
    size_t payloadSize;
    memcpy(&payloadSize, data.data() + offset, sizeof(size_t));
    offset += sizeof(size_t);
    
    if (offset + payloadSize <= data.size()) {
        msg.payload.assign(data.begin() + offset, data.begin() + offset + payloadSize);
    }
    
    msg.timestamp = std::chrono::steady_clock::now();
    
    return msg;
}

std::vector<uint8_t> DroneMessaging::encodePositionUpdate(const Point2D& position, float altitude) {
    std::vector<uint8_t> payload(3 * sizeof(float));
    memcpy(payload.data(), &position.x, sizeof(float));
    memcpy(payload.data() + sizeof(float), &position.y, sizeof(float));
    memcpy(payload.data() + 2 * sizeof(float), &altitude, sizeof(float));
    return payload;
}

std::tuple<Point2D, float> DroneMessaging::decodePositionUpdate(const std::vector<uint8_t>& payload) {
    Point2D position{0, 0};
    float altitude = 0;
    
    if (payload.size() >= 3 * sizeof(float)) {
        memcpy(&position.x, payload.data(), sizeof(float));
        memcpy(&position.y, payload.data() + sizeof(float), sizeof(float));
        memcpy(&altitude, payload.data() + 2 * sizeof(float), sizeof(float));
    }
    
    return {position, altitude};
}

std::vector<uint8_t> DroneMessaging::encodeBatteryStatus(float batteryLevel, float estimatedFlightTime) {
    std::vector<uint8_t> payload(2 * sizeof(float));
    memcpy(payload.data(), &batteryLevel, sizeof(float));
    memcpy(payload.data() + sizeof(float), &estimatedFlightTime, sizeof(float));
    return payload;
}

std::pair<float, float> DroneMessaging::decodeBatteryStatus(const std::vector<uint8_t>& payload) {
    float batteryLevel = 0;
    float estimatedFlightTime = 0;
    
    if (payload.size() >= 2 * sizeof(float)) {
        memcpy(&batteryLevel, payload.data(), sizeof(float));
        memcpy(&estimatedFlightTime, payload.data() + sizeof(float), sizeof(float));
    }
    
    return {batteryLevel, estimatedFlightTime};
}

std::vector<uint8_t> DroneMessaging::encodeDetectionReport(int objectType, const Point2D& location, float confidence) {
    std::vector<uint8_t> payload(sizeof(int) + 3 * sizeof(float));
    memcpy(payload.data(), &objectType, sizeof(int));
    memcpy(payload.data() + sizeof(int), &location.x, sizeof(float));
    memcpy(payload.data() + sizeof(int) + sizeof(float), &location.y, sizeof(float));
    memcpy(payload.data() + sizeof(int) + 2 * sizeof(float), &confidence, sizeof(float));
    return payload;
}

std::tuple<int, Point2D, float> DroneMessaging::decodeDetectionReport(const std::vector<uint8_t>& payload) {
    int objectType = 0;
    Point2D location{0, 0};
    float confidence = 0;
    
    if (payload.size() >= sizeof(int) + 3 * sizeof(float)) {
        memcpy(&objectType, payload.data(), sizeof(int));
        memcpy(&location.x, payload.data() + sizeof(int), sizeof(float));
        memcpy(&location.y, payload.data() + sizeof(int) + sizeof(float), sizeof(float));
        memcpy(&confidence, payload.data() + sizeof(int) + 2 * sizeof(float), sizeof(float));
    }
    
    return {objectType, location, confidence};
}

void DroneSwarm::updateThreadFunc() {
    const float updateInterval = 0.1f; // 100ms updates
    auto lastUpdate = std::chrono::steady_clock::now();
    
    while (isRunning_) {
        auto now = std::chrono::steady_clock::now();
        float deltaTime = std::chrono::duration<float>(now - lastUpdate).count();
        
        if (deltaTime >= updateInterval) {
            updateSwarm(deltaTime);
            lastUpdate = now;
        }
        
        // Small sleep to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void DroneSwarm::initializeV2XSocket() {
    if (v2xSocket_ > 0) {
        return;  // Already initialized
    }
    
    // Create UDP socket for V2X communication
    v2xSocket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (v2xSocket_ < 0) {
        return;
    }
    
    // Configure socket for V2X requirements
    int optval = 1;
    setsockopt(v2xSocket_, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
    setsockopt(v2xSocket_, SOL_SOCKET, SO_BROADCAST, &optval, sizeof(optval));
    
    // Set priority for V2X traffic (DSRC AC_VI - Video priority)
    // Note: SO_PRIORITY is Linux-specific
#ifdef __linux__
    int priority = 5;  // 802.11p AC_VI priority
    setsockopt(v2xSocket_, SOL_SOCKET, SO_PRIORITY, &priority, sizeof(priority));
#endif
    // For other platforms, QoS settings would be configured at the network layer
    
    // Set non-blocking mode for real-time operation
    int flags = fcntl(v2xSocket_, F_GETFL, 0);
    fcntl(v2xSocket_, F_SETFL, flags | O_NONBLOCK);
    
    // Bind to V2X port for receiving
    struct sockaddr_in localAddr;
    memset(&localAddr, 0, sizeof(localAddr));
    localAddr.sin_family = AF_INET;
    localAddr.sin_addr.s_addr = INADDR_ANY;
    localAddr.sin_port = htons(config_.v2xPort);
    
    if (bind(v2xSocket_, reinterpret_cast<struct sockaddr*>(&localAddr), sizeof(localAddr)) < 0) {
        close(v2xSocket_);
        v2xSocket_ = 0;
    }
}

void DroneSwarm::cleanupV2XSocket() {
    if (v2xSocket_ > 0) {
        close(v2xSocket_);
        v2xSocket_ = 0;
    }
}

} // namespace tacs