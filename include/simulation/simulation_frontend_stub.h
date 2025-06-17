// Stub implementation for systems without SDL2
#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <iomanip>
#include <cstdlib>
#include "models/tacs_pipeline.h"
#include "rl/rl_policy_net.h"

// For Detection type
#include "utils/metrics.h"

namespace tacs {

// Stub types when SDL2 is not available
enum class EntityType {
    VEHICLE,
    PEDESTRIAN,
    CYCLIST,
    OBSTACLE,
    ACCIDENT_REAR_END,
    ACCIDENT_SIDE_IMPACT,
    ACCIDENT_PILE_UP
};

struct SimulatedEntity {
    EntityType type;
    float x, y, vx, vy;
    float width, height;
    int id;
    uint32_t color;
    std::string label;
};

enum class SignalState {
    RED,
    YELLOW,
    GREEN
};

struct IntersectionConfig {
    float x, y;
    float size;
    std::vector<std::pair<float, float>> lanes;
    std::map<std::string, SignalState> signals;
};

struct PerformanceMetrics {
    float fps = 60.0f;
    float detection_ms = 0.0f;
    float tracking_ms = 0.0f;
    float rl_decision_ms = 0.0f;
    float render_ms = 0.0f;
    int detected_vehicles = 0;
    int detected_pedestrians = 0;
    int detected_cyclists = 0;
    std::string weather_condition = "clear";
    std::string accident_status = "none";
};

// Stub implementation that prints to console instead of rendering
class SimulationFrontend {
public:
    SimulationFrontend(int width = 1280, int height = 720) 
        : width_(width), height_(height), running_(false) {
        std::cout << "WARNING: SDL2 not available. Using console output only." << std::endl;
        
        // Initialize default intersection
        intersection_.x = width_ / 2.0f;
        intersection_.y = height_ / 2.0f;
        intersection_.size = 200.0f;
        intersection_.signals["north"] = SignalState::GREEN;
        intersection_.signals["south"] = SignalState::GREEN;
        intersection_.signals["east"] = SignalState::RED;
        intersection_.signals["west"] = SignalState::RED;
    }
    
    ~SimulationFrontend() {}
    
    bool initialize() {
        std::cout << "SimulationFrontend initialized (console mode)" << std::endl;
        std::cout << "Window size: " << width_ << "x" << height_ << std::endl;
        return true;
    }
    
    void run() {
        std::cout << "=== TACS Simulation Running (Console Mode) ===" << std::endl;
        std::cout << "Real-time updates every 100ms. Press Ctrl+C to exit.\n" << std::endl;
        running_ = true;
        
        // Console-based simulation loop with real-time state display
        auto start_time = std::chrono::steady_clock::now();
        auto last_update = start_time;
        int frame_count = 0;
        
        while (running_) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - last_update).count();
            
            // Update display every 100ms
            if (elapsed_ms >= 100) {
                frame_count++;
                
                // Clear previous output (platform-specific)
                #ifdef _WIN32
                    system("cls");
                #else
                    std::cout << "\033[2J\033[H";  // ANSI escape codes
                #endif
                
                // Display header
                std::cout << "=== TACS Console Simulation - Frame " << frame_count << " ===" << std::endl;
                std::cout << "Time: " << std::fixed << std::setprecision(1) 
                         << std::chrono::duration<float>(current_time - start_time).count() 
                         << "s" << std::endl;
                std::cout << std::string(50, '-') << std::endl;
                
                // Display entity counts
                std::cout << "\nENTITIES:" << std::endl;
                std::cout << "  Vehicles: " << vehicle_count_ << std::endl;
                std::cout << "  Pedestrians: " << pedestrian_count_ << std::endl;
                std::cout << "  Cyclists: " << cyclist_count_ << std::endl;
                std::cout << "  Obstacles: " << obstacle_count_ << std::endl;
                std::cout << "  Accidents: " << accident_count_ << std::endl;
                
                // Display intersection state
                std::cout << "\nINTERSECTION STATUS:" << std::endl;
                std::cout << "  Position: (" << intersection_.x << ", " << intersection_.y << ")" << std::endl;
                std::cout << "  Signals: ";
                for (const auto& [dir, state] : intersection_.signals) {
                    std::cout << dir << "=" << getSignalStateString(state) << " ";
                }
                std::cout << std::endl;
                
                // Display AI metrics if available
                if (tacs_pipeline_ || rl_policy_) {
                    std::cout << "\nAI PERFORMANCE:" << std::endl;
                    std::cout << "  Detection: " << std::fixed << std::setprecision(2) 
                             << metrics_.detection_ms << "ms" << std::endl;
                    std::cout << "  Tracking: " << metrics_.tracking_ms << "ms" << std::endl;
                    std::cout << "  RL Decision: " << metrics_.rl_decision_ms << "ms" << std::endl;
                    std::cout << "  Total Pipeline: " 
                             << (metrics_.detection_ms + metrics_.tracking_ms + metrics_.rl_decision_ms) 
                             << "ms" << std::endl;
                }
                
                // Display detection results
                std::cout << "\nDETECTIONS:" << std::endl;
                std::cout << "  Vehicles: " << metrics_.detected_vehicles << std::endl;
                std::cout << "  Pedestrians: " << metrics_.detected_pedestrians << std::endl;
                std::cout << "  Cyclists: " << metrics_.detected_cyclists << std::endl;
                
                // Display environment status
                std::cout << "\nENVIRONMENT:" << std::endl;
                std::cout << "  Weather: " << metrics_.weather_condition << std::endl;
                std::cout << "  Accident Status: " << metrics_.accident_status << std::endl;
                
                // Display recent events
                if (!recent_events_.empty()) {
                    std::cout << "\nRECENT EVENTS:" << std::endl;
                    for (const auto& event : recent_events_) {
                        std::cout << "  - " << event << std::endl;
                    }
                }
                
                // Display controls reminder
                std::cout << "\nCONTROLS (simulated in console mode):" << std::endl;
                std::cout << "  Press Ctrl+C to exit" << std::endl;
                
                last_update = current_time;
                
                // Simulate entity updates
                updateEntities(0.1f);
                
                // Real AI processing if pipeline connected
                if (tacs_pipeline_) {
                    // Create a simulated camera frame tensor
                    core::Tensor frame_tensor({1, 3, 416, 416});
                    generateSimulatedFrame(frame_tensor);
                    
                    // Process with real AI pipeline
                    auto process_start = std::chrono::high_resolution_clock::now();
                    auto output = tacs_pipeline_->processFrame(frame_tensor);
                    auto process_end = std::chrono::high_resolution_clock::now();
                    
                    metrics_.detection_ms = output.detection_time_ms;
                    metrics_.tracking_ms = output.tracking_time_ms;
                    
                    // Count detections by type
                    metrics_.detected_vehicles = 0;
                    metrics_.detected_pedestrians = 0;
                    metrics_.detected_cyclists = 0;
                    
                    for (const auto& det : output.detections) {
                        switch (det.class_id) {
                            case 0: metrics_.detected_vehicles++; break;
                            case 1: metrics_.detected_pedestrians++; break;
                            case 2: metrics_.detected_cyclists++; break;
                        }
                    }
                    
                    // Get weather classification
                    switch (output.weather_type) {
                        case WeatherNet::CLEAR: metrics_.weather_condition = "clear"; break;
                        case WeatherNet::RAIN: metrics_.weather_condition = "rain"; break;
                        case WeatherNet::FOG: metrics_.weather_condition = "fog"; break;
                        case WeatherNet::SNOW: metrics_.weather_condition = "snow"; break;
                    }
                    
                    // Get accident status
                    switch (output.accident_type) {
                        case AccidentNet::NORMAL: metrics_.accident_status = "none"; break;
                        case AccidentNet::REAR_END: metrics_.accident_status = "rear-end"; break;
                        case AccidentNet::SIDE_IMPACT: metrics_.accident_status = "side-impact"; break;
                        case AccidentNet::PILE_UP: metrics_.accident_status = "pile-up"; break;
                    }
                }
                
                // Real RL decision making
                if (rl_policy_) {
                    // Create state from current simulation
                    RLState state;
                    state.queue_lengths = {float(vehicle_count_), float(vehicle_count_), 
                                         float(vehicle_count_), float(vehicle_count_)};
                    state.pedestrian_counts = {float(pedestrian_count_), float(pedestrian_count_)};
                    state.cyclist_counts = {float(cyclist_count_), float(cyclist_count_)};
                    state.weather_condition = 0.0f; // Clear
                    state.accident_indicator = accident_count_ > 0 ? 1.0f : 0.0f;
                    state.current_phase_duration = frame_count * 0.1f;
                    state.time_of_day = 0.5f; // Noon
                    
                    // Get real RL decision
                    auto decision_start = std::chrono::high_resolution_clock::now();
                    SignalPhase phase = rl_policy_->selectAction(state);
                    auto decision_end = std::chrono::high_resolution_clock::now();
                    
                    metrics_.rl_decision_ms = std::chrono::duration<float, std::milli>(
                        decision_end - decision_start).count();
                    
                    // Apply signal changes based on RL decision
                    if (frame_count % 50 == 0) {  // Every 5 seconds
                        applyRLDecision(phase);
                    }
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            // Demo timeout for validation
            auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                current_time - start_time).count();
            if (total_elapsed > 30) {
                std::cout << "\n\nDemo completed (30 seconds). Exiting..." << std::endl;
                break;
            }
        }
    }
    
    void setTACSPipeline(std::shared_ptr<TACSpipeline> pipeline) {
        tacs_pipeline_ = pipeline;
        std::cout << "TACS pipeline connected" << std::endl;
    }
    
    void setRLPolicy(std::shared_ptr<RLPolicyNet> policy) {
        rl_policy_ = policy;
        std::cout << "RL policy connected" << std::endl;
    }
    
    void spawnEntity(EntityType type, float x, float y) {
        std::cout << "Spawned entity type " << static_cast<int>(type) 
                  << " at (" << x << ", " << y << ")" << std::endl;
        
        // Update entity counts
        switch (type) {
            case EntityType::VEHICLE:
                vehicle_count_++;
                recent_events_.push_back("Vehicle spawned at (" + std::to_string(x) + ", " + std::to_string(y) + ")");
                break;
            case EntityType::PEDESTRIAN:
                pedestrian_count_++;
                recent_events_.push_back("Pedestrian spawned at (" + std::to_string(x) + ", " + std::to_string(y) + ")");
                break;
            case EntityType::CYCLIST:
                cyclist_count_++;
                recent_events_.push_back("Cyclist spawned at (" + std::to_string(x) + ", " + std::to_string(y) + ")");
                break;
            case EntityType::OBSTACLE:
                obstacle_count_++;
                recent_events_.push_back("Obstacle placed at (" + std::to_string(x) + ", " + std::to_string(y) + ")");
                break;
            case EntityType::ACCIDENT_REAR_END:
            case EntityType::ACCIDENT_SIDE_IMPACT:
            case EntityType::ACCIDENT_PILE_UP:
                accident_count_++;
                recent_events_.push_back("Accident occurred at (" + std::to_string(x) + ", " + std::to_string(y) + ")");
                break;
        }
        
        // Keep only recent events
        if (recent_events_.size() > 5) {
            recent_events_.erase(recent_events_.begin());
        }
    }
    
    void updateEntities(float dt) {
        // Simulate some entity movement/changes in console mode
        // Random entity removal to simulate traffic flow
        if (rand() % 100 < 5 && vehicle_count_ > 0) {
            vehicle_count_--;
            recent_events_.push_back("Vehicle left the intersection");
        }
        if (rand() % 100 < 3 && pedestrian_count_ > 0) {
            pedestrian_count_--;
            recent_events_.push_back("Pedestrian crossed successfully");
        }
        if (rand() % 100 < 4 && cyclist_count_ > 0) {
            cyclist_count_--;
            recent_events_.push_back("Cyclist passed through");
        }
        
        // Keep event list manageable
        while (recent_events_.size() > 5) {
            recent_events_.erase(recent_events_.begin());
        }
    }
    
    void clearEntities() {
        vehicle_count_ = 0;
        pedestrian_count_ = 0;
        cyclist_count_ = 0;
        obstacle_count_ = 0;
        accident_count_ = 0;
        recent_events_.clear();
        recent_events_.push_back("All entities cleared");
        std::cout << "All entities cleared" << std::endl;
    }
    
    void setupIntersection(const IntersectionConfig& config) {
        std::cout << "Intersection configured at (" << config.x << ", " << config.y 
                  << ") with size " << config.size << std::endl;
    }
    
    void updateSignalState(const std::string& direction, SignalState state) {
        std::cout << "Signal " << direction << " set to " << static_cast<int>(state) << std::endl;
    }
    
    void updateMetrics(const PerformanceMetrics& metrics) {
        std::cout << "Metrics: FPS=" << metrics.fps 
                  << ", Detection=" << metrics.detection_ms << "ms"
                  << ", Vehicles=" << metrics.detected_vehicles
                  << ", Pedestrians=" << metrics.detected_pedestrians
                  << ", Cyclists=" << metrics.detected_cyclists << std::endl;
    }
    
    // Helper method for test that doesn't exist in SDL version
    std::vector<utils::Detection> process(const core::Tensor& input) {
        // Return dummy detections for stub
        return {};
    }
    
    void setDetectionEnabled(bool enabled) {}
    void setTrackingEnabled(bool enabled) {}
    void setAccidentDetectionEnabled(bool enabled) {}
    void setWeatherDetectionEnabled(bool enabled) {}

private:
    // Helper methods
    std::string getSignalStateString(SignalState state) const {
        switch (state) {
            case SignalState::RED: return "RED";
            case SignalState::YELLOW: return "YELLOW";
            case SignalState::GREEN: return "GREEN";
            default: return "UNKNOWN";
        }
    }
    
    void simulateSignalChange() {
        // Simple signal rotation simulation
        for (auto& [dir, state] : intersection_.signals) {
            if (state == SignalState::GREEN) {
                state = SignalState::YELLOW;
                recent_events_.push_back("Signal " + dir + " changed to YELLOW");
            } else if (state == SignalState::YELLOW) {
                state = SignalState::RED;
                recent_events_.push_back("Signal " + dir + " changed to RED");
            } else if (rand() % 4 == 0) {  // Random chance to turn green
                state = SignalState::GREEN;
                recent_events_.push_back("Signal " + dir + " changed to GREEN");
            }
        }
    }
    
    void generateSimulatedFrame(core::Tensor& frame) {
        // Generate a simulated camera view with entities
        // Fill with base road color
        frame.fill(0.3f);  // Gray road
        
        // Draw intersection area (lighter gray)
        int cx = 208, cy = 208;  // Center in 416x416 frame
        int size = 100;
        for (int y = cy - size/2; y < cy + size/2; y++) {
            for (int x = cx - size/2; x < cx + size/2; x++) {
                if (x >= 0 && x < 416 && y >= 0 && y < 416) {
                    // CHW format: [batch, channel, height, width]
                    float* data = frame.data_float();
                    data[0 * 416 * 416 + y * 416 + x] = 0.4f;  // R channel
                    data[1 * 416 * 416 + y * 416 + x] = 0.4f;  // G channel
                    data[2 * 416 * 416 + y * 416 + x] = 0.4f;  // B channel
                }
            }
        }
        
        // Draw simulated entities as colored rectangles
        int entity_idx = 0;
        
        // Vehicles (red)
        for (int i = 0; i < std::min(vehicle_count_, 10); i++) {
            int x = 50 + (i * 30) % 300;
            int y = 50 + (i * 40) % 300;
            drawEntity(frame, x, y, 40, 20, 0.8f, 0.2f, 0.2f);
        }
        
        // Pedestrians (green)
        for (int i = 0; i < std::min(pedestrian_count_, 10); i++) {
            int x = 100 + (i * 25) % 250;
            int y = 100 + (i * 35) % 250;
            drawEntity(frame, x, y, 10, 10, 0.2f, 0.8f, 0.2f);
        }
        
        // Cyclists (blue)
        for (int i = 0; i < std::min(cyclist_count_, 10); i++) {
            int x = 150 + (i * 35) % 200;
            int y = 150 + (i * 30) % 200;
            drawEntity(frame, x, y, 15, 15, 0.2f, 0.2f, 0.8f);
        }
    }
    
    void drawEntity(core::Tensor& frame, int cx, int cy, int w, int h,
                   float r, float g, float b) {
        for (int y = cy - h/2; y < cy + h/2; y++) {
            for (int x = cx - w/2; x < cx + w/2; x++) {
                if (x >= 0 && x < 416 && y >= 0 && y < 416) {
                    // CHW format: [batch, channel, height, width]
                    float* data = frame.data_float();
                    data[0 * 416 * 416 + y * 416 + x] = r;  // R channel
                    data[1 * 416 * 416 + y * 416 + x] = g;  // G channel
                    data[2 * 416 * 416 + y * 416 + x] = b;  // B channel
                }
            }
        }
    }
    
    void applyRLDecision(SignalPhase phase) {
        // Apply signal changes based on RL phase
        switch (phase) {
            case SignalPhase::NS_GREEN_EW_RED:
                intersection_.signals["north"] = SignalState::GREEN;
                intersection_.signals["south"] = SignalState::GREEN;
                intersection_.signals["east"] = SignalState::RED;
                intersection_.signals["west"] = SignalState::RED;
                recent_events_.push_back("RL: North-South GREEN phase");
                break;
            case SignalPhase::NS_RED_EW_GREEN:
                intersection_.signals["north"] = SignalState::RED;
                intersection_.signals["south"] = SignalState::RED;
                intersection_.signals["east"] = SignalState::GREEN;
                intersection_.signals["west"] = SignalState::GREEN;
                recent_events_.push_back("RL: East-West GREEN phase");
                break;
            case SignalPhase::ALL_RED:
                intersection_.signals["north"] = SignalState::RED;
                intersection_.signals["south"] = SignalState::RED;
                intersection_.signals["east"] = SignalState::RED;
                intersection_.signals["west"] = SignalState::RED;
                recent_events_.push_back("RL: All RED safety phase");
                break;
            case SignalPhase::NS_YELLOW_EW_RED:
            case SignalPhase::NS_RED_EW_YELLOW:
                for (auto& [dir, state] : intersection_.signals) {
                    if (state == SignalState::GREEN) {
                        state = SignalState::YELLOW;
                    }
                }
                recent_events_.push_back("RL: Yellow transition phase");
                break;
            case SignalPhase::EMERGENCY_OVERRIDE:
                recent_events_.push_back("RL: Emergency override activated");
                break;
        }
        
        // Keep event list manageable
        while (recent_events_.size() > 5) {
            recent_events_.erase(recent_events_.begin());
        }
    }

    // Member variables
    int width_, height_;
    bool running_;
    std::shared_ptr<TACSpipeline> tacs_pipeline_;
    std::shared_ptr<RLPolicyNet> rl_policy_;
    
    // Simulation state
    int vehicle_count_ = 0;
    int pedestrian_count_ = 0;
    int cyclist_count_ = 0;
    int obstacle_count_ = 0;
    int accident_count_ = 0;
    IntersectionConfig intersection_;
    PerformanceMetrics metrics_;
    std::vector<std::string> recent_events_;
};

} // namespace tacs