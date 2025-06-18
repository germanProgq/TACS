// Enhanced console visualization for TACS
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>

#include "simulation/simulation_frontend_stub.h"
#include "models/tacs_pipeline.h"
#include "rl/rl_policy_net.h"

namespace tacs {

// ANSI color codes for console output
namespace Color {
    const std::string RESET = "\033[0m";
    const std::string BLACK = "\033[30m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string BLUE = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN = "\033[36m";
    const std::string WHITE = "\033[37m";
    const std::string BRIGHT_RED = "\033[91m";
    const std::string BRIGHT_GREEN = "\033[92m";
    const std::string BRIGHT_YELLOW = "\033[93m";
    const std::string BRIGHT_BLUE = "\033[94m";
    const std::string BRIGHT_MAGENTA = "\033[95m";
    const std::string BRIGHT_CYAN = "\033[96m";
    const std::string BRIGHT_WHITE = "\033[97m";
    const std::string BG_BLACK = "\033[40m";
    const std::string BG_RED = "\033[41m";
    const std::string BG_GREEN = "\033[42m";
    const std::string BG_YELLOW = "\033[43m";
    const std::string BG_BLUE = "\033[44m";
    const std::string BOLD = "\033[1m";
}

// ASCII art visualization grid
class ConsoleGrid {
public:
    ConsoleGrid(int width = 80, int height = 40) 
        : width_(width), height_(height), grid_(height, std::vector<char>(width, ' ')) {
        clear();
    }
    
    int width_;
    int height_;
    
    void clear() {
        for (auto& row : grid_) {
            std::fill(row.begin(), row.end(), ' ');
        }
        
        // Draw roads
        drawHorizontalRoad();
        drawVerticalRoad();
        drawIntersection();
    }
    
    void drawEntity(float x, float y, float w, float h, char symbol) {
        // Convert world coordinates to grid coordinates
        int gx = int(x * width_ / 1280.0f);
        int gy = int(y * height_ / 720.0f);
        int gw = std::max(1, int(w * width_ / 1280.0f));
        int gh = std::max(1, int(h * height_ / 720.0f));
        
        // Draw rectangle
        for (int dy = -gh/2; dy <= gh/2; dy++) {
            for (int dx = -gw/2; dx <= gw/2; dx++) {
                int px = gx + dx;
                int py = gy + dy;
                if (px >= 0 && px < width_ && py >= 0 && py < height_) {
                    grid_[py][px] = symbol;
                }
            }
        }
    }
    
    void drawBoundingBox(float x, float y, float w, float h) {
        int gx = int(x * width_);
        int gy = int(y * height_);
        int gw = std::max(2, int(w * width_));
        int gh = std::max(2, int(h * height_));
        
        // Draw corners only for cleaner look
        for (int i = 0; i < 3; i++) {
            setPixel(gx - gw/2 + i, gy - gh/2, '+');
            setPixel(gx + gw/2 - i, gy - gh/2, '+');
            setPixel(gx - gw/2 + i, gy + gh/2, '+');
            setPixel(gx + gw/2 - i, gy + gh/2, '+');
            
            setPixel(gx - gw/2, gy - gh/2 + i, '+');
            setPixel(gx - gw/2, gy + gh/2 - i, '+');
            setPixel(gx + gw/2, gy - gh/2 + i, '+');
            setPixel(gx + gw/2, gy + gh/2 - i, '+');
        }
    }
    
    void drawTrafficLight(int x, int y, SignalState state) {
        if (x >= 0 && x < width_ && y >= 0 && y < height_) {
            switch (state) {
                case SignalState::RED:    grid_[y][x] = 'R'; break;
                case SignalState::YELLOW: grid_[y][x] = 'Y'; break;
                case SignalState::GREEN:  grid_[y][x] = 'G'; break;
            }
        }
    }
    
    std::string render() const {
        std::stringstream output;
        
        // Top border
        output << "+" << std::string(width_, '-') << "+\n";
        
        // Grid content with colors
        for (int y = 0; y < height_; y++) {
            output << "|";
            for (int x = 0; x < width_; x++) {
                char c = grid_[y][x];
                
                // Apply colors based on character
                switch (c) {
                    case 'V': // Vehicle
                        output << Color::BRIGHT_RED << c << Color::RESET;
                        break;
                    case 'P': // Pedestrian
                        output << Color::BRIGHT_GREEN << c << Color::RESET;
                        break;
                    case 'C': // Cyclist
                        output << Color::BRIGHT_BLUE << c << Color::RESET;
                        break;
                    case 'O': // Obstacle
                        output << Color::YELLOW << c << Color::RESET;
                        break;
                    case 'X': // Accident
                        output << Color::BG_RED << Color::WHITE << c << Color::RESET;
                        break;
                    case 'R': // Red light
                        output << Color::BG_RED << Color::WHITE << c << Color::RESET;
                        break;
                    case 'Y': // Yellow light
                        output << Color::BG_YELLOW << Color::BLACK << c << Color::RESET;
                        break;
                    case 'G': // Green light
                        output << Color::BG_GREEN << Color::BLACK << c << Color::RESET;
                        break;
                    case '=': // Road
                    case '|':
                        output << Color::WHITE << c << Color::RESET;
                        break;
                    case '#': // Intersection
                        output << Color::BRIGHT_CYAN << c << Color::RESET;
                        break;
                    case '+': // Detection box
                        output << Color::BRIGHT_MAGENTA << c << Color::RESET;
                        break;
                    default:
                        output << c;
                }
            }
            output << "|\n";
        }
        
        // Bottom border
        output << "+" << std::string(width_, '-') << "+\n";
        
        return output.str();
    }
    
private:
    void drawHorizontalRoad() {
        int road_y = height_ / 2;
        int road_height = 6;
        
        for (int y = road_y - road_height/2; y <= road_y + road_height/2; y++) {
            if (y >= 0 && y < height_) {
                for (int x = 0; x < width_; x++) {
                    if (y == road_y) {
                        grid_[y][x] = '=';  // Center line
                    } else {
                        grid_[y][x] = '-';  // Road surface
                    }
                }
            }
        }
    }
    
    void drawVerticalRoad() {
        int road_x = width_ / 2;
        int road_width = 10;
        
        for (int x = road_x - road_width/2; x <= road_x + road_width/2; x++) {
            if (x >= 0 && x < width_) {
                for (int y = 0; y < height_; y++) {
                    if (x == road_x) {
                        grid_[y][x] = '|';  // Center line
                    } else if (grid_[y][x] == ' ') {
                        grid_[y][x] = ':';  // Road surface
                    }
                }
            }
        }
    }
    
    void drawIntersection() {
        int cx = width_ / 2;
        int cy = height_ / 2;
        int size = 12;
        
        for (int y = cy - size/2; y <= cy + size/2; y++) {
            for (int x = cx - size/2; x <= cx + size/2; x++) {
                if (x >= 0 && x < width_ && y >= 0 && y < height_) {
                    grid_[y][x] = '#';
                }
            }
        }
    }
    
    void setPixel(int x, int y, char c) {
        if (x >= 0 && x < width_ && y >= 0 && y < height_) {
            grid_[y][x] = c;
        }
    }
    
    std::vector<std::vector<char>> grid_;
};

// Statistics tracking structure
struct SimulationStats {
    // Total entities that have existed across the simulation
    int total_vehicles_ever_spawned = 0;
    int total_pedestrians_ever_spawned = 0;
    int total_cyclists_ever_spawned = 0;
    int total_accidents_spawned = 0;
    
    // Per-frame detection tracking for better accuracy calculation
    int total_vehicle_detection_opportunities = 0;  // Frame-seconds vehicles were present
    int total_pedestrian_detection_opportunities = 0;
    int total_cyclist_detection_opportunities = 0;
    
    // Correct detections (entities correctly identified per frame)
    int vehicles_detected_correctly = 0;
    int pedestrians_detected_correctly = 0;
    int cyclists_detected_correctly = 0;
    int accidents_detected_correctly = 0;
    
    // False positives/negatives
    int false_positive_detections = 0;
    int missed_detections = 0;
    
    // Crash detection specific
    int crashes_spawned = 0;
    int crashes_detected = 0;
    int crash_detection_attempts = 0;
    
    // Frame statistics
    int total_frames = 0;
    float total_detection_time = 0.0f;
    float total_tracking_time = 0.0f;
    
    void reset() {
        *this = SimulationStats{};
    }
    
    float getVehicleDetectionAccuracy() const {
        return total_vehicle_detection_opportunities > 0 ? (float)vehicles_detected_correctly / total_vehicle_detection_opportunities * 100.0f : 0.0f;
    }
    
    float getPedestrianDetectionAccuracy() const {
        return total_pedestrian_detection_opportunities > 0 ? (float)pedestrians_detected_correctly / total_pedestrian_detection_opportunities * 100.0f : 0.0f;
    }
    
    float getCyclistDetectionAccuracy() const {
        return total_cyclist_detection_opportunities > 0 ? (float)cyclists_detected_correctly / total_cyclist_detection_opportunities * 100.0f : 0.0f;
    }
    
    float getCrashDetectionAccuracy() const {
        return crash_detection_attempts > 0 ? (float)crashes_detected / crash_detection_attempts * 100.0f : 0.0f;
    }
    
    float getOverallDetectionAccuracy() const {
        int total_correct = vehicles_detected_correctly + pedestrians_detected_correctly + cyclists_detected_correctly;
        int total_opportunities = total_vehicle_detection_opportunities + total_pedestrian_detection_opportunities + total_cyclist_detection_opportunities;
        return total_opportunities > 0 ? (float)total_correct / total_opportunities * 100.0f : 0.0f;
    }
    
    float getAverageDetectionTime() const {
        return total_frames > 0 ? total_detection_time / total_frames : 0.0f;
    }
    
    float getAverageTrackingTime() const {
        return total_frames > 0 ? total_tracking_time / total_frames : 0.0f;
    }
};

// Enhanced console visualization frontend
class EnhancedConsoleFrontend {
public:
    EnhancedConsoleFrontend() : grid_(80, 30), running_(false), stats_() {}
    
    void run(std::shared_ptr<TACSpipeline> pipeline, std::shared_ptr<RLPolicyNet> rl_policy) {
        running_ = true;
        
        // Clear screen
        std::cout << "\033[2J\033[H";
        
        // Print header
        std::cout << Color::BOLD << Color::BRIGHT_CYAN;
        std::cout << "╔═══════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║               TACS - Traffic AI Control System - Real-time Visualization      ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════════════════════╝\n";
        std::cout << Color::RESET;
        
        auto start_time = std::chrono::steady_clock::now();
        int frame_count = 0;
        
        // Simulation state
        std::vector<SimulatedEntity> entities;
        IntersectionConfig intersection;
        intersection.x = 640;
        intersection.y = 360;
        intersection.signals["north"] = SignalState::GREEN;
        intersection.signals["south"] = SignalState::GREEN;
        intersection.signals["east"] = SignalState::RED;
        intersection.signals["west"] = SignalState::RED;
        
        // Spawn initial entities and track counts
        int initial_vehicles = 8;
        int initial_pedestrians = 4;
        int initial_cyclists = 3;
        
        stats_.total_vehicles_ever_spawned = initial_vehicles;
        stats_.total_pedestrians_ever_spawned = initial_pedestrians;
        stats_.total_cyclists_ever_spawned = initial_cyclists;
        
        for (int i = 0; i < initial_vehicles; i++) {
            entities.push_back({
                EntityType::VEHICLE,
                float(200 + rand() % 880),
                float(100 + rand() % 520),
                float(rand() % 40 - 20),
                float(rand() % 40 - 20),
                40, 20,
                i + 1,
                0xFF8080,
                "Vehicle"
            });
        }
        
        for (int i = 0; i < initial_pedestrians; i++) {
            entities.push_back({
                EntityType::PEDESTRIAN,
                float(300 + rand() % 680),
                float(200 + rand() % 320),
                float(rand() % 10 - 5),
                float(rand() % 10 - 5),
                10, 10,
                100 + i,
                0x80FF80,
                "Pedestrian"
            });
        }
        
        for (int i = 0; i < initial_cyclists; i++) {
            entities.push_back({
                EntityType::CYCLIST,
                float(250 + rand() % 780),
                float(150 + rand() % 420),
                float(rand() % 20 - 10),
                float(rand() % 20 - 10),
                15, 15,
                200 + i,
                0x8080FF,
                "Cyclist"
            });
        }
        
        while (running_ && frame_count < 300) {  // Run for 30 seconds for 99% accuracy testing
            auto frame_start = std::chrono::high_resolution_clock::now();
            frame_count++;
            
            // Dynamically spawn new entities occasionally
            if (frame_count % 100 == 0 && entities.size() < 20) {
                if (rand() % 100 < 30) { // 30% chance to spawn vehicle
                    entities.push_back({
                        EntityType::VEHICLE,
                        float(rand() % 1280),
                        float(rand() % 720),
                        float(rand() % 40 - 20),
                        float(rand() % 40 - 20),
                        40, 20,
                        1000 + frame_count,
                        0xFF8080,
                        "Vehicle"
                    });
                    stats_.total_vehicles_ever_spawned++;
                }
                if (rand() % 100 < 20) { // 20% chance to spawn pedestrian
                    entities.push_back({
                        EntityType::PEDESTRIAN,
                        float(rand() % 1280),
                        float(rand() % 720),
                        float(rand() % 10 - 5),
                        float(rand() % 10 - 5),
                        10, 10,
                        2000 + frame_count,
                        0x80FF80,
                        "Pedestrian"
                    });
                    stats_.total_pedestrians_ever_spawned++;
                }
            }
            
            // Occasionally spawn accidents for crash detection testing
            if (frame_count > 100 && frame_count % 200 == 0 && rand() % 100 < 10) {
                entities.push_back({
                    EntityType::ACCIDENT_REAR_END,
                    float(400 + rand() % 480),
                    float(300 + rand() % 120),
                    0, 0, // No movement for accidents
                    60, 30,
                    300 + stats_.crashes_spawned,
                    0xFF0000,
                    "Accident"
                });
                stats_.crashes_spawned++;
                stats_.total_accidents_spawned++;
            }
            
            // Clear grid
            grid_.clear();
            
            // Update entity positions
            for (auto& entity : entities) {
                if (entity.type != EntityType::OBSTACLE) {
                    entity.x += entity.vx * 0.1f;
                    entity.y += entity.vy * 0.1f;
                    
                    // Bounce off boundaries
                    if (entity.x < 0 || entity.x > 1280) {
                        entity.vx = -entity.vx;
                        entity.x = std::max(0.0f, std::min(1280.0f, entity.x));
                    }
                    if (entity.y < 0 || entity.y > 720) {
                        entity.vy = -entity.vy;
                        entity.y = std::max(0.0f, std::min(720.0f, entity.y));
                    }
                }
            }
            
            // Draw entities
            for (const auto& entity : entities) {
                char symbol = ' ';
                switch (entity.type) {
                    case EntityType::VEHICLE: symbol = 'V'; break;
                    case EntityType::PEDESTRIAN: symbol = 'P'; break;
                    case EntityType::CYCLIST: symbol = 'C'; break;
                    case EntityType::OBSTACLE: symbol = 'O'; break;
                    case EntityType::ACCIDENT_REAR_END:
                    case EntityType::ACCIDENT_SIDE_IMPACT:
                    case EntityType::ACCIDENT_PILE_UP: symbol = 'X'; break;
                }
                grid_.drawEntity(entity.x, entity.y, entity.width, entity.height, symbol);
            }
            
            // Draw traffic lights at intersection corners (more visible)
            int center_x = grid_.width_ / 2;
            int center_y = grid_.height_ / 2;
            grid_.drawTrafficLight(center_x, center_y - 8, intersection.signals["north"]);
            grid_.drawTrafficLight(center_x, center_y + 8, intersection.signals["south"]);
            grid_.drawTrafficLight(center_x - 8, center_y, intersection.signals["west"]);
            grid_.drawTrafficLight(center_x + 8, center_y, intersection.signals["east"]);
            
            // Process with AI pipeline if available
            PipelineOutput ai_output;
            if (pipeline) {
                core::Tensor frame_tensor({1, 3, 416, 416});
                generateFrameFromEntities(frame_tensor, entities);
                
                auto process_start = std::chrono::high_resolution_clock::now();
                ai_output = pipeline->processFrame(frame_tensor);
                auto process_end = std::chrono::high_resolution_clock::now();
                
                // Track timing statistics
                stats_.total_detection_time += ai_output.detection_time_ms;
                stats_.total_tracking_time += ai_output.tracking_time_ms;
                stats_.total_frames++;
                
                // Analyze detection accuracy
                analyzeDetectionAccuracy(entities, ai_output.detections);
                
                // Check for crash detection
                bool has_accident = false;
                for (const auto& entity : entities) {
                    if (entity.type == EntityType::ACCIDENT_REAR_END ||
                        entity.type == EntityType::ACCIDENT_SIDE_IMPACT ||
                        entity.type == EntityType::ACCIDENT_PILE_UP) {
                        has_accident = true;
                        break;
                    }
                }
                
                if (has_accident) {
                    stats_.crash_detection_attempts++;
                    if (ai_output.accident_type != AccidentNet::NORMAL) {
                        stats_.crashes_detected++;
                    }
                }
                
                // Draw detection boxes
                for (const auto& det : ai_output.detections) {
                    grid_.drawBoundingBox(det.x, det.y, det.width, det.height);
                }
            }
            
            // RL decision making
            if (rl_policy && frame_count % 50 == 0) {  // Every 5 seconds
                RLState state;
                state.queue_lengths = {float(entities.size()), float(entities.size()), 
                                     float(entities.size()), float(entities.size())};
                state.weather_condition = 0.0f;
                state.accident_indicator = 0.0f;
                
                SignalPhase phase = rl_policy->selectAction(state);
                
                // Update signals based on RL decision
                if (phase == SignalPhase::NS_GREEN_EW_RED) {
                    intersection.signals["north"] = SignalState::GREEN;
                    intersection.signals["south"] = SignalState::GREEN;
                    intersection.signals["east"] = SignalState::RED;
                    intersection.signals["west"] = SignalState::RED;
                } else if (phase == SignalPhase::NS_RED_EW_GREEN) {
                    intersection.signals["north"] = SignalState::RED;
                    intersection.signals["south"] = SignalState::RED;
                    intersection.signals["east"] = SignalState::GREEN;
                    intersection.signals["west"] = SignalState::GREEN;
                }
            }
            
            // Clear screen completely and render
            std::cout << "\033[2J\033[H";  // Clear entire screen and move cursor to home
            
            // Print header again
            std::cout << Color::BOLD << Color::BRIGHT_CYAN;
            std::cout << "╔═══════════════════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║               TACS - Traffic AI Control System - Real-time Visualization      ║\n";
            std::cout << "╚═══════════════════════════════════════════════════════════════════════════════╝\n";
            std::cout << Color::RESET;
            
            // Render grid
            std::cout << grid_.render();
            
            // Display metrics
            auto current_time = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(current_time - start_time).count();
            
            std::cout << "\n" << Color::BOLD << "REAL-TIME METRICS:" << Color::RESET << "\n";
            std::cout << std::string(80, '-') << "\n";
            
            // Performance metrics with color coding
            std::cout << Color::BRIGHT_CYAN << "Frame: " << Color::RESET << frame_count 
                     << "  " << Color::BRIGHT_CYAN << "Time: " << Color::RESET 
                     << std::fixed << std::setprecision(1) << elapsed << "s"
                     << "  " << Color::BRIGHT_CYAN << "FPS: " << Color::RESET;
            
            float fps = frame_count / elapsed;
            if (fps >= 9.0f) {
                std::cout << Color::BRIGHT_GREEN;
            } else {
                std::cout << Color::BRIGHT_RED;
            }
            std::cout << std::fixed << std::setprecision(1) << fps << Color::RESET << "\n";
            
            // AI pipeline metrics
            std::cout << Color::BRIGHT_YELLOW << "AI Pipeline: " << Color::RESET
                     << "Detection: ";
            if (ai_output.detection_time_ms <= 20.0f) {
                std::cout << Color::GREEN;
            } else {
                std::cout << Color::RED;
            }
            std::cout << std::fixed << std::setprecision(2) << ai_output.detection_time_ms << "ms" << Color::RESET
                     << "  Tracking: ";
            if (ai_output.tracking_time_ms <= 10.0f) {
                std::cout << Color::GREEN;
            } else {
                std::cout << Color::RED;
            }
            std::cout << ai_output.tracking_time_ms << "ms" << Color::RESET
                     << "  Total: ";
            float total_ms = ai_output.detection_time_ms + ai_output.tracking_time_ms;
            if (total_ms <= 50.0f) {
                std::cout << Color::GREEN;
            } else {
                std::cout << Color::RED;
            }
            std::cout << total_ms << "ms" << Color::RESET << "\n";
            
            // Detection counts
            int vehicle_count = 0, pedestrian_count = 0, cyclist_count = 0;
            for (const auto& det : ai_output.detections) {
                switch (det.class_id) {
                    case 0: vehicle_count++; break;
                    case 1: pedestrian_count++; break;
                    case 2: cyclist_count++; break;
                }
            }
            
            // Entity counts comparison
            int current_vehicles = 0, current_pedestrians = 0, current_cyclists = 0, current_accidents = 0;
            for (const auto& entity : entities) {
                switch (entity.type) {
                    case EntityType::VEHICLE: current_vehicles++; break;
                    case EntityType::PEDESTRIAN: current_pedestrians++; break;
                    case EntityType::CYCLIST: current_cyclists++; break;
                    case EntityType::ACCIDENT_REAR_END:
                    case EntityType::ACCIDENT_SIDE_IMPACT:
                    case EntityType::ACCIDENT_PILE_UP: current_accidents++; break;
                }
            }
            
            std::cout << Color::BRIGHT_MAGENTA << "SPAWNED vs DETECTED:" << Color::RESET << "\n";
            // Vehicle accuracy with 99% target highlighting
            float vehicle_acc = stats_.getVehicleDetectionAccuracy();
            std::cout << Color::RED << "Vehicles: " << Color::RESET << "Spawned=" << current_vehicles 
                     << " Detected=" << vehicle_count << " Accuracy=";
            if (vehicle_acc >= 99.0f) {
                std::cout << Color::BRIGHT_GREEN << std::fixed << std::setprecision(1) << vehicle_acc << "% ✓" << Color::RESET;
            } else {
                std::cout << Color::BRIGHT_YELLOW << std::fixed << std::setprecision(1) << vehicle_acc << "%" << Color::RESET;
            }
            std::cout << "\n";
            
            // Pedestrian accuracy with 99% target highlighting
            float ped_acc = stats_.getPedestrianDetectionAccuracy();
            std::cout << Color::GREEN << "Pedestrians: " << Color::RESET << "Spawned=" << current_pedestrians 
                     << " Detected=" << pedestrian_count << " Accuracy=";
            if (ped_acc >= 99.0f) {
                std::cout << Color::BRIGHT_GREEN << std::fixed << std::setprecision(1) << ped_acc << "% ✓" << Color::RESET;
            } else {
                std::cout << Color::BRIGHT_YELLOW << std::fixed << std::setprecision(1) << ped_acc << "%" << Color::RESET;
            }
            std::cout << "\n";
            
            // Cyclist accuracy with 99% target highlighting
            float cyclist_acc = stats_.getCyclistDetectionAccuracy();
            std::cout << Color::BLUE << "Cyclists: " << Color::RESET << "Spawned=" << current_cyclists 
                     << " Detected=" << cyclist_count << " Accuracy=";
            if (cyclist_acc >= 99.0f) {
                std::cout << Color::BRIGHT_GREEN << std::fixed << std::setprecision(1) << cyclist_acc << "% ✓" << Color::RESET;
            } else {
                std::cout << Color::BRIGHT_YELLOW << std::fixed << std::setprecision(1) << cyclist_acc << "%" << Color::RESET;
            }
            std::cout << "\n";
            std::cout << Color::YELLOW << "Accidents: " << Color::RESET << "Spawned=" << current_accidents 
                     << " Detected=" << (ai_output.accident_type != AccidentNet::NORMAL ? 1 : 0)
                     << " Accuracy=" << stats_.getCrashDetectionAccuracy() << "%\n";
            
            // Overall statistics
            std::cout << Color::BRIGHT_CYAN << "OVERALL STATISTICS:" << Color::RESET << "\n";
            float overall_acc = stats_.getOverallDetectionAccuracy();
            std::cout << "Total Detection Accuracy: ";
            if (overall_acc >= 99.0f) {
                std::cout << Color::BRIGHT_GREEN << Color::BOLD << overall_acc << "% ✓ TARGET ACHIEVED!" << Color::RESET;
            } else {
                std::cout << Color::BRIGHT_WHITE << overall_acc << "% (Target: 99%)" << Color::RESET;
            }
            std::cout << " | Avg Detection Time: " << stats_.getAverageDetectionTime() << "ms"
                     << " | Avg Tracking Time: " << stats_.getAverageTrackingTime() << "ms\n";
            std::cout << "Crash Detection: " << stats_.crashes_detected << "/" << stats_.crashes_spawned 
                     << " attempts (" << stats_.getCrashDetectionAccuracy() << "% success rate)\n";
            std::cout << "False Positives: " << stats_.false_positive_detections 
                     << " | Missed Detections: " << stats_.missed_detections << "\n";
            
            // Display sample detection details
            if (!ai_output.detections.empty()) {
                std::cout << Color::BRIGHT_CYAN << "Sample Detection Details:" << Color::RESET << "\n";
                int det_count = 0;
                for (const auto& det : ai_output.detections) {
                    if (det_count >= 2) {  // Show only first 2 detections
                        std::cout << "  ... and " << (ai_output.detections.size() - 2) << " more detections\n";
                        break;
                    }
                    std::string class_name = (det.class_id == 0) ? "Vehicle" : 
                                           (det.class_id == 1) ? "Pedestrian" : "Cyclist";
                    std::cout << "  " << class_name << ": (" 
                             << std::fixed << std::setprecision(3) 
                             << det.x << ", " << det.y << ") "
                             << det.width << "x" << det.height 
                             << " conf=" << std::setprecision(2) << det.confidence << "\n";
                    det_count++;
                }
            }
            
            // Environment status
            std::cout << Color::BRIGHT_CYAN << "Environment: " << Color::RESET
                     << "Weather: ";
            switch (ai_output.weather_type) {
                case WeatherNet::CLEAR: std::cout << Color::BRIGHT_YELLOW << "☀ Clear"; break;
                case WeatherNet::RAIN: std::cout << Color::BRIGHT_BLUE << "☔ Rain"; break;
                case WeatherNet::FOG: std::cout << Color::WHITE << "🌫 Fog"; break;
                case WeatherNet::SNOW: std::cout << Color::BRIGHT_WHITE << "❄ Snow"; break;
            }
            std::cout << Color::RESET << "  Accident: ";
            
            switch (ai_output.accident_type) {
                case AccidentNet::NORMAL: std::cout << Color::GREEN << "✓ None"; break;
                case AccidentNet::REAR_END: std::cout << Color::RED << "⚠ Rear-end"; break;
                case AccidentNet::SIDE_IMPACT: std::cout << Color::RED << "⚠ Side-impact"; break;
                case AccidentNet::PILE_UP: std::cout << Color::BRIGHT_RED << "⚠ Pile-up"; break;
            }
            std::cout << Color::RESET << "\n";
            
            // Signal states with visual indicators and position info
            std::cout << Color::BRIGHT_YELLOW << "Traffic Lights: " << Color::RESET;
            std::vector<std::string> directions = {"north", "south", "east", "west"};
            for (const auto& dir : directions) {
                if (intersection.signals.find(dir) != intersection.signals.end()) {
                    std::cout << dir << ": ";
                    switch (intersection.signals[dir]) {
                        case SignalState::RED:
                            std::cout << Color::BG_RED << Color::WHITE << " RED " << Color::RESET;
                            break;
                        case SignalState::YELLOW:
                            std::cout << Color::BG_YELLOW << Color::BLACK << " YEL " << Color::RESET;
                            break;
                        case SignalState::GREEN:
                            std::cout << Color::BG_GREEN << Color::BLACK << " GRN " << Color::RESET;
                            break;
                    }
                    std::cout << "  ";
                }
            }
            std::cout << "\n";
            
            std::cout << std::string(80, '-') << "\n";
            std::cout << Color::WHITE << "Legend: " << Color::RESET
                     << Color::BRIGHT_RED << "V" << Color::RESET << "=Vehicle "
                     << Color::BRIGHT_GREEN << "P" << Color::RESET << "=Pedestrian "
                     << Color::BRIGHT_BLUE << "C" << Color::RESET << "=Cyclist "
                     << Color::YELLOW << "O" << Color::RESET << "=Obstacle "
                     << Color::BG_RED << "X" << Color::RESET << "=Accident "
                     << Color::BRIGHT_MAGENTA << "+" << Color::RESET << "=AI Detection\n";
            
            // Frame limiting to ~10 FPS
            auto frame_end = std::chrono::high_resolution_clock::now();
            auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                frame_end - frame_start).count();
            
            if (frame_duration < 100) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100 - frame_duration));
            }
        }
        
        // Display final comprehensive statistics
        std::cout << "\n" << Color::BOLD << Color::BRIGHT_CYAN 
                 << "=== FINAL SIMULATION STATISTICS ===" << Color::RESET << "\n\n";
        
        std::cout << Color::BRIGHT_MAGENTA << "ENTITY DETECTION ACCURACY:" << Color::RESET << "\n";
        std::cout << "  Vehicles: " << stats_.vehicles_detected_correctly << "/" << stats_.total_vehicle_detection_opportunities 
                 << " correctly detected (" << std::fixed << std::setprecision(1) << stats_.getVehicleDetectionAccuracy() << "%)\n";
        std::cout << "  Pedestrians: " << stats_.pedestrians_detected_correctly << "/" << stats_.total_pedestrian_detection_opportunities 
                 << " correctly detected (" << stats_.getPedestrianDetectionAccuracy() << "%)\n";
        std::cout << "  Cyclists: " << stats_.cyclists_detected_correctly << "/" << stats_.total_cyclist_detection_opportunities 
                 << " correctly detected (" << stats_.getCyclistDetectionAccuracy() << "%)\n";
        
        int total_correct = stats_.vehicles_detected_correctly + stats_.pedestrians_detected_correctly + stats_.cyclists_detected_correctly;
        int total_opportunities = stats_.total_vehicle_detection_opportunities + stats_.total_pedestrian_detection_opportunities + stats_.total_cyclist_detection_opportunities;
        float overall_accuracy = stats_.getOverallDetectionAccuracy();
        std::cout << "  Overall: " << Color::BRIGHT_WHITE << total_correct << "/" << total_opportunities << " (" << overall_accuracy << "%)" << Color::RESET << "\n\n";
        
        std::cout << Color::BRIGHT_YELLOW << "CRASH DETECTION PERFORMANCE:" << Color::RESET << "\n";
        std::cout << "  Crashes Spawned: " << stats_.crashes_spawned << "\n";
        std::cout << "  Crashes Detected: " << stats_.crashes_detected << "\n";
        std::cout << "  Detection Attempts: " << stats_.crash_detection_attempts << "\n";
        std::cout << "  Success Rate: " << stats_.getCrashDetectionAccuracy() << "%\n\n";
        
        std::cout << Color::BRIGHT_CYAN << "PERFORMANCE METRICS:" << Color::RESET << "\n";
        std::cout << "  Total Frames Processed: " << stats_.total_frames << "\n";
        std::cout << "  Average Detection Time: " << std::setprecision(2) << stats_.getAverageDetectionTime() << "ms\n";
        std::cout << "  Average Tracking Time: " << stats_.getAverageTrackingTime() << "ms\n";
        std::cout << "  False Positives: " << stats_.false_positive_detections << "\n";
        std::cout << "  Missed Detections: " << stats_.missed_detections << "\n\n";
        
        // Calculate precision, recall, F1 score properly
        int total_true_positives = stats_.vehicles_detected_correctly + stats_.pedestrians_detected_correctly + stats_.cyclists_detected_correctly;
        int total_detections = total_true_positives + stats_.false_positive_detections;
        int total_actual_opportunities = stats_.total_vehicle_detection_opportunities + stats_.total_pedestrian_detection_opportunities + stats_.total_cyclist_detection_opportunities;
        
        float precision = total_detections > 0 ? (float)total_true_positives / total_detections * 100.0f : 0.0f;
        float recall = total_actual_opportunities > 0 ? (float)total_true_positives / total_actual_opportunities * 100.0f : 0.0f;
        float f1_score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0.0f;
        
        std::cout << Color::BRIGHT_GREEN << "ADVANCED METRICS:" << Color::RESET << "\n";
        std::cout << "  Precision: " << std::setprecision(1) << precision << "%\n";
        std::cout << "  Recall: " << recall << "%\n";
        std::cout << "  F1-Score: " << f1_score << "%\n\n";
        
        // Final accuracy summary
        std::cout << Color::BOLD << Color::BRIGHT_CYAN 
                 << "\n=== FINAL 99% ACCURACY REPORT ===" << Color::RESET << "\n";
        
        float final_vehicle_acc = stats_.getVehicleDetectionAccuracy();
        float final_ped_acc = stats_.getPedestrianDetectionAccuracy();
        float final_cyclist_acc = stats_.getCyclistDetectionAccuracy();
        float final_overall_acc = stats_.getOverallDetectionAccuracy();
        
        std::cout << "Vehicle Detection: " << std::setprecision(2) << final_vehicle_acc << "% ";
        std::cout << (final_vehicle_acc >= 99.0f ? "✓ PASSED" : "✗ FAILED") << "\n";
        
        std::cout << "Pedestrian Detection: " << final_ped_acc << "% ";
        std::cout << (final_ped_acc >= 99.0f ? "✓ PASSED" : "✗ FAILED") << "\n";
        
        std::cout << "Cyclist Detection: " << final_cyclist_acc << "% ";
        std::cout << (final_cyclist_acc >= 99.0f ? "✓ PASSED" : "✗ FAILED") << "\n";
        
        std::cout << "\nOverall Detection Accuracy: " << Color::BOLD << final_overall_acc << "%" << Color::RESET;
        if (final_overall_acc >= 99.0f) {
            std::cout << Color::BRIGHT_GREEN << " ✓ 99% TARGET ACHIEVED!" << Color::RESET << "\n";
        } else {
            std::cout << Color::BRIGHT_RED << " ✗ Target not met (Required: 99%)" << Color::RESET << "\n";
        }
        
        std::cout << "\nFalse Positives: " << stats_.false_positive_detections 
                 << " | Missed Detections: " << stats_.missed_detections << "\n";
        
        std::cout << Color::BOLD << Color::BRIGHT_GREEN 
                 << "\nSimulation completed successfully!" << Color::RESET << "\n";
    }
    
    void stop() {
        running_ = false;
    }
    
private:
    void generateFrameFromEntities(core::Tensor& frame, const std::vector<SimulatedEntity>& entities) {
        // Fill with road color
        frame.fill(0.3f);
        
        // Draw intersection area
        int cx = 208, cy = 208, size = 120;
        for (int y = cy - size/2; y < cy + size/2; y++) {
            for (int x = cx - size/2; x < cx + size/2; x++) {
                if (x >= 0 && x < 416 && y >= 0 && y < 416) {
                    float* data = frame.data_float();
                    data[0 * 416 * 416 + y * 416 + x] = 0.4f;
                    data[1 * 416 * 416 + y * 416 + x] = 0.4f;
                    data[2 * 416 * 416 + y * 416 + x] = 0.4f;
                }
            }
        }
        
        // Draw entities
        for (const auto& entity : entities) {
            int ex = int(entity.x * 416.0f / 1280.0f);
            int ey = int(entity.y * 416.0f / 720.0f);
            int ew = int(entity.width * 416.0f / 1280.0f);
            int eh = int(entity.height * 416.0f / 720.0f);
            
            float r = 0.5f, g = 0.5f, b = 0.5f;
            switch (entity.type) {
                case EntityType::VEHICLE: r = 0.9f; g = 0.3f; b = 0.3f; break;
                case EntityType::PEDESTRIAN: r = 0.3f; g = 0.9f; b = 0.3f; break;
                case EntityType::CYCLIST: r = 0.3f; g = 0.3f; b = 0.9f; break;
                case EntityType::OBSTACLE: r = 0.8f; g = 0.8f; b = 0.2f; break;
                case EntityType::ACCIDENT_REAR_END:
                case EntityType::ACCIDENT_SIDE_IMPACT:
                case EntityType::ACCIDENT_PILE_UP: r = 1.0f; g = 0.2f; b = 0.2f; break;
            }
            
            for (int y = ey - eh/2; y < ey + eh/2; y++) {
                for (int x = ex - ew/2; x < ex + ew/2; x++) {
                    if (x >= 0 && x < 416 && y >= 0 && y < 416) {
                        float* data = frame.data_float();
                        data[0 * 416 * 416 + y * 416 + x] = r;
                        data[1 * 416 * 416 + y * 416 + x] = g;
                        data[2 * 416 * 416 + y * 416 + x] = b;
                    }
                }
            }
        }
    }
    
    void analyzeDetectionAccuracy(const std::vector<SimulatedEntity>& entities, 
                                 const std::vector<Detection>& detections) {
        // Count detection opportunities this frame
        int vehicles_present = 0, pedestrians_present = 0, cyclists_present = 0;
        for (const auto& entity : entities) {
            if (entity.type == EntityType::OBSTACLE) continue; // Skip obstacles
            switch (entity.type) {
                case EntityType::VEHICLE: vehicles_present++; break;
                case EntityType::PEDESTRIAN: pedestrians_present++; break;
                case EntityType::CYCLIST: cyclists_present++; break;
                default: break;
            }
        }
        
        // Update detection opportunities (frame-entity counts)
        stats_.total_vehicle_detection_opportunities += vehicles_present;
        stats_.total_pedestrian_detection_opportunities += pedestrians_present;
        stats_.total_cyclist_detection_opportunities += cyclists_present;
        
        // Match detections to actual entities using proximity
        std::vector<bool> entity_matched(entities.size(), false);
        std::vector<bool> detection_matched(detections.size(), false);
        
        const float match_threshold = 0.2f; // 20% of screen space
        
        int matched_vehicles = 0, matched_pedestrians = 0, matched_cyclists = 0;
        
        // Try to match each detection to an actual entity
        for (size_t i = 0; i < entities.size(); i++) {
            const auto& entity = entities[i];
            if (entity.type == EntityType::OBSTACLE) continue; // Skip obstacles
            
            float entity_x = entity.x / 1280.0f;
            float entity_y = entity.y / 720.0f;
            
            for (size_t j = 0; j < detections.size(); j++) {
                if (detection_matched[j]) continue; // Already matched
                
                const auto& det = detections[j];
                float dx = std::abs(entity_x - det.x);
                float dy = std::abs(entity_y - det.y);
                
                if (dx < match_threshold && dy < match_threshold) {
                    // Check if class matches
                    bool class_matches = false;
                    switch (det.class_id) {
                        case 0: class_matches = (entity.type == EntityType::VEHICLE); break;
                        case 1: class_matches = (entity.type == EntityType::PEDESTRIAN); break;
                        case 2: class_matches = (entity.type == EntityType::CYCLIST); break;
                    }
                    
                    if (class_matches) {
                        entity_matched[i] = true;
                        detection_matched[j] = true;
                        
                        // Count the correct match
                        switch (entity.type) {
                            case EntityType::VEHICLE: matched_vehicles++; break;
                            case EntityType::PEDESTRIAN: matched_pedestrians++; break;
                            case EntityType::CYCLIST: matched_cyclists++; break;
                            default: break;
                        }
                        break; // Move to next entity
                    }
                }
            }
        }
        
        // Update cumulative correct detection stats
        stats_.vehicles_detected_correctly += matched_vehicles;
        stats_.pedestrians_detected_correctly += matched_pedestrians;
        stats_.cyclists_detected_correctly += matched_cyclists;
        
        // Count false positives (detections that didn't match any real entity)
        for (size_t j = 0; j < detections.size(); j++) {
            if (!detection_matched[j]) {
                stats_.false_positive_detections++;
            }
        }
        
        // Count missed detections (real entities that weren't detected)
        for (size_t i = 0; i < entities.size(); i++) {
            if (entities[i].type != EntityType::OBSTACLE && !entity_matched[i]) {
                stats_.missed_detections++;
            }
        }
    }
    
    ConsoleGrid grid_;
    std::atomic<bool> running_;
    SimulationStats stats_;
};

} // namespace tacs