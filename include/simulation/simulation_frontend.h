// Simulation frontend for TACS visualization
#pragma once

#include <SDL2/SDL.h>
#include <vector>
#include <memory>
#include <map>
#include <chrono>
#include <mutex>
#include <string>
#include "models/tacs_pipeline.h"
#include "rl/rl_policy_net.h"
#include "utils/metrics.h"
#include "simulation/text_renderer.h"

namespace tacs {

// Entity types for spawning
enum class EntityType {
    VEHICLE,
    PEDESTRIAN,
    CYCLIST,
    OBSTACLE,
    ACCIDENT_REAR_END,
    ACCIDENT_SIDE_IMPACT,
    ACCIDENT_PILE_UP
};

// Simulated entity in the world
struct SimulatedEntity {
    EntityType type;
    float x, y, vx, vy;  // Position and velocity
    float width, height;  // Bounding box dimensions
    int id;
    uint32_t color;      // RGB color for rendering
    std::string label;   // Optional label for display
};

// Traffic light state
enum class SignalState {
    RED,
    YELLOW,
    GREEN
};

// Intersection configuration
struct IntersectionConfig {
    float x, y;          // Center position
    float size;          // Size of intersection
    std::vector<std::pair<float, float>> lanes;  // Lane positions
    std::map<std::string, SignalState> signals;   // Signal states by direction
};

// Performance metrics for display
struct PerformanceMetrics {
    float fps;
    float detection_ms;
    float tracking_ms;
    float rl_decision_ms;
    float render_ms;
    int detected_vehicles;
    int detected_pedestrians;
    int detected_cyclists;
    std::string weather_condition;
    std::string accident_status;
};

class SimulationFrontend {
public:
    SimulationFrontend(int width = 1280, int height = 720);
    ~SimulationFrontend();

    // Initialize SDL and create window
    bool initialize();
    
    // Main simulation loop
    void run();
    
    // Connect to TACS backend
    void setTACSPipeline(std::shared_ptr<TACSpipeline> pipeline);
    void setRLPolicy(std::shared_ptr<RLPolicyNet> policy);
    
    // Entity management
    void spawnEntity(EntityType type, float x, float y);
    void updateEntities(float dt);
    void clearEntities();
    
    // Intersection configuration
    void setupIntersection(const IntersectionConfig& config);
    void updateSignalState(const std::string& direction, SignalState state);
    
    // Performance monitoring
    void updateMetrics(const PerformanceMetrics& metrics);

private:
    // Rendering methods
    void render();
    void renderIntersection();
    void renderEntities();
    void renderDetections();
    void renderSignals();
    void renderWeatherOverlay();
    void renderPerformanceMetrics();
    void renderBoundingBox(float x, float y, float w, float h, uint32_t color, const std::string& label = "");
    
    // Event handling
    void handleEvents();
    void handleMouseClick(int x, int y);
    void handleKeyPress(SDL_Keycode key);
    
    // Coordinate conversion
    std::pair<float, float> screenToWorld(int sx, int sy);
    std::pair<int, int> worldToScreen(float wx, float wy);
    
    // Color utilities
    uint32_t getEntityColor(EntityType type);
    uint32_t getClassColor(int class_id);
    
    // Frame timing
    void updateFrameTime();
    
    // Frame capture for AI processing
    void captureFrameToTensor(Tensor& output);
    
    // Member variables
    int width_, height_;
    SDL_Window* window_;
    SDL_Renderer* renderer_;
    bool running_;
    bool paused_;
    
    // Backend connections
    std::shared_ptr<TACSpipeline> tacs_pipeline_;
    std::shared_ptr<RLPolicyNet> rl_policy_;
    
    // Simulation state
    std::vector<SimulatedEntity> entities_;
    std::vector<utils::Detection> last_detections_;
    IntersectionConfig intersection_;
    int next_entity_id_;
    EntityType spawn_mode_;
    
    // Performance tracking
    PerformanceMetrics metrics_;
    std::chrono::high_resolution_clock::time_point last_frame_time_;
    std::chrono::high_resolution_clock::time_point last_metrics_update_;
    std::vector<float> fps_history_;
    
    // Thread safety
    std::mutex entity_mutex_;
    std::mutex detection_mutex_;
    std::mutex metrics_mutex_;
    
    // UI state
    bool show_metrics_;
    bool show_bounding_boxes_;
    bool show_weather_;
    bool show_signals_;
    
    // Camera/viewport
    float camera_x_, camera_y_;
    float zoom_level_;
    
    // Text renderer for production-ready text display
    std::unique_ptr<TextRenderer> text_renderer_;
};

} // namespace tacs