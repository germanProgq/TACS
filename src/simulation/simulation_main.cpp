// Standalone TACS simulation application
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <atomic>

#ifdef SDL2_FOUND
#include "simulation/simulation_frontend.h"
#else
#include "simulation/simulation_frontend_stub.h"
#endif

#include "models/tacs_pipeline.h"
#include "models/tacsnet.h"
#include "models/accidentnet.h"
#include "models/weathernet.h"
#include "tracking/memory_tracker.h"
#include "rl/rl_policy_net.h"
#include "utils/serialization.h"

using namespace tacs;

// Configuration structure
struct SimulationConfig {
    int window_width = 1280;
    int window_height = 720;
    bool fullscreen = false;
    
    // Model paths
    std::string tacsnet_model = "models/tacsnet.model";
    std::string accidentnet_model = "models/accidentnet.model";
    std::string weathernet_model = "models/weathernet.model";
    std::string rl_policy_model = "models/rl_policy.model";
    
    // Simulation parameters
    int initial_vehicles = 15;
    int initial_pedestrians = 8;
    int initial_cyclists = 5;
    float spawn_rate = 0.1f;  // New entities per second
    
    // Performance settings
    bool enable_vsync = true;
    int target_fps = 60;
    
    // Feature toggles
    bool enable_detection = true;
    bool enable_tracking = true;
    bool enable_accident_detection = true;
    bool enable_weather_detection = true;
    bool enable_rl_control = true;
};

// Load configuration from file
SimulationConfig loadConfig(const std::string& path) {
    SimulationConfig config;
    
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "No config file found, using defaults" << std::endl;
        return config;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, '=')) {
            std::string value;
            if (std::getline(iss, value)) {
                // Trim whitespace
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                // Parse known keys
                if (key == "window_width") config.window_width = std::stoi(value);
                else if (key == "window_height") config.window_height = std::stoi(value);
                else if (key == "fullscreen") config.fullscreen = (value == "true");
                else if (key == "initial_vehicles") config.initial_vehicles = std::stoi(value);
                else if (key == "initial_pedestrians") config.initial_pedestrians = std::stoi(value);
                else if (key == "initial_cyclists") config.initial_cyclists = std::stoi(value);
                else if (key == "spawn_rate") config.spawn_rate = std::stof(value);
                else if (key == "enable_vsync") config.enable_vsync = (value == "true");
                else if (key == "target_fps") config.target_fps = std::stoi(value);
            }
        }
    }
    
    return config;
}

// Initialize TACS backend components
std::shared_ptr<TACSpipeline> initializeTACSBackend(const SimulationConfig& config) {
    // Create pipeline with default configuration
    auto pipeline = std::make_shared<TACSpipeline>();
    
    // Try to load pre-trained models
    try {
        pipeline->loadModels("models/");
        std::cout << "Loaded pre-trained models from models/ directory" << std::endl;
    } catch (...) {
        std::cout << "No pre-trained models found, using random weights" << std::endl;
    }
    
    return pipeline;
}

// Initialize RL policy
std::shared_ptr<RLPolicyNet> initializeRLPolicy(const SimulationConfig& config) {
    RLPolicyNet::Config rl_config;
    rl_config.state_dim = 16;
    rl_config.num_actions = 4;
    auto policy = std::make_shared<RLPolicyNet>(rl_config);
    
    try {
        loadModel(*policy, config.rl_policy_model);
        std::cout << "Loaded RL policy model from " << config.rl_policy_model << std::endl;
    } catch (...) {
        std::cout << "No pre-trained RL policy found, using random weights" << std::endl;
    }
    
    return policy;
}

// Spawn initial entities
void spawnInitialEntities(SimulationFrontend& frontend, const SimulationConfig& config) {
    // Spawn vehicles
    for (int i = 0; i < config.initial_vehicles; i++) {
        float x = 640 + (rand() % 600 - 300);
        float y = 360 + (rand() % 400 - 200);
        frontend.spawnEntity(EntityType::VEHICLE, x, y);
    }
    
    // Spawn pedestrians
    for (int i = 0; i < config.initial_pedestrians; i++) {
        float x = 640 + (rand() % 400 - 200);
        float y = 360 + (rand() % 300 - 150);
        frontend.spawnEntity(EntityType::PEDESTRIAN, x, y);
    }
    
    // Spawn cyclists
    for (int i = 0; i < config.initial_cyclists; i++) {
        float x = 640 + (rand() % 500 - 250);
        float y = 360 + (rand() % 350 - 175);
        frontend.spawnEntity(EntityType::CYCLIST, x, y);
    }
    
    std::cout << "Spawned initial entities: " 
              << config.initial_vehicles << " vehicles, "
              << config.initial_pedestrians << " pedestrians, "
              << config.initial_cyclists << " cyclists" << std::endl;
}

// Print usage information
void printUsage() {
    std::cout << "\nTACS Simulation Controls:" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "Mouse:" << std::endl;
    std::cout << "  Left Click    - Spawn entity at cursor" << std::endl;
    std::cout << "  Scroll Wheel  - Zoom in/out" << std::endl;
    std::cout << "\nKeyboard:" << std::endl;
    std::cout << "  SPACE         - Pause/unpause simulation" << std::endl;
    std::cout << "  C             - Clear all entities" << std::endl;
    std::cout << "  V             - Spawn vehicle mode" << std::endl;
    std::cout << "  P             - Spawn pedestrian mode" << std::endl;
    std::cout << "  B             - Spawn cyclist (bicycle) mode" << std::endl;
    std::cout << "  O             - Spawn obstacle mode" << std::endl;
    std::cout << "  X             - Spawn accident mode (cycles types)" << std::endl;
    std::cout << "  R             - Reload models" << std::endl;
    std::cout << "  M             - Toggle metrics display" << std::endl;
    std::cout << "  D             - Toggle detection boxes" << std::endl;
    std::cout << "  W             - Toggle weather overlay" << std::endl;
    std::cout << "  S             - Toggle signal display" << std::endl;
    std::cout << "  Arrow Keys    - Pan camera" << std::endl;
    std::cout << "  Home          - Reset camera" << std::endl;
    std::cout << "  ESC           - Exit simulation" << std::endl;
    std::cout << "=========================" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=====================================" << std::endl;
    std::cout << "TACS SIMULATION FRONTEND" << std::endl;
    std::cout << "Traffic-Aware Control System v1.0" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Load configuration
    std::string config_path = "simulation.conf";
    if (argc > 1) {
        config_path = argv[1];
    }
    
    SimulationConfig config = loadConfig(config_path);
    
    // Initialize simulation frontend
    SimulationFrontend frontend(config.window_width, config.window_height);
    
    if (!frontend.initialize()) {
        std::cerr << "Failed to initialize simulation frontend" << std::endl;
        return 1;
    }
    
    std::cout << "Simulation window created: " 
              << config.window_width << "x" << config.window_height << std::endl;
    
    // Initialize backend if enabled
    if (config.enable_detection || config.enable_tracking || 
        config.enable_accident_detection || config.enable_weather_detection) {
        
        std::cout << "\nInitializing TACS backend..." << std::endl;
        auto pipeline = initializeTACSBackend(config);
        frontend.setTACSPipeline(pipeline);
        
        // Configure pipeline features
        pipeline->setModuleEnabled("detection", config.enable_detection);
        pipeline->setModuleEnabled("tracking", config.enable_tracking);
        pipeline->setModuleEnabled("accident", config.enable_accident_detection);
        pipeline->setModuleEnabled("weather", config.enable_weather_detection);
    }
    
    // Initialize RL policy if enabled
    if (config.enable_rl_control) {
        std::cout << "Initializing RL policy..." << std::endl;
        auto policy = initializeRLPolicy(config);
        frontend.setRLPolicy(policy);
    }
    
    // Set up intersection
    IntersectionConfig intersection;
    intersection.x = config.window_width / 2.0f;
    intersection.y = config.window_height / 2.0f;
    intersection.size = 200.0f;
    
    // Configure 4-way intersection
    intersection.lanes = {
        {intersection.x - intersection.size, intersection.y},  // West
        {intersection.x + intersection.size, intersection.y},  // East
        {intersection.x, intersection.y - intersection.size},  // North
        {intersection.x, intersection.y + intersection.size}   // South
    };
    
    // Initial signal states
    intersection.signals["north"] = SignalState::GREEN;
    intersection.signals["south"] = SignalState::GREEN;
    intersection.signals["east"] = SignalState::RED;
    intersection.signals["west"] = SignalState::RED;
    
    frontend.setupIntersection(intersection);
    
    // Spawn initial entities
    spawnInitialEntities(frontend, config);
    
    // Print usage
    printUsage();
    
    // Create background thread for entity spawning
    std::atomic<bool> spawning_active(true);
    std::thread spawn_thread([&]() {
        auto last_spawn = std::chrono::steady_clock::now();
        
        while (spawning_active) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<float>(now - last_spawn).count();
            
            if (elapsed >= 1.0f / config.spawn_rate) {
                // Randomly spawn a new entity
                int type = rand() % 3;
                float x, y;
                
                // Spawn from edges
                if (rand() % 2 == 0) {
                    // Horizontal edge
                    x = (rand() % 2) ? 0 : config.window_width;
                    y = rand() % config.window_height;
                } else {
                    // Vertical edge
                    x = rand() % config.window_width;
                    y = (rand() % 2) ? 0 : config.window_height;
                }
                
                switch (type) {
                    case 0:
                        frontend.spawnEntity(EntityType::VEHICLE, x, y);
                        break;
                    case 1:
                        frontend.spawnEntity(EntityType::PEDESTRIAN, x, y);
                        break;
                    case 2:
                        frontend.spawnEntity(EntityType::CYCLIST, x, y);
                        break;
                }
                
                last_spawn = now;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    
    // Run simulation
    std::cout << "\nSimulation started. Press ESC or close window to exit." << std::endl;
    
    try {
        frontend.run();
    } catch (const std::exception& e) {
        std::cerr << "Simulation error: " << e.what() << std::endl;
    }
    
    // Cleanup
    spawning_active = false;
    if (spawn_thread.joinable()) {
        spawn_thread.join();
    }
    
    std::cout << "\nSimulation ended." << std::endl;
    
    return 0;
}