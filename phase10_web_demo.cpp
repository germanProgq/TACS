// Phase 10 Web Demo - Graphical frontend with real AI processing
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>

#include "simulation/web_frontend.h"
#include "models/tacs_pipeline.h"
#include "rl/rl_policy_net.h"

using namespace tacs;

std::atomic<bool> running(true);

void signalHandler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\nShutting down web frontend..." << std::endl;
        running = false;
    }
}

int main() {
    std::cout << "=====================================" << std::endl;
    std::cout << "PHASE 10: WEB-BASED GRAPHICAL FRONTEND" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "\nStarting TACS web interface with real AI processing..." << std::endl;
    
    // Set up signal handler for graceful shutdown
    signal(SIGINT, signalHandler);
    
    try {
        // Create web frontend
        WebFrontend frontend(8080);
        
        // Create and configure TACS pipeline
        PipelineConfig config;
        config.enable_detection = true;
        config.enable_tracking = true;
        config.enable_accident_detection = true;
        config.enable_weather_classification = true;
        config.use_fp16 = true;
        
        auto pipeline = std::make_shared<TACSpipeline>(config);
        std::cout << "✓ TACS pipeline created" << std::endl;
        
        // Create RL policy
        RLPolicyNet::Config rl_config;
        rl_config.state_dim = 16;
        rl_config.num_actions = 4;
        rl_config.hidden_dim = 64;
        rl_config.learning_rate = 0.001f;
        
        auto rl_policy = std::make_shared<RLPolicyNet>(rl_config);
        std::cout << "✓ RL policy created" << std::endl;
        
        // Connect AI components to frontend
        frontend.setTACSPipeline(pipeline);
        frontend.setRLPolicy(rl_policy);
        
        // Start web server
        if (!frontend.start()) {
            std::cerr << "Failed to start web frontend" << std::endl;
            return 1;
        }
        
        std::cout << "\n✓ Web interface started successfully!" << std::endl;
        std::cout << "✓ Open your browser and navigate to: http://localhost:8080" << std::endl;
        std::cout << "\nFeatures:" << std::endl;
        std::cout << "  - Real-time AI object detection (vehicles, pedestrians, cyclists)" << std::endl;
        std::cout << "  - Live tracking with Kalman filters" << std::endl;
        std::cout << "  - Weather classification" << std::endl;
        std::cout << "  - Accident detection" << std::endl;
        std::cout << "  - RL-based traffic signal control" << std::endl;
        std::cout << "  - Interactive entity spawning" << std::endl;
        std::cout << "\nKeyboard shortcuts:" << std::endl;
        std::cout << "  V - Spawn vehicle" << std::endl;
        std::cout << "  P - Spawn pedestrian" << std::endl;
        std::cout << "  B - Spawn cyclist" << std::endl;
        std::cout << "  O - Spawn obstacle" << std::endl;
        std::cout << "  X - Spawn accident" << std::endl;
        std::cout << "  C - Clear all entities" << std::endl;
        std::cout << "\nPress Ctrl+C to stop the server." << std::endl;
        
        // Spawn some initial entities for demonstration
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // Run until interrupted
        while (running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        frontend.stop();
        std::cout << "\n✓ Web frontend stopped successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}