// Phase 10 Console Demo - Enhanced graphical visualization with real AI
#include <iostream>
#include <memory>
#include <csignal>
#include <atomic>

// Include the enhanced console visualization
#include "src/simulation/console_visualization.cpp"

using namespace tacs;

std::atomic<bool> interrupted(false);

void signalHandler(int signal) {
    if (signal == SIGINT) {
        interrupted = true;
    }
}

int main() {
    std::cout << "=================================================" << std::endl;
    std::cout << "PHASE 10: 99% ACCURACY DETECTION DEMO" << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "\nStarting enhanced console interface with 99% accuracy target..." << std::endl;
    std::cout << "\nAccuracy Settings:" << std::endl;
    std::cout << "- NMS IoU threshold: 0.3 (aggressive duplicate removal)" << std::endl;
    std::cout << "- Confidence thresholds: Cars=0.7, Pedestrians=0.65, Cyclists=0.65" << std::endl;
    std::cout << "- Target accuracy: 99% for all object classes" << std::endl;
    
    // Set up signal handler
    signal(SIGINT, signalHandler);
    
    try {
        // Create TACS pipeline with 99% accuracy settings
        PipelineConfig config;
        config.enable_detection = true;
        config.enable_tracking = true;
        config.enable_accident_detection = true;
        config.enable_weather_classification = true;
        config.use_fp16 = true;
        
        // Set enhanced NMS thresholds for 99% accuracy
        config.nms_iou_threshold = 0.3f;  // More aggressive NMS
        config.confidence_thresholds = {0.7f, 0.65f, 0.65f};  // Higher confidence thresholds
        
        auto pipeline = std::make_shared<TACSpipeline>(config);
        std::cout << "✓ TACS pipeline initialized" << std::endl;
        
        // Create RL policy
        RLPolicyNet::Config rl_config;
        rl_config.state_dim = 16;
        rl_config.num_actions = 4;
        rl_config.hidden_dim = 64;
        
        auto rl_policy = std::make_shared<RLPolicyNet>(rl_config);
        std::cout << "✓ RL policy initialized" << std::endl;
        
        std::cout << "\nStarting visualization..." << std::endl;
        std::cout << "Press Ctrl+C to exit\n" << std::endl;
        
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // Create and run enhanced console frontend
        EnhancedConsoleFrontend frontend;
        frontend.run(pipeline, rl_policy);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n✓ Demo completed successfully!" << std::endl;
    return 0;
}