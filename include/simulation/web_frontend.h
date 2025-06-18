// Web-based simulation frontend for TACS visualization
#pragma once

#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <string>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include "models/tacs_pipeline.h"
#include "rl/rl_policy_net.h"
#include "utils/metrics.h"

namespace tacs {

// JSON message types for WebSocket communication
enum class MessageType {
    FRAME_UPDATE,
    DETECTION_UPDATE,
    METRICS_UPDATE,
    SIGNAL_UPDATE,
    ENTITY_SPAWN,
    ENTITY_UPDATE,
    WEATHER_UPDATE,
    ACCIDENT_UPDATE,
    CONFIG_UPDATE
};

// Web frontend that serves HTML/JS and streams AI data via WebSocket
class WebFrontend {
public:
    WebFrontend(int port = 8080);
    ~WebFrontend();
    
    // Start web server and WebSocket server
    bool start();
    void stop();
    
    // Connect to TACS backend
    void setTACSPipeline(std::shared_ptr<TACSpipeline> pipeline);
    void setRLPolicy(std::shared_ptr<RLPolicyNet> policy);
    
    // Main processing loop
    void run();
    
    // Send updates to connected clients
    void sendFrameData(const std::string& imageData);
    void sendDetections(const std::vector<utils::Detection>& detections);
    void sendMetrics(const PerformanceMetrics& metrics);
    void sendSignalStates(const std::map<std::string, SignalState>& signals);
    void sendWeatherUpdate(const std::string& weather);
    void sendAccidentUpdate(const std::string& accident);
    
    // Handle client commands
    void handleClientMessage(const std::string& message);
    
private:
    // Server configuration
    int port_;
    std::atomic<bool> running_;
    
    // Backend connections
    std::shared_ptr<TACSpipeline> tacs_pipeline_;
    std::shared_ptr<RLPolicyNet> rl_policy_;
    
    // WebSocket handling
    std::thread server_thread_;
    std::thread processing_thread_;
    std::mutex message_mutex_;
    std::queue<std::string> outgoing_messages_;
    std::condition_variable message_cv_;
    
    // Simulation state
    struct SimulationState {
        std::vector<SimulatedEntity> entities;
        IntersectionConfig intersection;
        PerformanceMetrics metrics;
        std::chrono::steady_clock::time_point last_update;
        int frame_count = 0;
    } state_;
    std::mutex state_mutex_;
    
    // Helper methods
    std::string createJSONMessage(MessageType type, const std::string& data);
    std::string serializeDetections(const std::vector<utils::Detection>& detections);
    std::string serializeMetrics(const PerformanceMetrics& metrics);
    std::string serializeSignals(const std::map<std::string, SignalState>& signals);
    void generateSimulatedFrame(core::Tensor& frame);
    std::string encodeFrameToBase64(const core::Tensor& frame);
    
    // HTTP server methods
    void serveHTTP();
    std::string getHTMLContent();
    std::string getJavaScriptContent();
    std::string getCSSContent();
    
    // Processing loop
    void processingLoop();
};

// Minimal HTTP server for serving the web interface
class SimpleHTTPServer {
public:
    SimpleHTTPServer(int port);
    ~SimpleHTTPServer();
    
    void start();
    void stop();
    void handleRequest(int client_socket);
    
private:
    int port_;
    int server_socket_;
    std::atomic<bool> running_;
    std::thread server_thread_;
    
    std::string getMimeType(const std::string& path);
    void sendResponse(int client_socket, int status_code, 
                     const std::string& content_type,
                     const std::string& body);
};

} // namespace tacs