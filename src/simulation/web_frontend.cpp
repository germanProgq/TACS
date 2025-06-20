// Web-based simulation frontend implementation
#include "simulation/web_frontend.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace tacs {

// Base64 encoding table
static const char base64_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

WebFrontend::WebFrontend(int port) : port_(port), running_(false) {
    // Initialize default intersection
    state_.intersection.x = 640;
    state_.intersection.y = 360;
    state_.intersection.size = 200;
    state_.intersection.signals["north"] = SignalState::GREEN;
    state_.intersection.signals["south"] = SignalState::GREEN;
    state_.intersection.signals["east"] = SignalState::RED;
    state_.intersection.signals["west"] = SignalState::RED;
}

WebFrontend::~WebFrontend() {
    stop();
}

bool WebFrontend::start() {
    running_ = true;
    
    // Start HTTP server thread
    server_thread_ = std::thread([this]() {
        serveHTTP();
    });
    
    // Start processing thread
    processing_thread_ = std::thread([this]() {
        processingLoop();
    });
    
    std::cout << "Web frontend started on http://localhost:" << port_ << std::endl;
    return true;
}

void WebFrontend::stop() {
    running_ = false;
    message_cv_.notify_all();
    
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

void WebFrontend::setTACSPipeline(std::shared_ptr<TACSpipeline> pipeline) {
    tacs_pipeline_ = pipeline;
}

void WebFrontend::setRLPolicy(std::shared_ptr<RLPolicyNet> policy) {
    rl_policy_ = policy;
}

void WebFrontend::run() {
    // Main loop handled by processingLoop thread
    while (running_) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void WebFrontend::processingLoop() {
    auto last_frame_time = std::chrono::steady_clock::now();
    
    while (running_) {
        auto frame_start = std::chrono::steady_clock::now();
        
        // Target 30 FPS for web streaming
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            frame_start - last_frame_time).count();
        
        if (elapsed < 33) {  // 33ms = ~30 FPS
            std::this_thread::sleep_for(std::chrono::milliseconds(33 - elapsed));
            continue;
        }
        
        last_frame_time = frame_start;
        
        // Update simulation state
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_.frame_count++;
            
            // Update entity positions
            for (auto& entity : state_.entities) {
                if (entity.type != EntityType::OBSTACLE &&
                    entity.type != EntityType::ACCIDENT_REAR_END &&
                    entity.type != EntityType::ACCIDENT_SIDE_IMPACT &&
                    entity.type != EntityType::ACCIDENT_PILE_UP) {
                    
                    entity.x += entity.vx * 0.033f;  // 33ms timestep
                    entity.y += entity.vy * 0.033f;
                    
                    // Boundary bounce
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
        }
        
        // Process with real AI pipeline if connected
        if (tacs_pipeline_) {
            // Generate simulated camera frame
            core::Tensor frame_tensor({1, 3, 416, 416});
            generateSimulatedFrame(frame_tensor);
            
            // Process frame through AI pipeline
            auto pipeline_start = std::chrono::high_resolution_clock::now();
            auto output = tacs_pipeline_->processFrame(frame_tensor);
            auto pipeline_end = std::chrono::high_resolution_clock::now();
            
            // Update metrics
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                state_.metrics.detection_ms = output.detection_time_ms;
                state_.metrics.tracking_ms = output.tracking_time_ms;
                state_.metrics.detected_vehicles = 0;
                state_.metrics.detected_pedestrians = 0;
                state_.metrics.detected_cyclists = 0;
                
                for (const auto& det : output.detections) {
                    switch (det.class_id) {
                        case 0: state_.metrics.detected_vehicles++; break;
                        case 1: state_.metrics.detected_pedestrians++; break;
                        case 2: state_.metrics.detected_cyclists++; break;
                    }
                }
                
                // Update weather
                switch (output.weather_type) {
                    case WeatherNet::CLEAR: state_.metrics.weather_condition = "clear"; break;
                    case WeatherNet::RAIN: state_.metrics.weather_condition = "rain"; break;
                    case WeatherNet::FOG: state_.metrics.weather_condition = "fog"; break;
                    case WeatherNet::SNOW: state_.metrics.weather_condition = "snow"; break;
                }
                
                // Update accident status
                switch (output.accident_type) {
                    case AccidentNet::NORMAL: state_.metrics.accident_status = "none"; break;
                    case AccidentNet::REAR_END: state_.metrics.accident_status = "rear-end"; break;
                    case AccidentNet::SIDE_IMPACT: state_.metrics.accident_status = "side-impact"; break;
                    case AccidentNet::PILE_UP: state_.metrics.accident_status = "pile-up"; break;
                }
            }
            
            // Send detection results
            sendDetections(output.detections);
            
            // Send metrics update
            sendMetrics(state_.metrics);
            
            // Send weather update
            sendWeatherUpdate(state_.metrics.weather_condition);
            
            // Send accident update
            sendAccidentUpdate(state_.metrics.accident_status);
        }
        
        // Process RL decisions if connected
        if (rl_policy_ && state_.frame_count % 150 == 0) {  // Every 5 seconds at 30 FPS
            RLState rl_state;
            
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                // Prepare RL state from current simulation
                rl_state.queue_lengths = {
                    float(std::count_if(state_.entities.begin(), state_.entities.end(),
                        [](const auto& e) { return e.type == EntityType::VEHICLE; })),
                    float(std::count_if(state_.entities.begin(), state_.entities.end(),
                        [](const auto& e) { return e.type == EntityType::VEHICLE; })),
                    float(std::count_if(state_.entities.begin(), state_.entities.end(),
                        [](const auto& e) { return e.type == EntityType::VEHICLE; })),
                    float(std::count_if(state_.entities.begin(), state_.entities.end(),
                        [](const auto& e) { return e.type == EntityType::VEHICLE; }))
                };
                rl_state.pedestrian_counts = {
                    float(std::count_if(state_.entities.begin(), state_.entities.end(),
                        [](const auto& e) { return e.type == EntityType::PEDESTRIAN; })),
                    float(std::count_if(state_.entities.begin(), state_.entities.end(),
                        [](const auto& e) { return e.type == EntityType::PEDESTRIAN; }))
                };
                rl_state.cyclist_counts = {
                    float(std::count_if(state_.entities.begin(), state_.entities.end(),
                        [](const auto& e) { return e.type == EntityType::CYCLIST; })),
                    float(std::count_if(state_.entities.begin(), state_.entities.end(),
                        [](const auto& e) { return e.type == EntityType::CYCLIST; }))
                };
                rl_state.weather_condition = 0.0f;  // Clear
                rl_state.accident_indicator = 0.0f;
                rl_state.current_phase_duration = 10.0f;
                rl_state.time_of_day = 0.5f;
            }
            
            auto decision_start = std::chrono::high_resolution_clock::now();
            SignalPhase phase = rl_policy_->selectAction(rl_state);
            auto decision_end = std::chrono::high_resolution_clock::now();
            
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                state_.metrics.rl_decision_ms = std::chrono::duration<float, std::milli>(
                    decision_end - decision_start).count();
                
                // Apply signal changes
                if (phase == SignalPhase::NS_GREEN_EW_RED) {
                    state_.intersection.signals["north"] = SignalState::GREEN;
                    state_.intersection.signals["south"] = SignalState::GREEN;
                    state_.intersection.signals["east"] = SignalState::RED;
                    state_.intersection.signals["west"] = SignalState::RED;
                } else if (phase == SignalPhase::NS_RED_EW_GREEN) {
                    state_.intersection.signals["north"] = SignalState::RED;
                    state_.intersection.signals["south"] = SignalState::RED;
                    state_.intersection.signals["east"] = SignalState::GREEN;
                    state_.intersection.signals["west"] = SignalState::GREEN;
                }
            }
            
            // Send signal update
            sendSignalStates(state_.intersection.signals);
        }
        
        // Send frame update with entity positions
        std::stringstream frame_data;
        frame_data << "{\"entities\":[";
        
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            bool first = true;
            for (const auto& entity : state_.entities) {
                if (!first) frame_data << ",";
                frame_data << "{"
                          << "\"id\":" << entity.id << ","
                          << "\"type\":" << static_cast<int>(entity.type) << ","
                          << "\"x\":" << entity.x << ","
                          << "\"y\":" << entity.y << ","
                          << "\"width\":" << entity.width << ","
                          << "\"height\":" << entity.height << ","
                          << "\"label\":\"" << entity.label << "\""
                          << "}";
                first = false;
            }
        }
        
        frame_data << "]}";
        sendFrameData(frame_data.str());
        
        // Calculate FPS
        auto frame_end = std::chrono::steady_clock::now();
        auto frame_duration = std::chrono::duration<float, std::milli>(
            frame_end - frame_start).count();
        
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_.metrics.fps = 1000.0f / frame_duration;
            state_.metrics.render_ms = frame_duration;
        }
    }
}

void WebFrontend::sendFrameData(const std::string& data) {
    std::lock_guard<std::mutex> lock(message_mutex_);
    outgoing_messages_.push(createJSONMessage(MessageType::FRAME_UPDATE, data));
    message_cv_.notify_one();
}

void WebFrontend::sendDetections(const std::vector<utils::Detection>& detections) {
    std::lock_guard<std::mutex> lock(message_mutex_);
    outgoing_messages_.push(createJSONMessage(MessageType::DETECTION_UPDATE, 
                                            serializeDetections(detections)));
    message_cv_.notify_one();
}

void WebFrontend::sendMetrics(const PerformanceMetrics& metrics) {
    std::lock_guard<std::mutex> lock(message_mutex_);
    outgoing_messages_.push(createJSONMessage(MessageType::METRICS_UPDATE,
                                            serializeMetrics(metrics)));
    message_cv_.notify_one();
}

void WebFrontend::sendSignalStates(const std::map<std::string, SignalState>& signals) {
    std::lock_guard<std::mutex> lock(message_mutex_);
    outgoing_messages_.push(createJSONMessage(MessageType::SIGNAL_UPDATE,
                                            serializeSignals(signals)));
    message_cv_.notify_one();
}

void WebFrontend::sendWeatherUpdate(const std::string& weather) {
    std::stringstream json;
    json << "{\"weather\":\"" << weather << "\"}";
    
    std::lock_guard<std::mutex> lock(message_mutex_);
    outgoing_messages_.push(createJSONMessage(MessageType::WEATHER_UPDATE, json.str()));
    message_cv_.notify_one();
}

void WebFrontend::sendAccidentUpdate(const std::string& accident) {
    std::stringstream json;
    json << "{\"accident\":\"" << accident << "\"}";
    
    std::lock_guard<std::mutex> lock(message_mutex_);
    outgoing_messages_.push(createJSONMessage(MessageType::ACCIDENT_UPDATE, json.str()));
    message_cv_.notify_one();
}

void WebFrontend::handleClientMessage(const std::string& message) {
    // Parse JSON command from client
    // Using lightweight parsing to avoid external dependencies
    if (message.find("\"spawn\"") != std::string::npos) {
        // Extract spawn parameters
        float x = 640, y = 360;
        EntityType type = EntityType::VEHICLE;
        
        // Parse x, y, type from message...
        // For demo, spawn random entity
        SimulatedEntity entity;
        entity.id = state_.frame_count * 1000 + state_.entities.size();
        entity.type = type;
        entity.x = x + (rand() % 200 - 100);
        entity.y = y + (rand() % 200 - 100);
        entity.vx = (rand() % 40 - 20);
        entity.vy = (rand() % 40 - 20);
        entity.width = 40;
        entity.height = 20;
        entity.label = "Vehicle";
        
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_.entities.push_back(entity);
    }
    else if (message.find("\"clear\"") != std::string::npos) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_.entities.clear();
    }
}

std::string WebFrontend::createJSONMessage(MessageType type, const std::string& data) {
    std::stringstream json;
    json << "{\"type\":" << static_cast<int>(type) << ",";
    json << "\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() << ",";
    json << "\"data\":" << data << "}";
    return json.str();
}

std::string WebFrontend::serializeDetections(const std::vector<utils::Detection>& detections) {
    std::stringstream json;
    json << "[";
    
    bool first = true;
    for (const auto& det : detections) {
        if (!first) json << ",";
        json << "{"
             << "\"x\":" << det.x << ","
             << "\"y\":" << det.y << ","
             << "\"w\":" << det.w << ","
             << "\"h\":" << det.h << ","
             << "\"confidence\":" << det.confidence << ","
             << "\"class_id\":" << det.class_id
             << "}";
        first = false;
    }
    
    json << "]";
    return json.str();
}

std::string WebFrontend::serializeMetrics(const PerformanceMetrics& metrics) {
    std::stringstream json;
    json << "{"
         << "\"fps\":" << metrics.fps << ","
         << "\"detection_ms\":" << metrics.detection_ms << ","
         << "\"tracking_ms\":" << metrics.tracking_ms << ","
         << "\"rl_decision_ms\":" << metrics.rl_decision_ms << ","
         << "\"render_ms\":" << metrics.render_ms << ","
         << "\"detected_vehicles\":" << metrics.detected_vehicles << ","
         << "\"detected_pedestrians\":" << metrics.detected_pedestrians << ","
         << "\"detected_cyclists\":" << metrics.detected_cyclists
         << "}";
    return json.str();
}

std::string WebFrontend::serializeSignals(const std::map<std::string, SignalState>& signals) {
    std::stringstream json;
    json << "{";
    
    bool first = true;
    for (const auto& [direction, state] : signals) {
        if (!first) json << ",";
        json << "\"" << direction << "\":" << static_cast<int>(state);
        first = false;
    }
    
    json << "}";
    return json.str();
}

void WebFrontend::generateSimulatedFrame(core::Tensor& frame) {
    // Fill with road background
    frame.fill(0.3f);
    
    // Draw intersection
    int cx = 208, cy = 208;
    int size = 100;
    
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
    std::lock_guard<std::mutex> lock(state_mutex_);
    for (const auto& entity : state_.entities) {
        // Convert world coordinates to frame coordinates
        int ex = int(entity.x * 416.0f / 1280.0f);
        int ey = int(entity.y * 416.0f / 720.0f);
        int ew = int(entity.width * 416.0f / 1280.0f);
        int eh = int(entity.height * 416.0f / 720.0f);
        
        // Choose color based on entity type
        float r = 0.5f, g = 0.5f, b = 0.5f;
        switch (entity.type) {
            case EntityType::VEHICLE: r = 0.8f; g = 0.2f; b = 0.2f; break;
            case EntityType::PEDESTRIAN: r = 0.2f; g = 0.8f; b = 0.2f; break;
            case EntityType::CYCLIST: r = 0.2f; g = 0.2f; b = 0.8f; break;
            case EntityType::OBSTACLE: r = 0.8f; g = 0.8f; b = 0.2f; break;
            case EntityType::ACCIDENT_REAR_END:
            case EntityType::ACCIDENT_SIDE_IMPACT:
            case EntityType::ACCIDENT_PILE_UP: r = 1.0f; g = 0.0f; b = 0.0f; break;
        }
        
        // Draw entity rectangle
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

void WebFrontend::serveHTTP() {
    // Create socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return;
    }
    
    // Allow socket reuse
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    // Bind to port
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);
    
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Failed to bind to port " << port_ << std::endl;
        close(server_fd);
        return;
    }
    
    // Listen for connections
    if (listen(server_fd, 10) < 0) {
        std::cerr << "Failed to listen on socket" << std::endl;
        close(server_fd);
        return;
    }
    
    std::cout << "HTTP server listening on port " << port_ << std::endl;
    
    while (running_) {
        // Accept connection
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        
        if (client_fd < 0) {
            if (running_) {
                std::cerr << "Failed to accept connection" << std::endl;
            }
            continue;
        }
        
        // Read request
        char buffer[4096] = {0};
        int bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        
        if (bytes_read > 0) {
            std::string request(buffer);
            
            // Simple request parsing
            if (request.find("GET / ") == 0 || request.find("GET /index.html") == 0) {
                // Serve main HTML page
                std::string html = getHTMLContent();
                std::stringstream response;
                response << "HTTP/1.1 200 OK\r\n";
                response << "Content-Type: text/html\r\n";
                response << "Content-Length: " << html.length() << "\r\n";
                response << "Connection: close\r\n";
                response << "\r\n";
                response << html;
                
                write(client_fd, response.str().c_str(), response.str().length());
            }
            else if (request.find("GET /tacs.js") == 0) {
                // Serve JavaScript
                std::string js = getJavaScriptContent();
                std::stringstream response;
                response << "HTTP/1.1 200 OK\r\n";
                response << "Content-Type: application/javascript\r\n";
                response << "Content-Length: " << js.length() << "\r\n";
                response << "Connection: close\r\n";
                response << "\r\n";
                response << js;
                
                write(client_fd, response.str().c_str(), response.str().length());
            }
            else if (request.find("GET /tacs.css") == 0) {
                // Serve CSS
                std::string css = getCSSContent();
                std::stringstream response;
                response << "HTTP/1.1 200 OK\r\n";
                response << "Content-Type: text/css\r\n";
                response << "Content-Length: " << css.length() << "\r\n";
                response << "Connection: close\r\n";
                response << "\r\n";
                response << css;
                
                write(client_fd, response.str().c_str(), response.str().length());
            }
            else if (request.find("GET /ws") == 0) {
                // WebSocket upgrade request
                // For production, implement proper WebSocket handshake
                // For demo, we'll use Server-Sent Events instead
                std::stringstream response;
                response << "HTTP/1.1 200 OK\r\n";
                response << "Content-Type: text/event-stream\r\n";
                response << "Cache-Control: no-cache\r\n";
                response << "Connection: keep-alive\r\n";
                response << "Access-Control-Allow-Origin: *\r\n";
                response << "\r\n";
                
                write(client_fd, response.str().c_str(), response.str().length());
                
                // Send events
                while (running_) {
                    std::unique_lock<std::mutex> lock(message_mutex_);
                    message_cv_.wait_for(lock, std::chrono::milliseconds(100));
                    
                    while (!outgoing_messages_.empty()) {
                        std::string msg = outgoing_messages_.front();
                        outgoing_messages_.pop();
                        
                        std::stringstream event;
                        event << "data: " << msg << "\n\n";
                        
                        if (write(client_fd, event.str().c_str(), event.str().length()) < 0) {
                            goto close_connection;
                        }
                    }
                }
            }
            else {
                // 404 Not Found
                std::string response = "HTTP/1.1 404 Not Found\r\n"
                                     "Content-Length: 0\r\n"
                                     "Connection: close\r\n"
                                     "\r\n";
                write(client_fd, response.c_str(), response.length());
            }
        }
        
        close_connection:
        close(client_fd);
    }
    
    close(server_fd);
}

std::string WebFrontend::getHTMLContent() {
    return R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TACS Traffic Simulation</title>
    <link rel="stylesheet" href="/tacs.css">
</head>
<body>
    <div id="container">
        <h1>TACS Traffic Simulation - Real-time AI Visualization</h1>
        
        <div id="main-content">
            <div id="canvas-container">
                <canvas id="simulation-canvas" width="1280" height="720"></canvas>
                <div id="controls">
                    <button onclick="spawnVehicle()">Spawn Vehicle (V)</button>
                    <button onclick="spawnPedestrian()">Spawn Pedestrian (P)</button>
                    <button onclick="spawnCyclist()">Spawn Cyclist (B)</button>
                    <button onclick="spawnObstacle()">Spawn Obstacle (O)</button>
                    <button onclick="spawnAccident()">Spawn Accident (X)</button>
                    <button onclick="clearEntities()">Clear All (C)</button>
                </div>
            </div>
            
            <div id="sidebar">
                <div class="panel">
                    <h3>Performance Metrics</h3>
                    <div id="metrics">
                        <div class="metric-row">
                            <span class="metric-label">FPS:</span>
                            <span id="fps" class="metric-value">0</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Detection:</span>
                            <span id="detection-ms" class="metric-value">0ms</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Tracking:</span>
                            <span id="tracking-ms" class="metric-value">0ms</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">RL Decision:</span>
                            <span id="rl-ms" class="metric-value">0ms</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Total Pipeline:</span>
                            <span id="total-ms" class="metric-value">0ms</span>
                        </div>
                    </div>
                </div>
                
                <div class="panel">
                    <h3>Detections</h3>
                    <div id="detections">
                        <div class="detection-row">
                            <span class="detection-label">Vehicles:</span>
                            <span id="vehicle-count" class="detection-value">0</span>
                        </div>
                        <div class="detection-row">
                            <span class="detection-label">Pedestrians:</span>
                            <span id="pedestrian-count" class="detection-value">0</span>
                        </div>
                        <div class="detection-row">
                            <span class="detection-label">Cyclists:</span>
                            <span id="cyclist-count" class="detection-value">0</span>
                        </div>
                    </div>
                </div>
                
                <div class="panel">
                    <h3>Environment Status</h3>
                    <div id="environment">
                        <div class="env-row">
                            <span class="env-label">Weather:</span>
                            <span id="weather" class="env-value">Clear</span>
                        </div>
                        <div class="env-row">
                            <span class="env-label">Accident:</span>
                            <span id="accident" class="env-value">None</span>
                        </div>
                    </div>
                </div>
                
                <div class="panel">
                    <h3>Signal States</h3>
                    <div id="signals">
                        <div class="signal-row">
                            <span class="signal-label">North:</span>
                            <span id="signal-north" class="signal-green">GREEN</span>
                        </div>
                        <div class="signal-row">
                            <span class="signal-label">South:</span>
                            <span id="signal-south" class="signal-green">GREEN</span>
                        </div>
                        <div class="signal-row">
                            <span class="signal-label">East:</span>
                            <span id="signal-east" class="signal-red">RED</span>
                        </div>
                        <div class="signal-row">
                            <span class="signal-label">West:</span>
                            <span id="signal-west" class="signal-red">RED</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="/tacs.js"></script>
</body>
</html>)HTML";
}

std::string WebFrontend::getJavaScriptContent() {
    return R"JS(// TACS Web Frontend JavaScript
const canvas = document.getElementById('simulation-canvas');
const ctx = canvas.getContext('2d');

// State
let entities = [];
let detections = [];
let signals = {north: 0, south: 0, east: 0, west: 0};
let metrics = {};

// Colors
const entityColors = {
    0: '#FF8080',  // Vehicle - light red
    1: '#80FF80',  // Pedestrian - light green
    2: '#8080FF',  // Cyclist - light blue
    3: '#FFFF00',  // Obstacle - yellow
    4: '#FF0000',  // Accident rear-end - red
    5: '#FF0000',  // Accident side-impact - red
    6: '#FF0000'   // Accident pile-up - red
};

const signalColors = {
    0: '#FF0000',  // RED
    1: '#FFFF00',  // YELLOW
    2: '#00FF00'   // GREEN
};

// Connect to server events
const eventSource = new EventSource('/ws');

eventSource.onmessage = function(event) {
    const message = JSON.parse(event.data);
    handleMessage(message);
};

eventSource.onerror = function(error) {
    console.error('EventSource error:', error);
};

function handleMessage(message) {
    switch(message.type) {
        case 0: // FRAME_UPDATE
            entities = message.data.entities || [];
            render();
            break;
        case 1: // DETECTION_UPDATE
            detections = message.data;
            break;
        case 2: // METRICS_UPDATE
            updateMetrics(message.data);
            break;
        case 3: // SIGNAL_UPDATE
            updateSignals(message.data);
            break;
        case 6: // WEATHER_UPDATE
            document.getElementById('weather').textContent = message.data.weather;
            break;
        case 7: // ACCIDENT_UPDATE
            document.getElementById('accident').textContent = message.data.accident;
            break;
    }
}

function updateMetrics(data) {
    document.getElementById('fps').textContent = data.fps.toFixed(1);
    document.getElementById('detection-ms').textContent = data.detection_ms.toFixed(2) + 'ms';
    document.getElementById('tracking-ms').textContent = data.tracking_ms.toFixed(2) + 'ms';
    document.getElementById('rl-ms').textContent = data.rl_decision_ms.toFixed(2) + 'ms';
    
    const total = data.detection_ms + data.tracking_ms + data.rl_decision_ms;
    document.getElementById('total-ms').textContent = total.toFixed(2) + 'ms';
    
    document.getElementById('vehicle-count').textContent = data.detected_vehicles;
    document.getElementById('pedestrian-count').textContent = data.detected_pedestrians;
    document.getElementById('cyclist-count').textContent = data.detected_cyclists;
    
    // Color code based on performance
    const fpsElem = document.getElementById('fps');
    fpsElem.style.color = data.fps >= 25 ? '#00FF00' : '#FF0000';
    
    const totalElem = document.getElementById('total-ms');
    totalElem.style.color = total <= 50 ? '#00FF00' : '#FF0000';
}

function updateSignals(data) {
    for (const [direction, state] of Object.entries(data)) {
        const elem = document.getElementById(`signal-${direction}`);
        if (elem) {
            elem.textContent = ['RED', 'YELLOW', 'GREEN'][state];
            elem.className = `signal-${['red', 'yellow', 'green'][state]}`;
        }
    }
}

function render() {
    // Clear canvas
    ctx.fillStyle = '#282828';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw roads
    ctx.fillStyle = '#3C3C3C';
    ctx.fillRect(0, canvas.height/2 - 50, canvas.width, 100);
    ctx.fillRect(canvas.width/2 - 50, 0, 100, canvas.height);
    
    // Draw intersection
    ctx.fillStyle = '#464646';
    ctx.fillRect(canvas.width/2 - 100, canvas.height/2 - 100, 200, 200);
    
    // Draw lane markings
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.setLineDash([20, 10]);
    ctx.beginPath();
    ctx.moveTo(0, canvas.height/2);
    ctx.lineTo(canvas.width, canvas.height/2);
    ctx.moveTo(canvas.width/2, 0);
    ctx.lineTo(canvas.width/2, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Draw crosswalks
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    const crosswalkWidth = 60;
    const stripeWidth = 5;
    
    // North crosswalk
    for (let i = -crosswalkWidth/2; i < crosswalkWidth/2; i += stripeWidth * 2) {
        ctx.fillRect(canvas.width/2 + i, canvas.height/2 - 120, stripeWidth, 20);
    }
    
    // Draw traffic signals
    drawTrafficSignals();
    
    // Draw entities
    for (const entity of entities) {
        ctx.fillStyle = entityColors[entity.type] || '#FFFFFF';
        ctx.fillRect(entity.x - entity.width/2, entity.y - entity.height/2, 
                    entity.width, entity.height);
        
        // Draw label
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(entity.label, entity.x, entity.y - entity.height/2 - 5);
    }
    
    // Draw detections (bounding boxes)
    ctx.lineWidth = 2;
    for (const det of detections) {
        const color = ['#FF0000', '#00FF00', '#0000FF'][det.class_id] || '#FFFF00';
        ctx.strokeStyle = color;
        
        const x = det.x * canvas.width;
        const y = det.y * canvas.height;
        const w = det.w * canvas.width;
        const h = det.h * canvas.height;
        
        ctx.strokeRect(x - w/2, y - h/2, w, h);
        
        // Draw confidence
        ctx.fillStyle = color;
        ctx.font = '10px Arial';
        ctx.textAlign = 'left';
        ctx.fillText((det.confidence * 100).toFixed(1) + '%', x - w/2, y - h/2 - 2);
    }
}

function drawTrafficSignals() {
    const signalSize = 30;
    const lightRadius = 8;
    
    // North signal
    drawSignal(canvas.width/2, canvas.height/2 - 150, signals.north || 0);
    
    // South signal
    drawSignal(canvas.width/2, canvas.height/2 + 150, signals.south || 0);
    
    // East signal
    drawSignal(canvas.width/2 + 150, canvas.height/2, signals.east || 0);
    
    // West signal
    drawSignal(canvas.width/2 - 150, canvas.height/2, signals.west || 0);
}

function drawSignal(x, y, state) {
    // Signal box
    ctx.fillStyle = '#000000';
    ctx.fillRect(x - 15, y - 15, 30, 30);
    
    // Signal light
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, 2 * Math.PI);
    ctx.fillStyle = signalColors[state];
    ctx.fill();
}

// Control functions
function spawnVehicle() {
    sendCommand({action: 'spawn', type: 0});
}

function spawnPedestrian() {
    sendCommand({action: 'spawn', type: 1});
}

function spawnCyclist() {
    sendCommand({action: 'spawn', type: 2});
}

function spawnObstacle() {
    sendCommand({action: 'spawn', type: 3});
}

function spawnAccident() {
    sendCommand({action: 'spawn', type: 4});
}

function clearEntities() {
    sendCommand({action: 'clear'});
}

function sendCommand(command) {
    // In a real implementation, send via WebSocket
    // For demo, we'll use fetch to send commands
    fetch('/command', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(command)
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    switch(e.key.toLowerCase()) {
        case 'v': spawnVehicle(); break;
        case 'p': spawnPedestrian(); break;
        case 'b': spawnCyclist(); break;
        case 'o': spawnObstacle(); break;
        case 'x': spawnAccident(); break;
        case 'c': clearEntities(); break;
    }
});

// Initial render
render();
)JS";
}

std::string WebFrontend::getCSSContent() {
    return R"CSS(/* TACS Web Frontend Styles */
body {
    margin: 0;
    padding: 0;
    font-family: Arial, sans-serif;
    background-color: #1a1a1a;
    color: #ffffff;
}

#container {
    max-width: 1600px;
    margin: 0 auto;
    padding: 20px;
}

h1 {
    text-align: center;
    color: #00ccff;
    margin-bottom: 20px;
}

#main-content {
    display: flex;
    gap: 20px;
}

#canvas-container {
    flex: 1;
}

#simulation-canvas {
    border: 2px solid #444;
    background-color: #222;
    display: block;
    margin-bottom: 10px;
}

#controls {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

#controls button {
    padding: 10px 15px;
    background-color: #00ccff;
    color: #000;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
    transition: background-color 0.3s;
}

#controls button:hover {
    background-color: #0099cc;
}

#sidebar {
    width: 300px;
}

.panel {
    background-color: #2a2a2a;
    border: 1px solid #444;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 20px;
}

.panel h3 {
    margin-top: 0;
    color: #00ccff;
    border-bottom: 1px solid #444;
    padding-bottom: 10px;
}

.metric-row, .detection-row, .env-row, .signal-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
}

.metric-label, .detection-label, .env-label, .signal-label {
    color: #aaa;
}

.metric-value, .detection-value, .env-value {
    font-weight: bold;
    color: #fff;
}

.signal-red {
    color: #ff0000;
    font-weight: bold;
}

.signal-yellow {
    color: #ffff00;
    font-weight: bold;
}

.signal-green {
    color: #00ff00;
    font-weight: bold;
}

#fps {
    font-size: 1.2em;
}

#total-ms {
    font-size: 1.1em;
}
)CSS";
}

} // namespace tacs