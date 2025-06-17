// SDL2-based simulation frontend implementation
#include "simulation/simulation_frontend.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace tacs {

SimulationFrontend::SimulationFrontend(int width, int height)
    : width_(width), height_(height), 
      window_(nullptr), renderer_(nullptr),
      running_(false), paused_(false),
      next_entity_id_(1), spawn_mode_(EntityType::VEHICLE),
      show_metrics_(true), show_bounding_boxes_(true),
      show_weather_(true), show_signals_(true),
      camera_x_(0), camera_y_(0), zoom_level_(1.0f),
      text_renderer_(nullptr) {
    
    // Initialize default intersection
    intersection_.x = width_ / 2.0f;
    intersection_.y = height_ / 2.0f;
    intersection_.size = 200.0f;
    
    // Default 4-way intersection lanes
    intersection_.lanes = {
        {intersection_.x - intersection_.size, intersection_.y},  // West
        {intersection_.x + intersection_.size, intersection_.y},  // East
        {intersection_.x, intersection_.y - intersection_.size},  // North
        {intersection_.x, intersection_.y + intersection_.size}   // South
    };
    
    // Default signal states
    intersection_.signals["north"] = SignalState::GREEN;
    intersection_.signals["south"] = SignalState::GREEN;
    intersection_.signals["east"] = SignalState::RED;
    intersection_.signals["west"] = SignalState::RED;
}

SimulationFrontend::~SimulationFrontend() {
    if (renderer_) SDL_DestroyRenderer(renderer_);
    if (window_) SDL_DestroyWindow(window_);
    SDL_Quit();
}

bool SimulationFrontend::initialize() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
        return false;
    }
    
    window_ = SDL_CreateWindow(
        "TACS Simulation Frontend",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        width_, height_,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
    );
    
    if (!window_) {
        std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
        return false;
    }
    
    renderer_ = SDL_CreateRenderer(
        window_, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
    );
    
    if (!renderer_) {
        std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
        return false;
    }
    
    // Enable alpha blending
    SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);
    
    // Initialize text renderer
    text_renderer_ = std::make_unique<TextRenderer>(renderer_);
    if (!text_renderer_->initialize()) {
        std::cerr << "Text renderer initialization failed" << std::endl;
        return false;
    }
    
    running_ = true;
    last_frame_time_ = std::chrono::high_resolution_clock::now();
    
    return true;
}

void SimulationFrontend::run() {
    if (!running_) return;
    
    while (running_) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        handleEvents();
        
        if (!paused_) {
            // Update simulation
            float dt = 0.016f; // 60 FPS target
            updateEntities(dt);
            
            // Run TACS pipeline if connected
            if (tacs_pipeline_) {
                // Create synthetic image from simulation state
                // Capture current frame to tensor for AI processing
                Tensor input(1, 3, 416, 416);
                captureFrameToTensor(input);
                
                auto pipeline_start = std::chrono::high_resolution_clock::now();
                // Process frame through pipeline
                auto pipeline_output = tacs_pipeline_->processFrame(input);
                auto pipeline_end = std::chrono::high_resolution_clock::now();
                
                {
                    std::lock_guard<std::mutex> lock(detection_mutex_);
                    // Convert from pipeline detections to utils::Detection format
                    last_detections_.clear();
                    for (const auto& det : pipeline_output.detections) {
                        last_detections_.emplace_back(det.x, det.y, det.w, det.h, 
                                                    det.confidence, det.class_id);
                    }
                }
                
                // Update metrics
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    pipeline_end - pipeline_start).count() / 1000.0f;
                
                {
                    std::lock_guard<std::mutex> lock(metrics_mutex_);
                    metrics_.detection_ms = duration;
                }
            }
            
            // Run RL policy if connected
            if (rl_policy_) {
                auto rl_start = std::chrono::high_resolution_clock::now();
                
                // Prepare RL state
                RLState rl_state;
                rl_state.queue_lengths = {5.0f, 3.0f, 7.0f, 2.0f};  // Example queue lengths
                rl_state.pedestrian_counts = {2.0f, 1.0f, 0.0f, 3.0f};
                rl_state.cyclist_counts = {1.0f, 0.0f, 2.0f, 1.0f};
                rl_state.weather_condition = 0.0f;  // Clear
                rl_state.accident_indicator = 0.0f;  // No accident
                rl_state.current_phase_duration = 10.0f;
                rl_state.time_of_day = 0.5f;  // Midday
                
                SignalPhase action = rl_policy_->selectAction(rl_state);
                
                // Map action to signal changes
                if (action == SignalPhase::NS_GREEN_EW_RED) {
                    // North-South green
                    intersection_.signals["north"] = SignalState::GREEN;
                    intersection_.signals["south"] = SignalState::GREEN;
                    intersection_.signals["east"] = SignalState::RED;
                    intersection_.signals["west"] = SignalState::RED;
                } else if (action == SignalPhase::NS_RED_EW_GREEN) {
                    // East-West green
                    intersection_.signals["north"] = SignalState::RED;
                    intersection_.signals["south"] = SignalState::RED;
                    intersection_.signals["east"] = SignalState::GREEN;
                    intersection_.signals["west"] = SignalState::GREEN;
                }
                
                auto rl_end = std::chrono::high_resolution_clock::now();
                auto rl_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    rl_end - rl_start).count() / 1000.0f;
                
                {
                    std::lock_guard<std::mutex> lock(metrics_mutex_);
                    metrics_.rl_decision_ms = rl_duration;
                }
            }
        }
        
        render();
        updateFrameTime();
        
        // Cap at 60 FPS
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            frame_end - frame_start);
        if (frame_duration.count() < 16667) {  // 16.67ms = 60 FPS
            SDL_Delay((16667 - frame_duration.count()) / 1000);
        }
    }
}

void SimulationFrontend::setTACSPipeline(std::shared_ptr<TACSpipeline> pipeline) {
    tacs_pipeline_ = pipeline;
}

void SimulationFrontend::setRLPolicy(std::shared_ptr<RLPolicyNet> policy) {
    rl_policy_ = policy;
}

void SimulationFrontend::spawnEntity(EntityType type, float x, float y) {
    std::lock_guard<std::mutex> lock(entity_mutex_);
    
    SimulatedEntity entity;
    entity.type = type;
    entity.x = x;
    entity.y = y;
    entity.id = next_entity_id_++;
    entity.color = getEntityColor(type);
    
    // Set default sizes and velocities based on type
    switch (type) {
        case EntityType::VEHICLE:
            entity.width = 40;
            entity.height = 20;
            entity.vx = (rand() % 40 - 20);
            entity.vy = (rand() % 40 - 20);
            entity.label = "Vehicle";
            break;
        case EntityType::PEDESTRIAN:
            entity.width = 10;
            entity.height = 10;
            entity.vx = (rand() % 10 - 5);
            entity.vy = (rand() % 10 - 5);
            entity.label = "Pedestrian";
            break;
        case EntityType::CYCLIST:
            entity.width = 15;
            entity.height = 15;
            entity.vx = (rand() % 20 - 10);
            entity.vy = (rand() % 20 - 10);
            entity.label = "Cyclist";
            break;
        case EntityType::OBSTACLE:
            entity.width = 30;
            entity.height = 30;
            entity.vx = 0;
            entity.vy = 0;
            entity.label = "Obstacle";
            break;
        case EntityType::ACCIDENT_REAR_END:
            entity.width = 60;
            entity.height = 30;
            entity.vx = 0;
            entity.vy = 0;
            entity.label = "Accident: Rear-End";
            break;
        case EntityType::ACCIDENT_SIDE_IMPACT:
            entity.width = 50;
            entity.height = 50;
            entity.vx = 0;
            entity.vy = 0;
            entity.label = "Accident: Side-Impact";
            break;
        case EntityType::ACCIDENT_PILE_UP:
            entity.width = 80;
            entity.height = 40;
            entity.vx = 0;
            entity.vy = 0;
            entity.label = "Accident: Pile-Up";
            break;
    }
    
    entities_.push_back(entity);
}

void SimulationFrontend::updateEntities(float dt) {
    std::lock_guard<std::mutex> lock(entity_mutex_);
    
    for (auto& entity : entities_) {
        // Skip static entities
        if (entity.type == EntityType::OBSTACLE ||
            entity.type == EntityType::ACCIDENT_REAR_END ||
            entity.type == EntityType::ACCIDENT_SIDE_IMPACT ||
            entity.type == EntityType::ACCIDENT_PILE_UP) {
            continue;
        }
        
        // Update position
        entity.x += entity.vx * dt;
        entity.y += entity.vy * dt;
        
        // Simple boundary bounce
        if (entity.x < 0 || entity.x > width_) {
            entity.vx = -entity.vx;
            entity.x = std::max(0.0f, std::min((float)width_, entity.x));
        }
        if (entity.y < 0 || entity.y > height_) {
            entity.vy = -entity.vy;
            entity.y = std::max(0.0f, std::min((float)height_, entity.y));
        }
        
        // Traffic light logic for vehicles
        if (entity.type == EntityType::VEHICLE) {
            // Check if approaching intersection
            float dist_to_intersection = std::sqrt(
                std::pow(entity.x - intersection_.x, 2) + 
                std::pow(entity.y - intersection_.y, 2)
            );
            
            if (dist_to_intersection < intersection_.size) {
                // Determine direction
                std::string direction;
                if (std::abs(entity.x - intersection_.x) > std::abs(entity.y - intersection_.y)) {
                    direction = entity.x < intersection_.x ? "west" : "east";
                } else {
                    direction = entity.y < intersection_.y ? "north" : "south";
                }
                
                // Stop at red light
                if (intersection_.signals[direction] == SignalState::RED) {
                    entity.vx *= 0.9f;  // Gradual stop
                    entity.vy *= 0.9f;
                }
            }
        }
    }
}

void SimulationFrontend::clearEntities() {
    std::lock_guard<std::mutex> lock(entity_mutex_);
    entities_.clear();
}

void SimulationFrontend::setupIntersection(const IntersectionConfig& config) {
    intersection_ = config;
}

void SimulationFrontend::updateSignalState(const std::string& direction, SignalState state) {
    intersection_.signals[direction] = state;
}

void SimulationFrontend::updateMetrics(const PerformanceMetrics& metrics) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_ = metrics;
}

void SimulationFrontend::render() {
    auto render_start = std::chrono::high_resolution_clock::now();
    
    // Clear screen
    SDL_SetRenderDrawColor(renderer_, 40, 40, 40, 255);
    SDL_RenderClear(renderer_);
    
    // Apply camera transform
    SDL_RenderSetScale(renderer_, zoom_level_, zoom_level_);
    
    // Render layers in order
    renderIntersection();
    renderEntities();
    if (show_bounding_boxes_) renderDetections();
    if (show_signals_) renderSignals();
    if (show_weather_) renderWeatherOverlay();
    if (show_metrics_) renderPerformanceMetrics();
    
    // Reset scale for UI elements
    SDL_RenderSetScale(renderer_, 1.0f, 1.0f);
    
    SDL_RenderPresent(renderer_);
    
    auto render_end = std::chrono::high_resolution_clock::now();
    auto render_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        render_end - render_start).count() / 1000.0f;
    
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.render_ms = render_duration;
    }
}

void SimulationFrontend::renderIntersection() {
    // Draw road surface
    SDL_SetRenderDrawColor(renderer_, 60, 60, 60, 255);
    
    // Horizontal road
    SDL_Rect h_road = {
        0, 
        (int)(intersection_.y - 50),
        width_,
        100
    };
    SDL_RenderFillRect(renderer_, &h_road);
    
    // Vertical road
    SDL_Rect v_road = {
        (int)(intersection_.x - 50),
        0,
        100,
        height_
    };
    SDL_RenderFillRect(renderer_, &v_road);
    
    // Draw intersection center
    SDL_SetRenderDrawColor(renderer_, 70, 70, 70, 255);
    SDL_Rect intersection_rect = {
        (int)(intersection_.x - intersection_.size/2),
        (int)(intersection_.y - intersection_.size/2),
        (int)intersection_.size,
        (int)intersection_.size
    };
    SDL_RenderFillRect(renderer_, &intersection_rect);
    
    // Draw lane markings
    SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 255);
    
    // Center lines
    for (int x = 0; x < width_; x += 30) {
        if (std::abs(x - intersection_.x) > intersection_.size/2) {
            SDL_Rect line = {x, (int)intersection_.y - 2, 20, 4};
            SDL_RenderFillRect(renderer_, &line);
        }
    }
    
    for (int y = 0; y < height_; y += 30) {
        if (std::abs(y - intersection_.y) > intersection_.size/2) {
            SDL_Rect line = {(int)intersection_.x - 2, y, 4, 20};
            SDL_RenderFillRect(renderer_, &line);
        }
    }
    
    // Crosswalks
    SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 200);
    int crosswalk_width = 60;
    int stripe_width = 5;
    
    // North crosswalk
    for (int i = -crosswalk_width/2; i < crosswalk_width/2; i += stripe_width * 2) {
        SDL_Rect stripe = {
            (int)(intersection_.x + i),
            (int)(intersection_.y - intersection_.size/2 - 20),
            stripe_width,
            20
        };
        SDL_RenderFillRect(renderer_, &stripe);
    }
    
    // Similar for other directions...
}

void SimulationFrontend::renderEntities() {
    std::lock_guard<std::mutex> lock(entity_mutex_);
    
    for (const auto& entity : entities_) {
        SDL_SetRenderDrawColor(renderer_, 
            (entity.color >> 16) & 0xFF,
            (entity.color >> 8) & 0xFF,
            entity.color & 0xFF,
            255);
        
        SDL_Rect rect = {
            (int)(entity.x - entity.width/2),
            (int)(entity.y - entity.height/2),
            (int)entity.width,
            (int)entity.height
        };
        
        SDL_RenderFillRect(renderer_, &rect);
        
        // Draw label if close enough
        if (zoom_level_ > 0.8f && text_renderer_) {
            // Render entity label with production text renderer
            int text_width, text_height;
            text_renderer_->getTextSize(entity.label, text_width, text_height);
            
            int label_x = (int)(entity.x - text_width/2);
            int label_y = (int)(entity.y - entity.height/2 - text_height - 4);
            
            // Draw background for better readability
            SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 180);
            SDL_Rect label_bg = {
                label_x - 2,
                label_y - 2,
                text_width + 4,
                text_height + 4
            };
            SDL_RenderFillRect(renderer_, &label_bg);
            
            // Render text
            text_renderer_->renderText(entity.label, label_x, label_y, 255, 255, 255);
        }
    }
}

void SimulationFrontend::renderDetections() {
    std::lock_guard<std::mutex> lock(detection_mutex_);
    
    for (const auto& detection : last_detections_) {
        uint32_t color = getClassColor(detection.class_id);
        
        // Convert detection coordinates to screen space
        float x = detection.x * width_;
        float y = detection.y * height_;
        float w = detection.w * width_;
        float h = detection.h * height_;
        
        // Get class label from class_id
        std::string class_label;
        switch (detection.class_id) {
            case 0: class_label = "Vehicle"; break;
            case 1: class_label = "Pedestrian"; break;
            case 2: class_label = "Cyclist"; break;
            default: class_label = "Unknown"; break;
        }
        renderBoundingBox(x, y, w, h, color, class_label);
    }
}

void SimulationFrontend::renderSignals() {
    int signal_size = 30;
    int light_radius = 8;
    
    // North signal
    SDL_Rect north_signal = {
        (int)(intersection_.x - signal_size/2),
        (int)(intersection_.y - intersection_.size/2 - signal_size - 10),
        signal_size,
        signal_size
    };
    SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
    SDL_RenderFillRect(renderer_, &north_signal);
    
    // Draw light
    auto north_state = intersection_.signals["north"];
    if (north_state == SignalState::RED) {
        SDL_SetRenderDrawColor(renderer_, 255, 0, 0, 255);
    } else if (north_state == SignalState::YELLOW) {
        SDL_SetRenderDrawColor(renderer_, 255, 255, 0, 255);
    } else {
        SDL_SetRenderDrawColor(renderer_, 0, 255, 0, 255);
    }
    
    // Draw traffic light circle with production-ready implementation
    DrawUtils::drawFilledCircle(renderer_, 
        north_signal.x + signal_size/2,
        north_signal.y + signal_size/2,
        light_radius);
    
    // Repeat for other directions...
}

void SimulationFrontend::renderWeatherOverlay() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    if (metrics_.weather_condition == "rain") {
        // Rain effect
        SDL_SetRenderDrawColor(renderer_, 100, 100, 150, 30);
        SDL_Rect overlay = {0, 0, width_, height_};
        SDL_RenderFillRect(renderer_, &overlay);
        
        // Rain drops
        SDL_SetRenderDrawColor(renderer_, 200, 200, 255, 100);
        for (int i = 0; i < 100; i++) {
            int x = rand() % width_;
            int y = rand() % height_;
            SDL_RenderDrawLine(renderer_, x, y, x - 2, y + 10);
        }
    } else if (metrics_.weather_condition == "fog") {
        // Fog effect
        SDL_SetRenderDrawColor(renderer_, 200, 200, 200, 80);
        SDL_Rect overlay = {0, 0, width_, height_};
        SDL_RenderFillRect(renderer_, &overlay);
    } else if (metrics_.weather_condition == "snow") {
        // Snow effect
        SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 40);
        SDL_Rect overlay = {0, 0, width_, height_};
        SDL_RenderFillRect(renderer_, &overlay);
        
        // Snow particles
        SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 200);
        for (int i = 0; i < 50; i++) {
            int x = rand() % width_;
            int y = rand() % height_;
            SDL_Rect snowflake = {x, y, 3, 3};
            SDL_RenderFillRect(renderer_, &snowflake);
        }
    }
}

void SimulationFrontend::renderPerformanceMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Background for metrics
    SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 180);
    SDL_Rect bg = {10, 10, 300, 250};
    SDL_RenderFillRect(renderer_, &bg);
    
    // Border
    SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 255);
    SDL_RenderDrawRect(renderer_, &bg);
    
    // Render metrics with production text renderer
    if (!text_renderer_) return;
    
    int y_offset = 20;
    char buffer[256];
    
    // Title
    text_renderer_->renderText("PERFORMANCE METRICS", 20, y_offset, 255, 255, 255);
    y_offset += 20;
    
    // FPS
    snprintf(buffer, sizeof(buffer), "FPS: %.1f", metrics_.fps);
    uint8_t fps_color = metrics_.fps >= 55.0f ? 0 : 255;
    text_renderer_->renderText(buffer, 20, y_offset, 255-fps_color, 255, fps_color);
    y_offset += 20;
    
    // Detection latency
    snprintf(buffer, sizeof(buffer), "Detection: %.2f ms", metrics_.detection_ms);
    uint8_t det_color = metrics_.detection_ms <= 50.0f ? 0 : 255;
    text_renderer_->renderText(buffer, 20, y_offset, 255-det_color, 255, det_color);
    y_offset += 20;
    
    // Tracking latency
    snprintf(buffer, sizeof(buffer), "Tracking: %.2f ms", metrics_.tracking_ms);
    uint8_t track_color = metrics_.tracking_ms <= 10.0f ? 0 : 255;
    text_renderer_->renderText(buffer, 20, y_offset, 255-track_color, 255, track_color);
    y_offset += 20;
    
    // RL decision latency
    snprintf(buffer, sizeof(buffer), "RL Decision: %.2f ms", metrics_.rl_decision_ms);
    uint8_t rl_color = metrics_.rl_decision_ms <= 3.0f ? 0 : 255;
    text_renderer_->renderText(buffer, 20, y_offset, 255-rl_color, 255, rl_color);
    y_offset += 20;
    
    // Total pipeline latency
    float total_ms = metrics_.detection_ms + metrics_.tracking_ms + metrics_.rl_decision_ms;
    snprintf(buffer, sizeof(buffer), "Total Pipeline: %.2f ms", total_ms);
    uint8_t total_color = total_ms <= 50.0f ? 0 : 255;
    text_renderer_->renderText(buffer, 20, y_offset, 255-total_color, 255, total_color);
    y_offset += 30;
    
    // Detection counts
    text_renderer_->renderText("DETECTIONS:", 20, y_offset, 255, 255, 255);
    y_offset += 20;
    snprintf(buffer, sizeof(buffer), "Vehicles: %d", metrics_.detected_vehicles);
    text_renderer_->renderText(buffer, 20, y_offset, 255, 100, 100);
    
    snprintf(buffer, sizeof(buffer), "Pedestrians: %d", metrics_.detected_pedestrians);
    text_renderer_->renderText(buffer, 120, y_offset, 100, 255, 100);
    
    snprintf(buffer, sizeof(buffer), "Cyclists: %d", metrics_.detected_cyclists);
    text_renderer_->renderText(buffer, 220, y_offset, 100, 100, 255);
    y_offset += 25;
    
    // Weather and accident status
    text_renderer_->renderText("STATUS:", 20, y_offset, 255, 255, 255);
    y_offset += 20;
    
    snprintf(buffer, sizeof(buffer), "Weather: %s", metrics_.weather_condition.c_str());
    text_renderer_->renderText(buffer, 20, y_offset, 200, 200, 255);
    y_offset += 20;
    
    snprintf(buffer, sizeof(buffer), "Accident: %s", metrics_.accident_status.c_str());
    uint8_t accident_r = metrics_.accident_status == "none" ? 100 : 255;
    uint8_t accident_g = metrics_.accident_status == "none" ? 255 : 100;
    text_renderer_->renderText(buffer, 20, y_offset, accident_r, accident_g, 100);
}

void SimulationFrontend::renderBoundingBox(float x, float y, float w, float h, 
                                          uint32_t color, const std::string& label) {
    SDL_SetRenderDrawColor(renderer_,
        (color >> 16) & 0xFF,
        (color >> 8) & 0xFF,
        color & 0xFF,
        255);
    
    // Draw box outline
    SDL_Rect box = {
        (int)(x - w/2),
        (int)(y - h/2),
        (int)w,
        (int)h
    };
    SDL_RenderDrawRect(renderer_, &box);
    
    // Draw thicker corners
    int corner_length = 10;
    SDL_Rect corners[] = {
        // Top-left
        {box.x, box.y, corner_length, 2},
        {box.x, box.y, 2, corner_length},
        // Top-right
        {box.x + box.w - corner_length, box.y, corner_length, 2},
        {box.x + box.w - 2, box.y, 2, corner_length},
        // Bottom-left
        {box.x, box.y + box.h - 2, corner_length, 2},
        {box.x, box.y + box.h - corner_length, 2, corner_length},
        // Bottom-right
        {box.x + box.w - corner_length, box.y + box.h - 2, corner_length, 2},
        {box.x + box.w - 2, box.y + box.h - corner_length, 2, corner_length}
    };
    
    for (const auto& corner : corners) {
        SDL_RenderFillRect(renderer_, &corner);
    }
    
    // Label background
    if (!label.empty()) {
        SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 180);
        SDL_Rect label_bg = {
            box.x,
            box.y - 20,
            (int)(label.length() * 8),
            18
        };
        SDL_RenderFillRect(renderer_, &label_bg);
    }
}

void SimulationFrontend::handleEvents() {
    SDL_Event event;
    
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                running_ = false;
                break;
                
            case SDL_KEYDOWN:
                handleKeyPress(event.key.keysym.sym);
                break;
                
            case SDL_MOUSEBUTTONDOWN:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    handleMouseClick(event.button.x, event.button.y);
                }
                break;
                
            case SDL_MOUSEWHEEL:
                // Zoom
                if (event.wheel.y > 0) {
                    zoom_level_ = std::min(zoom_level_ * 1.1f, 3.0f);
                } else if (event.wheel.y < 0) {
                    zoom_level_ = std::max(zoom_level_ * 0.9f, 0.5f);
                }
                break;
                
            case SDL_WINDOWEVENT:
                if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
                    width_ = event.window.data1;
                    height_ = event.window.data2;
                }
                break;
        }
    }
}

void SimulationFrontend::handleMouseClick(int x, int y) {
    auto [wx, wy] = screenToWorld(x, y);
    spawnEntity(spawn_mode_, wx, wy);
}

void SimulationFrontend::handleKeyPress(SDL_Keycode key) {
    switch (key) {
        case SDLK_SPACE:
            paused_ = !paused_;
            break;
            
        case SDLK_c:
            clearEntities();
            break;
            
        case SDLK_o:
            spawn_mode_ = EntityType::OBSTACLE;
            break;
            
        case SDLK_x:
            // Cycle through accident types
            if (spawn_mode_ == EntityType::ACCIDENT_REAR_END) {
                spawn_mode_ = EntityType::ACCIDENT_SIDE_IMPACT;
            } else if (spawn_mode_ == EntityType::ACCIDENT_SIDE_IMPACT) {
                spawn_mode_ = EntityType::ACCIDENT_PILE_UP;
            } else {
                spawn_mode_ = EntityType::ACCIDENT_REAR_END;
            }
            break;
            
        case SDLK_v:
            spawn_mode_ = EntityType::VEHICLE;
            break;
            
        case SDLK_p:
            spawn_mode_ = EntityType::PEDESTRIAN;
            break;
            
        case SDLK_b:
            spawn_mode_ = EntityType::CYCLIST;
            break;
            
        case SDLK_r:
            // Reload/retrain signal - reload all models from disk
            if (tacs_pipeline_) {
                std::cout << "Reloading models from disk..." << std::endl;
                
                // Reload TACSNet
                if (tacs_pipeline_->getTACSNet()) {
                    try {
                        tacs_pipeline_->getTACSNet()->load_model("models/tacsnet.bin");
                        std::cout << "TACSNet reloaded successfully" << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Failed to reload TACSNet: " << e.what() << std::endl;
                    }
                }
                
                // Reload AccidentNet
                if (tacs_pipeline_->getAccidentNet()) {
                    try {
                        tacs_pipeline_->getAccidentNet()->loadModel("models/accidentnet.bin");
                        std::cout << "AccidentNet reloaded successfully" << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Failed to reload AccidentNet: " << e.what() << std::endl;
                    }
                }
                
                // Reload WeatherNet
                if (tacs_pipeline_->getWeatherNet()) {
                    try {
                        tacs_pipeline_->getWeatherNet()->loadModel("models/weathernet.bin");
                        std::cout << "WeatherNet reloaded successfully" << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Failed to reload WeatherNet: " << e.what() << std::endl;
                    }
                }
                
                std::cout << "Model reload completed" << std::endl;
            }
            
            // Reload RL policy
            if (rl_policy_) {
                try {
                    rl_policy_->loadWeights("models/rl_policy.bin");
                    std::cout << "RL Policy reloaded successfully" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Failed to reload RL Policy: " << e.what() << std::endl;
                }
            }
            break;
            
        case SDLK_m:
            show_metrics_ = !show_metrics_;
            break;
            
        case SDLK_d:
            show_bounding_boxes_ = !show_bounding_boxes_;
            break;
            
        case SDLK_w:
            show_weather_ = !show_weather_;
            break;
            
        case SDLK_s:
            show_signals_ = !show_signals_;
            break;
            
        case SDLK_ESCAPE:
            running_ = false;
            break;
            
        // Camera controls
        case SDLK_LEFT:
            camera_x_ -= 20;
            break;
        case SDLK_RIGHT:
            camera_x_ += 20;
            break;
        case SDLK_UP:
            camera_y_ -= 20;
            break;
        case SDLK_DOWN:
            camera_y_ += 20;
            break;
        case SDLK_HOME:
            camera_x_ = 0;
            camera_y_ = 0;
            zoom_level_ = 1.0f;
            break;
    }
}

std::pair<float, float> SimulationFrontend::screenToWorld(int sx, int sy) {
    float wx = (sx / zoom_level_) + camera_x_;
    float wy = (sy / zoom_level_) + camera_y_;
    return {wx, wy};
}

std::pair<int, int> SimulationFrontend::worldToScreen(float wx, float wy) {
    int sx = (int)((wx - camera_x_) * zoom_level_);
    int sy = (int)((wy - camera_y_) * zoom_level_);
    return {sx, sy};
}

uint32_t SimulationFrontend::getEntityColor(EntityType type) {
    switch (type) {
        case EntityType::VEHICLE:
            return 0xFF8080;  // Light red
        case EntityType::PEDESTRIAN:
            return 0x80FF80;  // Light green
        case EntityType::CYCLIST:
            return 0x8080FF;  // Light blue
        case EntityType::OBSTACLE:
            return 0xFFFF00;  // Yellow
        case EntityType::ACCIDENT_REAR_END:
        case EntityType::ACCIDENT_SIDE_IMPACT:
        case EntityType::ACCIDENT_PILE_UP:
            return 0xFF0000;  // Red
        default:
            return 0xFFFFFF;  // White
    }
}

uint32_t SimulationFrontend::getClassColor(int class_id) {
    switch (class_id) {
        case 0:  // Vehicle
            return 0xFF0000;  // Red
        case 1:  // Pedestrian
            return 0x00FF00;  // Green
        case 2:  // Cyclist
            return 0x0000FF;  // Blue
        default:
            return 0xFFFF00;  // Yellow
    }
}

void SimulationFrontend::updateFrameTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        now - last_frame_time_);
    
    float current_fps = 1000000.0f / duration.count();
    
    // Moving average for smoother FPS display
    fps_history_.push_back(current_fps);
    if (fps_history_.size() > 30) {
        fps_history_.erase(fps_history_.begin());
    }
    
    float avg_fps = 0;
    for (float fps : fps_history_) {
        avg_fps += fps;
    }
    avg_fps /= fps_history_.size();
    
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.fps = avg_fps;
    }
    
    last_frame_time_ = now;
}

void SimulationFrontend::captureFrameToTensor(Tensor& output) {
    // Create surface to capture current frame
    SDL_Surface* surface = SDL_CreateRGBSurface(0, 416, 416, 32,
                                                0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000);
    if (!surface) return;
    
    // Read pixels from renderer
    SDL_Rect capture_rect = {
        (int)(camera_x_ + width_/2 - 208),
        (int)(camera_y_ + height_/2 - 208),
        416, 416
    };
    
    SDL_RenderReadPixels(renderer_, &capture_rect, 
                        SDL_PIXELFORMAT_ARGB8888, surface->pixels, surface->pitch);
    
    // Convert to tensor format (CHW)
    SDL_LockSurface(surface);
    uint8_t* pixels = (uint8_t*)surface->pixels;
    
    for (int y = 0; y < 416; y++) {
        for (int x = 0; x < 416; x++) {
            int idx = y * surface->pitch + x * 4;
            float r = pixels[idx + 2] / 255.0f;
            float g = pixels[idx + 1] / 255.0f;
            float b = pixels[idx + 0] / 255.0f;
            
            // Fill tensor in CHW format
            output.at(0, 0, y, x) = r;
            output.at(0, 1, y, x) = g;
            output.at(0, 2, y, x) = b;
        }
    }
    
    SDL_UnlockSurface(surface);
    SDL_FreeSurface(surface);
}

} // namespace tacs