/*
 * Traffic-Aware Control System (TACS)
 * MemoryTracker implementation - Multi-object tracking with Kalman filters
 */

#include "tracking/memory_tracker.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tacs {
namespace tracking {

Track::Track(int id, int class_id, const Detection& det) 
    : id(id), class_id(class_id), 
      frames_since_update(0), hits(1), age(1), 
      confidence(det.confidence), state(TENTATIVE) {
    kf = std::make_unique<KalmanFilter>();
    kf->initialize(det.x, det.y, det.w, det.h);
}

MemoryTracker::Config::Config() {
    // Production-tuned parameters based on real-world traffic data
    
    // Car configuration - cars move faster but more predictably
    car_config.process_noise_pos = 2.0f;      // Higher due to lane changes
    car_config.process_noise_vel = 0.5f;      // Moderate acceleration changes
    car_config.process_noise_size = 0.05f;    // Cars maintain consistent size
    car_config.measurement_noise_pos = 1.0f;   // Camera vibration affects position
    car_config.measurement_noise_size = 0.1f;  // Size estimation is reliable
    
    // Pedestrian configuration - slower but more erratic
    pedestrian_config.process_noise_pos = 1.0f;      // Can change direction suddenly
    pedestrian_config.process_noise_vel = 0.3f;      // Lower max velocity
    pedestrian_config.process_noise_size = 0.1f;     // Size varies with pose
    pedestrian_config.measurement_noise_pos = 0.5f;  // Better detection accuracy
    pedestrian_config.measurement_noise_size = 0.05f; // Smaller targets
    
    // Cyclist configuration - between car and pedestrian
    cyclist_config.process_noise_pos = 1.5f;      // Moderate maneuverability
    cyclist_config.process_noise_vel = 0.4f;      // Can accelerate quickly
    cyclist_config.process_noise_size = 0.08f;    // Size varies with angle
    cyclist_config.measurement_noise_pos = 0.8f;   // Medium detection accuracy
    cyclist_config.measurement_noise_size = 0.08f; // Medium-sized targets
}

MemoryTracker::MemoryTracker() : config_(), next_track_id_(1) {}

MemoryTracker::MemoryTracker(const Config& config) 
    : config_(config), next_track_id_(1) {}

std::vector<TrackedObject> MemoryTracker::update(const std::vector<Detection>& detections) {
    // Validate input
    if (detections.size() > 10000) {  // Sanity check
        throw std::runtime_error("Too many detections: " + std::to_string(detections.size()));
    }
    
    // Predict existing tracks
    predictTracks();
    
    // Get active tracks for matching
    std::vector<Track*> active_tracks = getActiveTracks();
    
    std::vector<TrackedObject> tracked_objects;
    tracked_objects.reserve(tracks_.size());  // Pre-allocate for performance
    
    if (!active_tracks.empty() && !detections.empty()) {
        // Compute cost matrix
        std::vector<std::vector<float>> cost_matrix = computeCostMatrix(detections, active_tracks);
        
        // Solve assignment problem
        std::vector<int> assignments = hungarian_.solve(cost_matrix);
        
        // Process assignments
        std::vector<bool> track_matched(active_tracks.size(), false);
        std::vector<bool> det_matched(detections.size(), false);
        
        for (size_t i = 0; i < assignments.size(); ++i) {
            if (assignments[i] >= 0 && assignments[i] < static_cast<int>(detections.size())) {
                float cost = cost_matrix[i][assignments[i]];
                if (cost < 1.0f - config_.iou_threshold) {  // Cost is 1 - IoU
                    updateTrack(active_tracks[i], detections[assignments[i]]);
                    track_matched[i] = true;
                    det_matched[assignments[i]] = true;
                }
            }
        }
        
        // Create new tracks for unmatched detections
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!det_matched[i] && detections[i].confidence >= config_.min_confidence) {
                createNewTrack(detections[i]);
            }
        }
        
        // Mark unmatched tracks
        for (size_t i = 0; i < active_tracks.size(); ++i) {
            if (!track_matched[i]) {
                active_tracks[i]->frames_since_update++;
            }
        }
    } else if (!detections.empty()) {
        // No active tracks, create new ones
        for (const auto& det : detections) {
            if (det.confidence >= config_.min_confidence) {
                createNewTrack(det);
            }
        }
    } else {
        // No detections, just update frame counts
        for (auto& track : tracks_) {
            track->frames_since_update++;
        }
    }
    
    // Update track states
    for (auto& track : tracks_) {
        if (track->state == Track::TENTATIVE && track->hits >= config_.min_hits_to_confirm) {
            track->state = Track::CONFIRMED;
        }
        
        if (track->frames_since_update > config_.max_frames_to_skip) {
            track->state = Track::LOST;
        }
    }
    
    // Remove dead tracks
    removeDeadTracks();
    
    // Generate output
    for (const auto& track : tracks_) {
        if (track->state != Track::LOST) {
            TrackedObject obj;
            obj.track_id = track->id;
            obj.class_id = track->class_id;
            obj.confidence = track->confidence;
            obj.state = track->state;
            
            track->kf->getState(obj.x, obj.y, obj.w, obj.h);
            track->kf->getVelocity(obj.vx, obj.vy);
            
            tracked_objects.push_back(obj);
        }
    }
    
    return tracked_objects;
}

void MemoryTracker::reset() {
    tracks_.clear();
    next_track_id_ = 1;
}

int MemoryTracker::getNumActiveTracks() const {
    int count = 0;
    for (const auto& track : tracks_) {
        if (track->state != Track::LOST) {
            count++;
        }
    }
    return count;
}

int MemoryTracker::getNumConfirmedTracks() const {
    int count = 0;
    for (const auto& track : tracks_) {
        if (track->state == Track::CONFIRMED) {
            count++;
        }
    }
    return count;
}

float MemoryTracker::computeIoU(float x1, float y1, float w1, float h1,
                                float x2, float y2, float w2, float h2) const {
    // Ensure positive dimensions
    w1 = std::max(w1, 1e-6f);
    h1 = std::max(h1, 1e-6f);
    w2 = std::max(w2, 1e-6f);
    h2 = std::max(h2, 1e-6f);
    
    float x1_min = x1 - w1 / 2.0f;
    float y1_min = y1 - h1 / 2.0f;
    float x1_max = x1 + w1 / 2.0f;
    float y1_max = y1 + h1 / 2.0f;
    
    float x2_min = x2 - w2 / 2.0f;
    float y2_min = y2 - h2 / 2.0f;
    float x2_max = x2 + w2 / 2.0f;
    float y2_max = y2 + h2 / 2.0f;
    
    float inter_x_min = std::max(x1_min, x2_min);
    float inter_y_min = std::max(y1_min, y2_min);
    float inter_x_max = std::min(x1_max, x2_max);
    float inter_y_max = std::min(y1_max, y2_max);
    
    if (inter_x_max < inter_x_min || inter_y_max < inter_y_min) {
        return 0.0f;
    }
    
    float inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min);
    float area1 = w1 * h1;
    float area2 = w2 * h2;
    float union_area = area1 + area2 - inter_area;
    
    // Prevent division by zero
    if (union_area < 1e-6f) {
        return 0.0f;
    }
    
    return std::min(1.0f, inter_area / union_area);
}

std::vector<std::vector<float>> MemoryTracker::computeCostMatrix(
    const std::vector<Detection>& detections,
    const std::vector<Track*>& active_tracks) const {
    
    size_t n_tracks = active_tracks.size();
    size_t n_dets = detections.size();
    
    // Initialize with high cost for impossible assignments
    const float IMPOSSIBLE_COST = 1e9f;
    std::vector<std::vector<float>> cost_matrix(n_tracks, std::vector<float>(n_dets, IMPOSSIBLE_COST));
    
    for (size_t i = 0; i < n_tracks; ++i) {
        float track_x, track_y, track_w, track_h;
        active_tracks[i]->kf->getState(track_x, track_y, track_w, track_h);
        
        for (size_t j = 0; j < n_dets; ++j) {
            // Only consider same class assignments
            if (active_tracks[i]->class_id == detections[j].class_id) {
                float iou = computeIoU(track_x, track_y, track_w, track_h,
                                      detections[j].x, detections[j].y, 
                                      detections[j].w, detections[j].h);
                
                // Use Mahalanobis distance as secondary metric for low IoU cases
                if (iou < 0.1f) {
                    float mahal_dist = active_tracks[i]->kf->getMahalanobisDistance(
                        detections[j].x, detections[j].y, detections[j].w, detections[j].h);
                    // Normalize Mahalanobis distance to [0, 1] range
                    float normalized_dist = 1.0f / (1.0f + std::exp(-0.5f * (mahal_dist - 5.0f)));
                    cost_matrix[i][j] = std::min(1.0f - iou, normalized_dist);
                } else {
                    cost_matrix[i][j] = 1.0f - iou;  // Cost is inverse of IoU
                }
            }
        }
    }
    
    return cost_matrix;
}

void MemoryTracker::createNewTrack(const Detection& det) {
    // Prevent track ID overflow
    if (next_track_id_ > 1000000) {
        next_track_id_ = 1;  // Reset track IDs after a million tracks
    }
    
    // Limit maximum number of tracks for memory safety
    const size_t MAX_TRACKS = 1000;
    if (tracks_.size() >= MAX_TRACKS) {
        // Remove oldest tentative tracks first
        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(),
                          [](const std::unique_ptr<Track>& track) {
                              return track->state == Track::TENTATIVE && track->hits < 2;
                          }),
            tracks_.end()
        );
        
        // If still too many, don't create new track
        if (tracks_.size() >= MAX_TRACKS) {
            return;
        }
    }
    
    auto track = std::make_unique<Track>(next_track_id_++, det.class_id, det);
    
    // Set class-specific noise parameters
    Config::ClassSpecific class_config = getClassConfig(det.class_id);
    track->kf->setProcessNoise(class_config.process_noise_pos,
                              class_config.process_noise_vel,
                              class_config.process_noise_size);
    track->kf->setMeasurementNoise(class_config.measurement_noise_pos,
                                  class_config.measurement_noise_size);
    
    tracks_.push_back(std::move(track));
}

void MemoryTracker::updateTrack(Track* track, const Detection& det) {
    // Validate detection bounds
    float x = det.x;
    float y = det.y;
    float w = std::max(det.w, 1.0f);
    float h = std::max(det.h, 1.0f);
    
    track->kf->update(x, y, w, h);
    track->frames_since_update = 0;
    track->hits++;
    track->age++;
    
    // Exponential moving average for confidence with bounds checking
    track->confidence = std::max(0.0f, std::min(1.0f, 
        0.9f * track->confidence + 0.1f * det.confidence));
}

void MemoryTracker::predictTracks() {
    for (auto& track : tracks_) {
        track->kf->predict();
        track->age++;
    }
}

std::vector<Track*> MemoryTracker::getActiveTracks() {
    std::vector<Track*> active_tracks;
    for (auto& track : tracks_) {
        if (track->state != Track::LOST) {
            active_tracks.push_back(track.get());
        }
    }
    return active_tracks;
}

void MemoryTracker::removeDeadTracks() {
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                      [](const std::unique_ptr<Track>& track) {
                          return track->state == Track::LOST;
                      }),
        tracks_.end()
    );
}

MemoryTracker::Config::ClassSpecific MemoryTracker::getClassConfig(int class_id) const {
    switch (class_id) {
        case 0:  // Car
            return config_.car_config;
        case 1:  // Pedestrian
            return config_.pedestrian_config;
        case 2:  // Cyclist
            return config_.cyclist_config;
        default:
            return config_.car_config;  // Default to car config
    }
}

} // namespace tracking
} // namespace tacs