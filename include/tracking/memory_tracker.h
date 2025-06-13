/*
 * Traffic-Aware Control System (TACS)
 * MemoryTracker - Multi-object tracking system with Kalman filters and Hungarian assignment
 */

#ifndef TACS_TRACKING_MEMORY_TRACKER_H
#define TACS_TRACKING_MEMORY_TRACKER_H

#include "tracking/kalman_filter.h"
#include "tracking/hungarian_algorithm.h"
#include <vector>
#include <memory>
#include <unordered_map>

namespace tacs {
namespace tracking {

struct Detection {
    float x, y, w, h;
    float confidence;
    int class_id;  // 0: car, 1: pedestrian, 2: cyclist
};

struct Track {
    int id;
    int class_id;
    std::unique_ptr<KalmanFilter> kf;
    int frames_since_update;
    int hits;
    int age;
    float confidence;
    
    enum State {
        TENTATIVE,
        CONFIRMED,
        LOST
    } state;
    
    Track(int id, int class_id, const Detection& det);
};

struct TrackedObject {
    int track_id;
    float x, y, w, h;
    float vx, vy;
    float confidence;
    int class_id;
    Track::State state;
};

class MemoryTracker {
public:
    struct Config {
        float iou_threshold = 0.3f;
        int max_frames_to_skip = 10;
        int min_hits_to_confirm = 3;
        float min_confidence = 0.4f;
        
        struct ClassSpecific {
            float process_noise_pos = 1.0f;
            float process_noise_vel = 0.1f;
            float process_noise_size = 0.01f;
            float measurement_noise_pos = 1.0f;
            float measurement_noise_size = 0.1f;
        };
        
        ClassSpecific car_config;
        ClassSpecific pedestrian_config;
        ClassSpecific cyclist_config;
        
        Config();
    };
    
    MemoryTracker();
    explicit MemoryTracker(const Config& config);
    ~MemoryTracker() = default;
    
    std::vector<TrackedObject> update(const std::vector<Detection>& detections);
    
    void setConfig(const Config& config) { config_ = config; }
    const Config& getConfig() const { return config_; }
    
    void reset();
    
    int getNumActiveTracks() const;
    int getNumConfirmedTracks() const;
    
private:
    Config config_;
    std::vector<std::unique_ptr<Track>> tracks_;
    HungarianAlgorithm hungarian_;
    int next_track_id_;
    
    float computeIoU(float x1, float y1, float w1, float h1,
                     float x2, float y2, float w2, float h2) const;
    
    std::vector<std::vector<float>> computeCostMatrix(
        const std::vector<Detection>& detections,
        const std::vector<Track*>& active_tracks) const;
    
    void createNewTrack(const Detection& det);
    
    void updateTrack(Track* track, const Detection& det);
    
    void predictTracks();
    
    std::vector<Track*> getActiveTracks();
    
    void removeDeadTracks();
    
    Config::ClassSpecific getClassConfig(int class_id) const;
};

} // namespace tracking
} // namespace tacs

#endif // TACS_TRACKING_MEMORY_TRACKER_H