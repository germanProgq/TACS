/*
 * Traffic-Aware Control System (TACS)
 * Implements 6D state tracking: [x, x_dot, y, y_dot, w, h]
 */

#ifndef TACS_TRACKING_KALMAN_FILTER_H
#define TACS_TRACKING_KALMAN_FILTER_H

#include <vector>
#include <cmath>

#ifdef __x86_64__
    #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
#endif

namespace tacs {
namespace tracking {

class KalmanFilter {
public:
    static constexpr int STATE_DIM = 6;
    static constexpr int MEAS_DIM = 4;
    
    KalmanFilter();
    ~KalmanFilter() = default;
    
    void initialize(float x, float y, float w, float h);
    
    void predict();
    
    void update(float x, float y, float w, float h);
    
    void getState(float& x, float& y, float& w, float& h) const;
    
    void getVelocity(float& vx, float& vy) const;
    
    const float* getStateVector() const { return state_; }
    
    float getMahalanobisDistance(float x, float y, float w, float h) const;
    
    void setProcessNoise(float q_pos, float q_vel, float q_size);
    void setMeasurementNoise(float r_pos, float r_size);
    
private:
    alignas(32) float state_[STATE_DIM];
    alignas(32) float covariance_[STATE_DIM * STATE_DIM];
    alignas(32) float predicted_state_[STATE_DIM];
    alignas(32) float predicted_covariance_[STATE_DIM * STATE_DIM];
    
    alignas(32) float transition_matrix_[STATE_DIM * STATE_DIM];
    alignas(32) float observation_matrix_[MEAS_DIM * STATE_DIM];
    alignas(32) float process_noise_[STATE_DIM * STATE_DIM];
    alignas(32) float measurement_noise_[MEAS_DIM * MEAS_DIM];
    
    alignas(32) float kalman_gain_[STATE_DIM * MEAS_DIM];
    alignas(32) float innovation_[MEAS_DIM];
    alignas(32) float innovation_covariance_[MEAS_DIM * MEAS_DIM];
    
    alignas(32) mutable float temp_matrix_state_state_[STATE_DIM * STATE_DIM];
    alignas(32) mutable float temp_matrix_state_meas_[STATE_DIM * MEAS_DIM];
    alignas(32) mutable float temp_matrix_meas_state_[MEAS_DIM * STATE_DIM];
    alignas(32) mutable float temp_matrix_meas_meas_[MEAS_DIM * MEAS_DIM];
    
    float dt_;
    
    void matmul_6x6_6x1(const float* A, const float* x, float* y) const;
    void matmul_6x6_6x6(const float* A, const float* B, float* C) const;
    void matmul_4x6_6x1(const float* A, const float* x, float* y) const;
    void matmul_4x6_6x6(const float* A, const float* B, float* C) const;
    void matmul_4x6_6x4(const float* A, const float* B, float* C) const;
    void matmul_6x4_4x4(const float* A, const float* B, float* C) const;
    void matmul_6x4_4x1(const float* A, const float* x, float* y) const;
    void matmul_6x6_6x4(const float* A, const float* B, float* C) const;
    void matmul_6x4_4x6(const float* A, const float* B, float* C) const;
    void matmul_4x4_4x1(const float* A, const float* x, float* y) const;
    
    void matrix_add_6x6(const float* A, const float* B, float* C) const;
    void matrix_add_4x4(const float* A, const float* B, float* C) const;
    void matrix_sub_4x1(const float* a, const float* b, float* c) const;
    void matrix_sub_6x1(const float* a, const float* b, float* c) const;
    void matrix_transpose_6x4(const float* A, float* AT) const;
    void matrix_transpose_4x6(const float* A, float* AT) const;
    void matrix_transpose_6x6(const float* A, float* AT) const;
    
    bool matrix_inverse_4x4(const float* A, float* Ainv) const;
    
    void initialize_matrices();
};

} // namespace tracking
} // namespace tacs

#endif // TACS_TRACKING_KALMAN_FILTER_H