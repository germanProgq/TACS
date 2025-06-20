/*
 * Traffic-Aware Control System (TACS)
 * Kalman Filter implementation - production-ready
 */

#include "tracking/kalman_filter.h"
#include <cstring>
#include <algorithm>

namespace tacs {
namespace tracking {

KalmanFilter::KalmanFilter() : dt_(1.0f) {
    std::memset(state_, 0, sizeof(state_));
    std::memset(covariance_, 0, sizeof(covariance_));
    std::memset(predicted_state_, 0, sizeof(predicted_state_));
    std::memset(predicted_covariance_, 0, sizeof(predicted_covariance_));
    
    initialize_matrices();
}

void KalmanFilter::initialize_matrices() {
    std::memset(transition_matrix_, 0, sizeof(transition_matrix_));
    transition_matrix_[0 * STATE_DIM + 0] = 1.0f;  // x = x + dt * x_dot
    transition_matrix_[0 * STATE_DIM + 1] = dt_;
    transition_matrix_[1 * STATE_DIM + 1] = 1.0f;  // x_dot = x_dot
    transition_matrix_[2 * STATE_DIM + 2] = 1.0f;  // y = y + dt * y_dot
    transition_matrix_[2 * STATE_DIM + 3] = dt_;
    transition_matrix_[3 * STATE_DIM + 3] = 1.0f;  // y_dot = y_dot
    transition_matrix_[4 * STATE_DIM + 4] = 1.0f;  // w = w
    transition_matrix_[5 * STATE_DIM + 5] = 1.0f;  // h = h
    
    std::memset(observation_matrix_, 0, sizeof(observation_matrix_));
    observation_matrix_[0 * STATE_DIM + 0] = 1.0f;  // x
    observation_matrix_[1 * STATE_DIM + 2] = 1.0f;  // y
    observation_matrix_[2 * STATE_DIM + 4] = 1.0f;  // w
    observation_matrix_[3 * STATE_DIM + 5] = 1.0f;  // h
    
    setProcessNoise(1.0f, 0.1f, 0.01f);
    setMeasurementNoise(1.0f, 0.1f);
}

void KalmanFilter::initialize(float x, float y, float w, float h) {
    state_[0] = x;
    state_[1] = 0.0f;  // x_dot
    state_[2] = y;
    state_[3] = 0.0f;  // y_dot
    state_[4] = w;
    state_[5] = h;
    
    std::memset(covariance_, 0, sizeof(covariance_));
    covariance_[0 * STATE_DIM + 0] = 10.0f;   // x variance
    covariance_[1 * STATE_DIM + 1] = 100.0f;  // x_dot variance
    covariance_[2 * STATE_DIM + 2] = 10.0f;   // y variance
    covariance_[3 * STATE_DIM + 3] = 100.0f;  // y_dot variance
    covariance_[4 * STATE_DIM + 4] = 10.0f;   // w variance
    covariance_[5 * STATE_DIM + 5] = 10.0f;   // h variance
}

void KalmanFilter::setProcessNoise(float q_pos, float q_vel, float q_size) {
    std::memset(process_noise_, 0, sizeof(process_noise_));
    process_noise_[0 * STATE_DIM + 0] = q_pos;
    process_noise_[1 * STATE_DIM + 1] = q_vel;
    process_noise_[2 * STATE_DIM + 2] = q_pos;
    process_noise_[3 * STATE_DIM + 3] = q_vel;
    process_noise_[4 * STATE_DIM + 4] = q_size;
    process_noise_[5 * STATE_DIM + 5] = q_size;
}

void KalmanFilter::setMeasurementNoise(float r_pos, float r_size) {
    std::memset(measurement_noise_, 0, sizeof(measurement_noise_));
    measurement_noise_[0 * MEAS_DIM + 0] = r_pos;
    measurement_noise_[1 * MEAS_DIM + 1] = r_pos;
    measurement_noise_[2 * MEAS_DIM + 2] = r_size;
    measurement_noise_[3 * MEAS_DIM + 3] = r_size;
}

void KalmanFilter::predict() {
    matmul_6x6_6x1(transition_matrix_, state_, predicted_state_);
    
    matmul_6x6_6x6(transition_matrix_, covariance_, temp_matrix_state_state_);
    
    float transition_transpose[STATE_DIM * STATE_DIM];
    matrix_transpose_6x6(transition_matrix_, transition_transpose);
    
    matmul_6x6_6x6(temp_matrix_state_state_, transition_transpose, predicted_covariance_);
    
    matrix_add_6x6(predicted_covariance_, process_noise_, predicted_covariance_);
}

void KalmanFilter::update(float x, float y, float w, float h) {
    float measurement[MEAS_DIM] = {x, y, w, h};
    
    float predicted_measurement[MEAS_DIM];
    matmul_4x6_6x1(observation_matrix_, predicted_state_, predicted_measurement);
    
    matrix_sub_4x1(measurement, predicted_measurement, innovation_);
    
    matmul_4x6_6x6(observation_matrix_, predicted_covariance_, temp_matrix_meas_state_);
    
    float observation_transpose[STATE_DIM * MEAS_DIM];
    matrix_transpose_4x6(observation_matrix_, observation_transpose);
    
    matmul_4x6_6x4(temp_matrix_meas_state_, observation_transpose, innovation_covariance_);
    
    matrix_add_4x4(innovation_covariance_, measurement_noise_, innovation_covariance_);
    
    float innovation_covariance_inv[MEAS_DIM * MEAS_DIM];
    if (!matrix_inverse_4x4(innovation_covariance_, innovation_covariance_inv)) {
        std::memcpy(state_, predicted_state_, sizeof(state_));
        std::memcpy(covariance_, predicted_covariance_, sizeof(covariance_));
        return;
    }
    
    matmul_6x4_4x4(observation_transpose, innovation_covariance_inv, temp_matrix_state_meas_);
    
    matmul_6x6_6x4(predicted_covariance_, temp_matrix_state_meas_, kalman_gain_);
    
    float state_correction[STATE_DIM];
    matmul_6x4_4x1(kalman_gain_, innovation_, state_correction);
    
    for (int i = 0; i < STATE_DIM; ++i) {
        state_[i] = predicted_state_[i] + state_correction[i];
    }
    
    float kg_h[STATE_DIM * STATE_DIM];
    matmul_6x4_4x6(kalman_gain_, observation_matrix_, kg_h);
    
    float eye_minus_kg_h[STATE_DIM * STATE_DIM];
    std::memset(eye_minus_kg_h, 0, sizeof(eye_minus_kg_h));
    for (int i = 0; i < STATE_DIM; ++i) {
        eye_minus_kg_h[i * STATE_DIM + i] = 1.0f;
    }
    
    for (int i = 0; i < STATE_DIM * STATE_DIM; ++i) {
        eye_minus_kg_h[i] -= kg_h[i];
    }
    
    matmul_6x6_6x6(eye_minus_kg_h, predicted_covariance_, covariance_);
}

void KalmanFilter::getState(float& x, float& y, float& w, float& h) const {
    x = state_[0];
    y = state_[2];
    w = state_[4];
    h = state_[5];
}

void KalmanFilter::getVelocity(float& vx, float& vy) const {
    vx = state_[1];
    vy = state_[3];
}

float KalmanFilter::getMahalanobisDistance(float x, float y, float w, float h) const {
    float measurement[MEAS_DIM] = {x, y, w, h};
    float predicted_measurement[MEAS_DIM];
    matmul_4x6_6x1(observation_matrix_, state_, predicted_measurement);
    
    float diff[MEAS_DIM];
    matrix_sub_4x1(measurement, predicted_measurement, diff);
    
    float S[MEAS_DIM * MEAS_DIM];
    matmul_4x6_6x6(observation_matrix_, covariance_, temp_matrix_meas_state_);
    float observation_transpose[STATE_DIM * MEAS_DIM];
    matrix_transpose_4x6(observation_matrix_, observation_transpose);
    matmul_4x6_6x4(temp_matrix_meas_state_, observation_transpose, S);
    matrix_add_4x4(S, measurement_noise_, S);
    
    float S_inv[MEAS_DIM * MEAS_DIM];
    if (!matrix_inverse_4x4(S, S_inv)) {
        return 1e9f;
    }
    
    float temp[MEAS_DIM];
    matmul_4x4_4x1(S_inv, diff, temp);
    
    float distance = 0.0f;
    for (int i = 0; i < MEAS_DIM; ++i) {
        distance += diff[i] * temp[i];
    }
    
    return std::sqrt(distance);
}

// Matrix operations - optimized with SIMD when available

void KalmanFilter::matmul_6x6_6x1(const float* A, const float* x, float* y) const {
#if defined(__SSE__) && defined(__x86_64__)
    // SSE implementation for x86_64
    for (int i = 0; i < STATE_DIM; ++i) {
        __m128 sum = _mm_setzero_ps();
        int j;
        for (j = 0; j <= STATE_DIM - 4; j += 4) {
            __m128 a = _mm_loadu_ps(&A[i * STATE_DIM + j]);
            __m128 b = _mm_loadu_ps(&x[j]);
            sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
        }
        // Horizontal sum
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        float result = _mm_cvtss_f32(sum);
        // Handle remaining elements
        for (; j < STATE_DIM; ++j) {
            result += A[i * STATE_DIM + j] * x[j];
        }
        y[i] = result;
    }
#elif defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm__))
    // NEON implementation for ARM
    for (int i = 0; i < STATE_DIM; ++i) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        int j;
        for (j = 0; j <= STATE_DIM - 4; j += 4) {
            float32x4_t a = vld1q_f32(&A[i * STATE_DIM + j]);
            float32x4_t b = vld1q_f32(&x[j]);
            sum = vmlaq_f32(sum, a, b);
        }
        // Horizontal sum
        float32x2_t sum_low = vget_low_f32(sum);
        float32x2_t sum_high = vget_high_f32(sum);
        float32x2_t sum_pair = vadd_f32(sum_low, sum_high);
        float result = vget_lane_f32(sum_pair, 0) + vget_lane_f32(sum_pair, 1);
        // Handle remaining elements
        for (; j < STATE_DIM; ++j) {
            result += A[i * STATE_DIM + j] * x[j];
        }
        y[i] = result;
    }
#else
    // Fallback scalar implementation
    for (int i = 0; i < STATE_DIM; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < STATE_DIM; ++j) {
            sum += A[i * STATE_DIM + j] * x[j];
        }
        y[i] = sum;
    }
#endif
}

void KalmanFilter::matmul_6x6_6x6(const float* A, const float* B, float* C) const {
#if defined(__SSE__) && defined(__x86_64__)
    // SSE optimized 6x6 matrix multiplication
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            __m128 sum = _mm_setzero_ps();
            int k;
            for (k = 0; k <= STATE_DIM - 4; k += 4) {
                __m128 a = _mm_loadu_ps(&A[i * STATE_DIM + k]);
                __m128 b = _mm_set_ps(B[(k+3) * STATE_DIM + j], B[(k+2) * STATE_DIM + j],
                                     B[(k+1) * STATE_DIM + j], B[k * STATE_DIM + j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
            }
            // Horizontal sum
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            float result = _mm_cvtss_f32(sum);
            // Handle remaining elements
            for (; k < STATE_DIM; ++k) {
                result += A[i * STATE_DIM + k] * B[k * STATE_DIM + j];
            }
            C[i * STATE_DIM + j] = result;
        }
    }
#elif defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm__))
    // NEON optimized 6x6 matrix multiplication
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            int k;
            for (k = 0; k <= STATE_DIM - 4; k += 4) {
                float32x4_t a = vld1q_f32(&A[i * STATE_DIM + k]);
                float32x4_t b = {B[k * STATE_DIM + j], B[(k+1) * STATE_DIM + j],
                               B[(k+2) * STATE_DIM + j], B[(k+3) * STATE_DIM + j]};
                sum = vmlaq_f32(sum, a, b);
            }
            // Horizontal sum
            float32x2_t sum_low = vget_low_f32(sum);
            float32x2_t sum_high = vget_high_f32(sum);
            float32x2_t sum_pair = vadd_f32(sum_low, sum_high);
            float result = vget_lane_f32(sum_pair, 0) + vget_lane_f32(sum_pair, 1);
            // Handle remaining elements
            for (; k < STATE_DIM; ++k) {
                result += A[i * STATE_DIM + k] * B[k * STATE_DIM + j];
            }
            C[i * STATE_DIM + j] = result;
        }
    }
#else
    // Fallback scalar implementation
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < STATE_DIM; ++k) {
                sum += A[i * STATE_DIM + k] * B[k * STATE_DIM + j];
            }
            C[i * STATE_DIM + j] = sum;
        }
    }
#endif
}

void KalmanFilter::matmul_4x6_6x1(const float* A, const float* x, float* y) const {
#if defined(__SSE__) && defined(__x86_64__)
    // SSE implementation
    for (int i = 0; i < MEAS_DIM; ++i) {
        __m128 sum = _mm_setzero_ps();
        // Process 4 elements at a time
        __m128 a1 = _mm_loadu_ps(&A[i * STATE_DIM]);
        __m128 x1 = _mm_loadu_ps(&x[0]);
        sum = _mm_add_ps(sum, _mm_mul_ps(a1, x1));
        // Handle remaining 2 elements
        float result = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(sum, sum), sum));
        result += A[i * STATE_DIM + 4] * x[4] + A[i * STATE_DIM + 5] * x[5];
        y[i] = result;
    }
#elif defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm__))
    // NEON implementation
    for (int i = 0; i < MEAS_DIM; ++i) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        float32x4_t a1 = vld1q_f32(&A[i * STATE_DIM]);
        float32x4_t x1 = vld1q_f32(&x[0]);
        sum = vmlaq_f32(sum, a1, x1);
        // Horizontal sum and handle remaining elements
        float32x2_t sum_low = vget_low_f32(sum);
        float32x2_t sum_high = vget_high_f32(sum);
        float32x2_t sum_pair = vadd_f32(sum_low, sum_high);
        float result = vget_lane_f32(sum_pair, 0) + vget_lane_f32(sum_pair, 1);
        result += A[i * STATE_DIM + 4] * x[4] + A[i * STATE_DIM + 5] * x[5];
        y[i] = result;
    }
#else
    // Fallback scalar implementation
    for (int i = 0; i < MEAS_DIM; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < STATE_DIM; ++j) {
            sum += A[i * STATE_DIM + j] * x[j];
        }
        y[i] = sum;
    }
#endif
}

void KalmanFilter::matmul_4x6_6x6(const float* A, const float* B, float* C) const {
    for (int i = 0; i < MEAS_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < STATE_DIM; ++k) {
                sum += A[i * STATE_DIM + k] * B[k * STATE_DIM + j];
            }
            C[i * STATE_DIM + j] = sum;
        }
    }
}

void KalmanFilter::matmul_4x6_6x4(const float* A, const float* B, float* C) const {
    for (int i = 0; i < MEAS_DIM; ++i) {
        for (int j = 0; j < MEAS_DIM; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < STATE_DIM; ++k) {
                sum += A[i * STATE_DIM + k] * B[k * MEAS_DIM + j];
            }
            C[i * MEAS_DIM + j] = sum;
        }
    }
}

void KalmanFilter::matmul_6x4_4x4(const float* A, const float* B, float* C) const {
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < MEAS_DIM; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < MEAS_DIM; ++k) {
                sum += A[i * MEAS_DIM + k] * B[k * MEAS_DIM + j];
            }
            C[i * MEAS_DIM + j] = sum;
        }
    }
}

void KalmanFilter::matmul_6x4_4x1(const float* A, const float* x, float* y) const {
    for (int i = 0; i < STATE_DIM; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < MEAS_DIM; ++j) {
            sum += A[i * MEAS_DIM + j] * x[j];
        }
        y[i] = sum;
    }
}

void KalmanFilter::matmul_6x6_6x4(const float* A, const float* B, float* C) const {
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < MEAS_DIM; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < STATE_DIM; ++k) {
                sum += A[i * STATE_DIM + k] * B[k * MEAS_DIM + j];
            }
            C[i * MEAS_DIM + j] = sum;
        }
    }
}

void KalmanFilter::matmul_6x4_4x6(const float* A, const float* B, float* C) const {
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < MEAS_DIM; ++k) {
                sum += A[i * MEAS_DIM + k] * B[k * STATE_DIM + j];
            }
            C[i * STATE_DIM + j] = sum;
        }
    }
}

void KalmanFilter::matmul_4x4_4x1(const float* A, const float* x, float* y) const {
    for (int i = 0; i < MEAS_DIM; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < MEAS_DIM; ++j) {
            sum += A[i * MEAS_DIM + j] * x[j];
        }
        y[i] = sum;
    }
}

void KalmanFilter::matrix_add_6x6(const float* A, const float* B, float* C) const {
#if defined(__SSE__) && defined(__x86_64__)
    // SSE vectorized addition
    int i;
    for (i = 0; i <= STATE_DIM * STATE_DIM - 4; i += 4) {
        __m128 a = _mm_loadu_ps(&A[i]);
        __m128 b = _mm_loadu_ps(&B[i]);
        __m128 c = _mm_add_ps(a, b);
        _mm_storeu_ps(&C[i], c);
    }
    // Handle remaining elements
    for (; i < STATE_DIM * STATE_DIM; ++i) {
        C[i] = A[i] + B[i];
    }
#elif defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm__))
    // NEON vectorized addition
    int i;
    for (i = 0; i <= STATE_DIM * STATE_DIM - 4; i += 4) {
        float32x4_t a = vld1q_f32(&A[i]);
        float32x4_t b = vld1q_f32(&B[i]);
        float32x4_t c = vaddq_f32(a, b);
        vst1q_f32(&C[i], c);
    }
    // Handle remaining elements
    for (; i < STATE_DIM * STATE_DIM; ++i) {
        C[i] = A[i] + B[i];
    }
#else
    // Fallback scalar implementation
    for (int i = 0; i < STATE_DIM * STATE_DIM; ++i) {
        C[i] = A[i] + B[i];
    }
#endif
}

void KalmanFilter::matrix_add_4x4(const float* A, const float* B, float* C) const {
#if defined(__SSE__) && defined(__x86_64__)
    // SSE vectorized addition - process all 16 elements in 4 iterations
    for (int i = 0; i < MEAS_DIM * MEAS_DIM; i += 4) {
        __m128 a = _mm_loadu_ps(&A[i]);
        __m128 b = _mm_loadu_ps(&B[i]);
        __m128 c = _mm_add_ps(a, b);
        _mm_storeu_ps(&C[i], c);
    }
#elif defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm__))
    // NEON vectorized addition
    for (int i = 0; i < MEAS_DIM * MEAS_DIM; i += 4) {
        float32x4_t a = vld1q_f32(&A[i]);
        float32x4_t b = vld1q_f32(&B[i]);
        float32x4_t c = vaddq_f32(a, b);
        vst1q_f32(&C[i], c);
    }
#else
    // Fallback scalar implementation
    for (int i = 0; i < MEAS_DIM * MEAS_DIM; ++i) {
        C[i] = A[i] + B[i];
    }
#endif
}

void KalmanFilter::matrix_sub_4x1(const float* a, const float* b, float* c) const {
#if defined(__SSE__) && defined(__x86_64__)
    __m128 va = _mm_loadu_ps(a);
    __m128 vb = _mm_loadu_ps(b);
    __m128 vc = _mm_sub_ps(va, vb);
    _mm_storeu_ps(c, vc);
#elif defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm__))
    float32x4_t va = vld1q_f32(a);
    float32x4_t vb = vld1q_f32(b);
    float32x4_t vc = vsubq_f32(va, vb);
    vst1q_f32(c, vc);
#else
    for (int i = 0; i < MEAS_DIM; ++i) {
        c[i] = a[i] - b[i];
    }
#endif
}


void KalmanFilter::matrix_transpose_6x4(const float* A, float* AT) const {
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < MEAS_DIM; ++j) {
            AT[j * STATE_DIM + i] = A[i * MEAS_DIM + j];
        }
    }
}

void KalmanFilter::matrix_transpose_4x6(const float* A, float* AT) const {
    for (int i = 0; i < MEAS_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            AT[j * MEAS_DIM + i] = A[i * STATE_DIM + j];
        }
    }
}

void KalmanFilter::matrix_transpose_6x6(const float* A, float* AT) const {
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            AT[j * STATE_DIM + i] = A[i * STATE_DIM + j];
        }
    }
}

bool KalmanFilter::matrix_inverse_4x4(const float* A, float* Ainv) const {
    // Use aligned memory for better performance
    alignas(16) float a[16];
    alignas(16) float inv[16];
    std::memcpy(a, A, 16 * sizeof(float));
    
    // Calculate cofactors
    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + 
             a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - 
             a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + 
             a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - 
              a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - 
             a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + 
             a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - 
             a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + 
              a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + 
             a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - 
             a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + 
              a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - 
              a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - 
             a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + 
             a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - 
              a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + 
              a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
    
    float det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    
    // Improved numerical stability check
    const float epsilon = 1e-10f;
    if (std::abs(det) < epsilon) {
        // Set identity matrix as fallback
        std::memset(Ainv, 0, 16 * sizeof(float));
        for (int i = 0; i < 4; ++i) {
            Ainv[i * 4 + i] = 1.0f;
        }
        return false;
    }
    
    float det_inv = 1.0f / det;
    
    for (int i = 0; i < 16; ++i) {
        Ainv[i] = inv[i] * det_inv;
    }
    
    return true;
}

} // namespace tracking
} // namespace tacs