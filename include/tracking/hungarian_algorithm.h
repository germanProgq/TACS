/*
 * Traffic-Aware Control System (TACS)
 * Hungarian Algorithm for optimal assignment in multi-object tracking
 */

#ifndef TACS_TRACKING_HUNGARIAN_ALGORITHM_H
#define TACS_TRACKING_HUNGARIAN_ALGORITHM_H

#include <vector>
#include <limits>
#include <cstdint>

namespace tacs {
namespace tracking {

class HungarianAlgorithm {
public:
    HungarianAlgorithm() = default;
    ~HungarianAlgorithm() = default;
    
    std::vector<int> solve(const std::vector<std::vector<float>>& cost_matrix);
    
    float computeTotalCost(const std::vector<std::vector<float>>& cost_matrix,
                          const std::vector<int>& assignment) const;
    
private:
    static constexpr float INF = 1e9f;
    
    struct WorkArrays {
        std::vector<float> u;
        std::vector<float> v;
        std::vector<int> p;
        std::vector<int> way;
        std::vector<float> minv;
        std::vector<bool> used;
        
        void resize(size_t n, size_t m) {
            u.resize(n + 1, 0.0f);
            v.resize(m + 1, 0.0f);
            p.resize(m + 1, 0);
            way.resize(m + 1, 0);
            minv.resize(m + 1, INF);
            used.resize(m + 1, false);
        }
    };
    
    void solve_rectangular(const std::vector<std::vector<float>>& cost,
                          int n, int m,
                          std::vector<int>& assignment);
    
    void augment(const std::vector<std::vector<float>>& cost,
                 int n, int m, int row,
                 WorkArrays& work);
    
    void optimize_step(const std::vector<std::vector<float>>& cost,
                      int m, int j0, int j1,
                      WorkArrays& work,
                      float& delta);
};

} // namespace tracking
} // namespace tacs

#endif // TACS_TRACKING_HUNGARIAN_ALGORITHM_H