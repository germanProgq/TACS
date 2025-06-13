/*
 * Traffic-Aware Control System (TACS)
 * Hungarian Algorithm implementation for multi-object tracking
 */

#include "tracking/hungarian_algorithm.h"
#include <algorithm>
#include <cassert>

namespace tacs {
namespace tracking {

std::vector<int> HungarianAlgorithm::solve(const std::vector<std::vector<float>>& cost_matrix) {
    if (cost_matrix.empty() || cost_matrix[0].empty()) {
        return std::vector<int>();
    }
    
    int n = static_cast<int>(cost_matrix.size());
    int m = static_cast<int>(cost_matrix[0].size());
    
    std::vector<int> assignment(n, -1);
    
    if (n == 0 || m == 0) {
        return assignment;
    }
    
    solve_rectangular(cost_matrix, n, m, assignment);
    
    return assignment;
}

void HungarianAlgorithm::solve_rectangular(const std::vector<std::vector<float>>& cost,
                                          int n, int m,
                                          std::vector<int>& assignment) {
    // Handle case where there are more rows than columns
    int max_assignments = std::min(n, m);
    
    WorkArrays work;
    work.resize(n, m);
    
    // Only augment for the minimum of n and m
    for (int i = 0; i < max_assignments; ++i) {
        augment(cost, n, m, i, work);
    }
    
    // Extract assignments
    for (int j = 1; j <= m; ++j) {
        if (work.p[j] != 0 && work.p[j] <= n) {
            assignment[work.p[j] - 1] = j - 1;
        }
    }
}

void HungarianAlgorithm::augment(const std::vector<std::vector<float>>& cost,
                                 int n, int m, int row,
                                 WorkArrays& work) {
    std::fill(work.way.begin(), work.way.end(), 0);
    std::fill(work.minv.begin(), work.minv.end(), INF);
    std::fill(work.used.begin(), work.used.end(), false);
    
    work.p[0] = row + 1;
    int j0 = 0;
    
    do {
        work.used[j0] = true;
        int i0 = work.p[j0];
        float delta = INF;
        int j1 = 0;
        
        for (int j = 1; j <= m; ++j) {
            if (!work.used[j]) {
                float cur = (i0 > 0 ? cost[i0 - 1][j - 1] : 0.0f) - work.u[i0] - work.v[j];
                if (cur < work.minv[j]) {
                    work.minv[j] = cur;
                    work.way[j] = j0;
                }
                if (work.minv[j] < delta) {
                    delta = work.minv[j];
                    j1 = j;
                }
            }
        }
        
        // Check if no improvement possible
        if (delta >= INF) {
            break;
        }
        
        for (int j = 0; j <= m; ++j) {
            if (work.used[j]) {
                work.u[work.p[j]] += delta;
                work.v[j] -= delta;
            } else {
                work.minv[j] -= delta;
            }
        }
        
        j0 = j1;
    } while (work.p[j0] != 0 && j0 != 0);
    
    do {
        int j1 = work.way[j0];
        work.p[j0] = work.p[j1];
        j0 = j1;
    } while (j0 != 0);
}

float HungarianAlgorithm::computeTotalCost(const std::vector<std::vector<float>>& cost_matrix,
                                          const std::vector<int>& assignment) const {
    float total_cost = 0.0f;
    
    for (size_t i = 0; i < assignment.size(); ++i) {
        if (assignment[i] >= 0 && assignment[i] < static_cast<int>(cost_matrix[i].size())) {
            total_cost += cost_matrix[i][assignment[i]];
        }
    }
    
    return total_cost;
}

} // namespace tracking
} // namespace tacs