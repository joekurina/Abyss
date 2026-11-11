#pragma once

#include <algorithm>
#include <cmath>

namespace strata::prefill {

// Learn from ready-to-result host service. Mixed helper-attempt/CPU calls retain the prior costs.
inline bool learn_ready_lane_cost(double& cpu_row_ms, double& igpu_row_ms, int rows, int experts,
                                  double service_ms, bool on_igpu, bool igpu_attempted,
                                  double cpu_expert_ms, double igpu_expert_ms) {
    if (rows < 64 || experts <= 0 || !std::isfinite(service_ms) || service_ms <= 0.0 ||
        (igpu_attempted && !on_igpu)) return false;
    if (on_igpu)
        igpu_row_ms = 0.5 * igpu_row_ms +
                      0.5 * std::max(0.002, (service_ms - (double) experts * igpu_expert_ms) / (double) rows);
    else
        cpu_row_ms = 0.5 * cpu_row_ms +
                     0.5 * std::max(0.005, (service_ms - 2.0 * (double) experts * cpu_expert_ms) / (double) rows);
    return true;
}

}  // namespace strata::prefill
