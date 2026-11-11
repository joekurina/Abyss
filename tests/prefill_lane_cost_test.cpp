// Offline learner tests for clean CPU/HIP service and declined helper calls.
#include <cmath>
#include <cstdio>
#include <limits>
#include <initializer_list>

#if __has_include("strata/prefill/lane_cost.hpp")
#include "strata/prefill/lane_cost.hpp"
int main() {
    using strata::prefill::learn_ready_lane_cost;
    int failures = 0;
    auto check = [&](bool ok, const char* name) {
        if (!ok) { std::fprintf(stderr, "FAIL %s\n", name); ++failures; }
    };
    double cpu = 0.05, hip = 0.009;
    check(learn_ready_lane_cost(cpu, hip, 100, 2, 10.0, true, true, 0.1, 0.045), "successful HIP update");
    check(std::abs(hip - (0.5 * 0.009 + 0.5 * (10.0 - 2 * 0.045) / 100)) < 1e-14, "HIP recurrence");
    check(cpu == 0.05, "HIP leaves CPU estimate unchanged");
    check(learn_ready_lane_cost(cpu, hip, 100, 2, 20.0, false, false, 0.1, 0.045), "clean CPU update");
    check(std::abs(cpu - (0.5 * 0.05 + 0.5 * (20.0 - 2.0 * 2 * 0.1) / 100)) < 1e-14, "CPU recurrence");
    for (int rc : {1, -1}) {
        const double before_cpu = cpu, before_hip = hip;
        check(!learn_ready_lane_cost(cpu, hip, 100, 2, 20.0, rc == 0, true, 0.1, 0.045), "helper fallback skips learning");
        check(cpu == before_cpu && hip == before_hip, "fallback preserves both estimates");
    }
    for (double ms : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        const double before_cpu = cpu, before_hip = hip;
        check(!learn_ready_lane_cost(cpu, hip, 100, 2, ms, true, true, 0.1, 0.045), "invalid interval skipped");
        check(cpu == before_cpu && hip == before_hip, "invalid interval preserves estimates");
    }
    for (int rows : {0, 1, 63})
        check(!learn_ready_lane_cost(cpu, hip, rows, 2, 10.0, true, true, 0.1, 0.045), "small sample skipped");
    cpu = hip = 0.0;
    check(learn_ready_lane_cost(cpu, hip, 64, 2, 0.01, true, true, 0.1, 0.045), "HIP floor update");
    check(hip == 0.001, "original HIP floor and EMA");
    check(learn_ready_lane_cost(cpu, hip, 64, 2, 0.01, false, false, 0.1, 0.045), "CPU floor update");
    check(cpu == 0.0025, "original CPU floor and EMA");
    std::printf("lane cost checks: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
#else
int main() {
    std::fprintf(stderr, "FAIL ready-to-result lane cost learner is absent\n");
    return 1;
}
#endif
