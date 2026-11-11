// Standalone CUDA differential checks for the opt-in SM80 32-head indexer.
// Build and qualification commands are in optimization/sm80-gfx1151-v1/idx32/NOTES.md.
#if !defined(STRATA_IDX32_HOST_FIXTURES)
#include "strata/kernels/dsv4_fast.hpp"
#include <cuda_fp16.h>
#endif
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(STRATA_IDX32_HOST_FIXTURES)
namespace ds = strata::kernels::dsv4;
#endif

namespace {
constexpr int D = 128;
#if !defined(STRATA_IDX32_HOST_FIXTURES)
constexpr int GUARD = 32;
constexpr float SENTINEL = -1234567.0f;
constexpr double ATOL = 5e-5, RTOL = 2e-5;
int cases = 0, selection_rows_changed = 0, selection_members_changed = 0;
bool strict_selection = false;
#endif

void require(bool ok, const std::string& text) {
    if (!ok) throw std::runtime_error(text);
}
#if !defined(STRATA_IDX32_HOST_FIXTURES)
void cu(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}

template <class T> struct Device {
    T* p = nullptr;
    size_t n;
    explicit Device(size_t count) : n(count) { cu(cudaMalloc(&p, count * sizeof(T)), "cudaMalloc"); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void put(const std::vector<T>& x) {
        require(x.size() == n, "fixture allocation count");
        cu(cudaMemcpy(p, x.data(), n * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<T> get() const {
        std::vector<T> x(n);
        cu(cudaMemcpy(x.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return x;
    }
};
#endif

unsigned mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
float random_value(unsigned i) { return (float(int(mix(i) & 65535u)) - 32768.0f) / 65536.0f; }
uint16_t bits(float x) {
#if defined(STRATA_IDX32_HOST_FIXTURES)
    static_assert(sizeof(_Float16) == sizeof(uint16_t), "IEEE binary16 fixture storage");
    const _Float16 h = (_Float16) x;
    uint16_t u;
    std::memcpy(&u, &h, sizeof(u));
    return u;
#else
    return __half_as_ushort(__float2half_rn(x));
#endif
}

struct Fixture {
    int n, T, H;
    std::string kind;
    std::vector<float> q, w;
    std::vector<uint16_t> k;
    Fixture(int rows, int tokens, int heads, const std::string& mode)
        : n(rows), T(tokens), H(heads), kind(mode), q(size_t(tokens) * heads * D),
          w(size_t(tokens) * heads), k(size_t(rows) * D) {
        require(q.size() == size_t(T) * H * D && w.size() == size_t(T) * H && k.size() == size_t(n) * D,
                "fixture element counts");
        for (int j = 0; j < n; ++j) for (int d = 0; d < D; ++d) {
            float x = random_value(unsigned(j) * D + d + 113);
            if (kind == "positive") x = std::fabs(x) + 0.125f;
            if (kind == "negative") x = -std::fabs(x) - 0.125f;
            if (kind == "zero") x = 0.0f;
            if (kind == "ties") x = 0.25f;
            if (kind == "distinct") x = d == 0 ? float(j % 1024 + 1) : d == 1 ? float(j / 1024) : 0.0f;
            if (kind == "residual") x = d < 2 ? float(1 << (j % 4)) : 0.0f;
            if (kind == "near") {
                x = 0.25f;
                if (d == j % D) x += (j & 1) ? 0.000244140625f : -0.0001220703125f;
            }
            k[size_t(j) * D + d] = bits(x);
        }
        for (int t = 0; t < T; ++t) for (int h = 0; h < H; ++h) {
            float weight = random_value(unsigned(t * H + h) + 7001);
            if (kind == "positive") weight = std::fabs(weight) + 0.125f;
            if (kind == "negative") weight = -std::fabs(weight) - 0.125f;
            if (kind == "zero") weight = 0.0f;
            if (kind == "ties") weight = 1.0f / float(H);
            if (kind == "distinct" || kind == "residual") weight = h == 0 ? 1.0f : 0.0f;
            if (kind == "near") weight = 1.0f / float(H);
            w[size_t(t) * H + h] = weight;
            for (int d = 0; d < D; ++d) {
                float x = random_value(unsigned((t * H + h) * D + d) + 19001);
                if (kind == "positive") x = std::fabs(x) + 0.125f;
                if (kind == "negative") x = -std::fabs(x) - 0.125f;
                if (kind == "zero") x = 0.0f;
                if (kind == "ties") x = 1.0f;
                if (kind == "distinct") x = h == 0 ? (d == 0 ? 1.0f : d == 1 ? 1024.0f : 0.0f) : 0.0f;
                if (kind == "residual") x = h == 0 ? (d == 0 ? 1.0001f : d == 1 ? -1.0f : 0.0f) : 0.0f;
                if (kind == "near") x = 1.0001f + random_value(unsigned(h * D + d) + 73) * 1e-5f;
                q[(size_t(t) * H + h) * D + d] = x;
            }
        }
    }
};

#if !defined(STRATA_IDX32_HOST_FIXTURES)
bool enabled() {
    const char* v = std::getenv("STRATA_DS41_IDX32");
    const char* old = std::getenv("STRATA_DS4_IDX");
    return v && std::strcmp(v, "1") == 0 && !(old && std::atoi(old) == 1);
}
std::vector<int> order(const float* s, int n, int top) {
    std::vector<int> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + top, ids.end(), [&](int a, int b) {
        return s[a] > s[b] || (s[a] == s[b] && a < b);
    });
    ids.resize(top);
    return ids;
}
void guards(const std::vector<float>& v) {
    for (int i = 0; i < GUARD; ++i)
        require(v[i] == SENTINEL && v[v.size() - GUARD + i] == SENTINEL, "score guard modified");
}
void compare(const std::string& label, const std::vector<float>& ref, const std::vector<float>& got,
             const std::vector<int>& visible, int ld, bool invisible_untouched, const std::string& kind) {
    double max_abs = 0, max_rel = 0, sumsq = 0;
    size_t count = 0, bad = 0;
    guards(ref); guards(got);
    for (int t = 0; t < int(visible.size()); ++t) {
        for (int j = 0; j < visible[t]; ++j) {
            const size_t at = GUARD + size_t(t) * ld + j;
            const double a = ref[at], b = got[at];
            require(std::isfinite(a) && std::isfinite(b), label + " nonfinite visible score");
            const double diff = std::fabs(a - b);
            max_abs = std::max(max_abs, diff);
            max_rel = std::max(max_rel, diff / std::max(std::fabs(a), 1e-20));
            sumsq += diff * diff;
            ++count;
            bad += diff > ATOL + RTOL * std::fabs(a);
            if (kind == "distinct") require(a == j + 1 && b == j + 1, "distinct fixture exact score");
            if (kind == "zero") require(a == 0 && b == 0, "zero fixture exact score");
            if (kind == "residual") require(a > 9e-5 && b > 9e-5, "low-half contribution lost");
        }
        if (invisible_untouched) for (int j = visible[t]; j < ld; ++j)
            require(got[GUARD + size_t(t) * ld + j] == SENTINEL, label + " wrote invisible row");
    }
    std::printf("scores %s count=%zu max_abs=%.9g max_rel=%.9g rms=%.9g violations=%zu\n", label.c_str(), count,
                max_abs, max_rel, count ? std::sqrt(sumsq / count) : 0, bad);
    require(bad == 0, label + " score tolerance");
}

void selection(const std::string& label, const std::string& kind, const Device<float>& dr,
               const Device<float>& dc, const std::vector<float>& ref, const std::vector<float>& got,
               const std::vector<int>& visible, int ld, int ratio, int p0, int top, bool batch) {
    const int T = int(visible.size()), stride = top + 3;
    Device<int> sr(size_t(T) * stride + 2 * GUARD), sc(sr.n);
    std::vector<int> fill(sr.n, -1234567);
    sr.put(fill); sc.put(fill);
    if (batch) {
        ds::indexer_batch_select(dr.p + GUARD, ld, ratio, p0, T, top, sr.p + GUARD, stride, nullptr);
        ds::indexer_batch_select(dc.p + GUARD, ld, ratio, p0, T, top, sc.p + GUARD, stride, nullptr);
    } else {
        ds::indexer_select(dr.p + GUARD, visible[0], top, sr.p + GUARD, nullptr);
        ds::indexer_select(dc.p + GUARD, visible[0], top, sc.p + GUARD, nullptr);
    }
    cu(cudaDeviceSynchronize(), "selection synchronize");
    auto a = sr.get(), b = sc.get();
    int changed = 0, members = 0;
    const bool exact = kind == "zero" || kind == "ties" || kind == "distinct" || kind == "residual";
    for (int t = 0; t < T; ++t) {
        const int kept = std::min(top, visible[t]);
        const float* r = ref.data() + GUARD + size_t(t) * ld;
        const float* c = got.data() + GUARD + size_t(t) * ld;
        const auto er = order(r, visible[t], kept), ec = order(c, visible[t], kept);
        for (int j = 0; j < kept; ++j) {
            const size_t at = GUARD + size_t(t) * stride + j;
            require(a[at] == er[j] && b[at] == ec[j], label + " selector differs from CPU score/index order");
            fill[at] = a[at];
            if (a[at] != b[at]) ++changed;
            if (kind == "zero" || kind == "ties") require(a[at] == j && b[at] == j, "tie must choose low row ID");
        }
        auto ar = er, ac = ec;
        std::sort(ar.begin(), ar.end()); std::sort(ac.begin(), ac.end());
        std::vector<int> only;
        std::set_difference(ar.begin(), ar.end(), ac.begin(), ac.end(), std::back_inserter(only));
        members += int(only.size());
        if (er != ec) {
            ++selection_rows_changed;
            double gap = 0;
            if (kept && visible[t] > kept) {
                auto edge = order(r, visible[t], kept + 1);
                gap = double(r[edge[kept - 1]]) - r[edge[kept]];
            }
            std::printf("selection-difference %s token=%d top=%d membership_removed=%zu reference_cut_gap=%.9g\n",
                        label.c_str(), t, top, only.size(), gap);
            for (int j = 0; j < kept; ++j) if (er[j] != ec[j]) {
                std::printf("first-order-difference rank=%d ref_id=%d candidate_id=%d ref_scores=(%.9g,%.9g) "
                            "candidate_scores=(%.9g,%.9g)\n", j, er[j], ec[j], r[er[j]], r[ec[j]], c[er[j]], c[ec[j]]);
                break;
            }
        }
    }
    for (size_t i = 0; i < fill.size(); ++i) if (fill[i] == -1234567)
        require(a[i] == -1234567 && b[i] == -1234567, "selection padding/guard modified");
    selection_members_changed += members;
    std::printf("selection %s top=%d changed_positions=%d membership_removed=%d required_exact=%d\n",
                label.c_str(), top, changed, members, int(exact || strict_selection));
    require(!(exact || strict_selection) || changed == 0, label + " exact selection mismatch");
}

void run_case(int n, int T, int H, const std::string& kind, bool batch, int ratio = 4, int start = -1,
              int top = 512) {
    const int p0 = start < 0 ? n * ratio - T : start;
    const int maximum = batch ? (p0 + T) / ratio : n;
    require(maximum <= n && p0 >= 0, "invalid fixture causal shape");
    const int ld = n + 7;
    std::vector<int> visible(T, n);
    if (batch) for (int t = 0; t < T; ++t) visible[t] = (p0 + t + 1) / ratio;
    Fixture f(n, T, H, kind);
    Device<float> q(f.q.size()), w(f.w.size());
    Device<uint16_t> k(f.k.size());
    q.put(f.q); w.put(f.w); k.put(f.k);
    Device<float> ref(size_t(T) * ld + 2 * GUARD), cand(ref.n), automatic(ref.n);
    std::vector<float> fill(ref.n, SENTINEL);
    ref.put(fill); cand.put(fill); automatic.put(fill);
    if (batch) {
        ds::indexer_batch_scores_reference(q.p, k.p, w.p, H, ratio, p0, T, ref.p + GUARD, ld, nullptr);
        if (H == 32) require(ds::indexer_batch_scores_idx32(q.p, k.p, w.p, H, D, ratio, p0, T,
                                                          cand.p + GUARD, ld, nullptr) == (maximum > 0),
                             "batch explicit dispatch");
        ds::indexer_batch_scores(q.p, k.p, w.p, H, ratio, p0, T, automatic.p + GUARD, ld, nullptr);
    } else {
        ds::indexer_scores_reference(q.p, k.p, w.p, H, n, ref.p + GUARD, nullptr);
        if (H == 32) require(ds::indexer_scores_idx32(q.p, k.p, w.p, H, D, n, cand.p + GUARD, nullptr),
                             "decode explicit refused");
        ds::indexer_scores(q.p, k.p, w.p, H, n, automatic.p + GUARD, nullptr);
    }
    cu(cudaDeviceSynchronize(), "scores synchronize");
    auto r = ref.get(), c = cand.get(), a = automatic.get();
    const std::string label = std::string(batch ? "batch" : "decode") + "/" + kind + "/n=" + std::to_string(n) +
                             "/T=" + std::to_string(T) + "/H=" + std::to_string(H);
    guards(a);
    if (H == 32) {
        compare(label, r, c, visible, ld, true, kind);
        const auto& expected = enabled() ? c : r;
        require(std::memcmp(a.data(), expected.data(), a.size() * sizeof(float)) == 0, label + " automatic dispatch");
        selection(label, kind, ref, cand, r, c, visible, ld, ratio, p0, top, batch);
    } else {
        compare(label + "/fallback", r, a, visible, ld, false, kind);
        if (batch || H != 64 || (std::getenv("STRATA_DS4_IDX") && std::atoi(std::getenv("STRATA_DS4_IDX")) == 1))
            require(std::memcmp(r.data(), a.data(), r.size() * sizeof(float)) == 0, "plain fallback changed");
        require(!ds::indexer_idx32_supported(H, D), "unsupported heads admitted");
        selection(label, kind, ref, automatic, r, a, visible, ld, ratio, p0, top, batch);
    }
    if (H == 32 && maximum > 0 && n <= 513 && T <= 33 && (kind == "ties" || kind == "mixed")) {
        Device<float> combined(ref.n);
        Device<int> selected(size_t(T) * top + 2 * GUARD);
        combined.put(fill);
        std::vector<int> si(selected.n, -1234567);
        selected.put(si);
        if (batch) ds::indexer_batch(q.p, k.p, w.p, H, ratio, p0, T, top, combined.p + GUARD, ld,
                                     selected.p + GUARD, nullptr);
        else ds::indexer(q.p, k.p, w.p, H, D, n, top, combined.p + GUARD, selected.p + GUARD, nullptr);
        cu(cudaDeviceSynchronize(), "combined indexer synchronize");
        const auto both = combined.get();
        require(std::memcmp(both.data(), a.data(), a.size() * sizeof(float)) == 0, "combined scorer dispatch");
        const auto ids = selected.get();
        for (int t = 0; t < T; ++t) {
            const auto expected = order(a.data() + GUARD + size_t(t) * ld, visible[t], std::min(top, visible[t]));
            for (size_t j = 0; j < expected.size(); ++j) si[GUARD + size_t(t) * top + j] = expected[j];
        }
        require(ids == si, "combined selector IDs/guard");
    }
    require(q.get() == f.q, "CUDA scorer modified Q");
    ++cases;
}

void refusal_tests() {
    Device<float> out(64);
    std::vector<float> fill(64, SENTINEL);
    out.put(fill);
    for (int h : {0, 16, 31, 33, 64}) {
        require(!ds::indexer_idx32_supported(h, 128), "unsupported heads accepted");
        require(!ds::indexer_scores_idx32(nullptr, nullptr, nullptr, h, 128, 17, out.p, nullptr), "decode heads");
        require(!ds::indexer_batch_scores_idx32(nullptr, nullptr, nullptr, h, 128, 4, 64, 1, out.p, 17, nullptr),
                "batch heads");
    }
    for (int d : {0, 64, 127, 129, 256}) {
        require(!ds::indexer_idx32_supported(32, d), "unsupported dimension accepted");
        require(!ds::indexer_scores_idx32(nullptr, nullptr, nullptr, 32, d, 17, out.p, nullptr), "decode dim");
        require(!ds::indexer_batch_scores_idx32(nullptr, nullptr, nullptr, 32, d, 4, 64, 1, out.p, 17, nullptr),
                "batch dim");
    }
    require(!ds::indexer_scores_idx32(nullptr, nullptr, nullptr, 32, 128, 0, out.p, nullptr), "empty decode");
    require(!ds::indexer_batch_scores_idx32(nullptr, nullptr, nullptr, 32, 128, 0, 64, 1, out.p, 17, nullptr),
            "zero ratio");
    require(!ds::indexer_batch_scores_idx32(nullptr, nullptr, nullptr, 32, 128, 4, 64, 1, out.p, 15, nullptr),
            "short score stride");
    require(!ds::indexer_batch_scores_idx32(nullptr, nullptr, nullptr, 32, 128, 4, 0, 0, out.p, 17, nullptr),
            "empty batch");
    require(!ds::indexer_batch_scores_idx32(nullptr, nullptr, nullptr, 32, 128, 4, 0, 1, out.p, 17, nullptr),
            "no visible rows");
    cu(cudaDeviceSynchronize(), "refusal synchronize");
    require(out.get() == fill, "refusal modified output");
}
#endif
} // namespace

#if defined(STRATA_IDX32_HOST_FIXTURES)
// This mode exercises fixture generation and a double-precision score oracle on the host.
int main() {
    try {
        int checked = 0;
        for (int n : {1, 17, 513}) for (int T : {1, 3, 33})
            for (const char* kind : {"positive", "negative", "mixed", "zero", "ties", "distinct", "residual", "near"}) {
                Fixture f(n, T, 32, kind);
                for (int t : {0, T - 1}) for (int j : {0, n - 1}) {
                    double score = 0;
                    for (int h = 0; h < f.H; ++h) {
                        double dot = 0;
                        for (int d = 0; d < D; ++d) {
                            _Float16 k;
                            const uint16_t u = f.k[size_t(j) * D + d];
                            std::memcpy(&k, &u, sizeof(k));
                            dot += double(f.q[(size_t(t) * f.H + h) * D + d]) * double(k);
                        }
                        score += std::max(0.0, dot) * f.w[size_t(t) * f.H + h];
                    }
                    require(std::isfinite(score), "fixture nonfinite score");
                    if (f.kind == "positive") require(score > 0, "positive fixture sign");
                    if (f.kind == "negative") require(score < 0, "negative fixture sign");
                    if (f.kind == "zero") require(score == 0, "zero fixture score");
                    if (f.kind == "ties") require(score == 32, "tie fixture score");
                    if (f.kind == "distinct") require(score == j + 1, "distinct fixture score");
                    if (f.kind == "residual") require(score > 9e-5, "residual fixture score");
                    ++checked;
                }
            }
        // The two-coordinate encoding stays exact throughout the optional one-million-row fixture.
        for (int j : {16383, 16384, 16385, 32767, 65535, 131071, 1048575}) {
            const uint16_t a = bits(float(j % 1024 + 1)), b = bits(float(j / 1024));
            _Float16 lo, hi;
            std::memcpy(&lo, &a, sizeof(lo)); std::memcpy(&hi, &b, sizeof(hi));
            require(float(lo) + 1024.0f * float(hi) == float(j + 1), "large distinct fixture encoding");
        }
        std::printf("PASS host fixture checks=%d. CUDA kernels, dispatch and selection were not executed.\n", checked);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL host fixtures: %s\n", e.what());
        return 1;
    }
}
#else
int main(int argc, char** argv) {
    try {
        bool quick = false, million = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--quick") quick = true;
            else if (arg == "--million") million = true;
            else if (arg == "--strict-selection") strict_selection = true;
            else throw std::runtime_error("usage: dsv41_idx32_test [--quick] [--million] [--strict-selection]");
        }
        int dev = 0;
        cu(cudaGetDevice(&dev), "cudaGetDevice");
        cudaDeviceProp prop{};
        cu(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
        std::printf("device=%s sm=%d%d automatic_idx32=%d atol=%.9g rtol=%.9g\n", prop.name, prop.major, prop.minor,
                    int(enabled()), ATOL, RTOL);
        refusal_tests();
        if (!ds::indexer_idx32_supported(32, 128)) {
            Fixture f(513, 3, 32, "mixed");
            Device<float> q(f.q.size()), w(f.w.size()), a(3 * 513), b(a.n), c(a.n);
            Device<uint16_t> k(f.k.size());
            q.put(f.q); w.put(f.w); k.put(f.k);
            std::vector<float> fill(a.n, SENTINEL);
            a.put(fill); b.put(fill); c.put(fill);
            require(!ds::indexer_scores_idx32(q.p, k.p, w.p, 32, 128, 513, c.p, nullptr), "unsupported decode");
            require(!ds::indexer_batch_scores_idx32(q.p, k.p, w.p, 32, 128, 4, 2049, 3, c.p, 513, nullptr),
                    "unsupported batch");
            ds::indexer_scores_reference(q.p, k.p, w.p, 32, 513, a.p, nullptr);
            ds::indexer_scores(q.p, k.p, w.p, 32, 513, b.p, nullptr);
            cu(cudaDeviceSynchronize(), "unsupported decode fallback");
            require(a.get() == b.get() && c.get() == fill, "unsupported decode fallback differs");
            a.put(fill); b.put(fill);
            ds::indexer_batch_scores_reference(q.p, k.p, w.p, 32, 4, 2049, 3, a.p, 513, nullptr);
            ds::indexer_batch_scores(q.p, k.p, w.p, 32, 4, 2049, 3, b.p, 513, nullptr);
            cu(cudaDeviceSynchronize(), "unsupported batch fallback");
            require(a.get() == b.get() && c.get() == fill, "unsupported batch fallback differs");
            std::printf("SKIP specialized qualification: unsupported device/code image, original fallback passed\n");
            return 77;
        }
        const std::vector<int> sizes = quick ? std::vector<int>{17, 513} :
            std::vector<int>{1, 7, 15, 16, 17, 127, 128, 129, 255, 256, 257, 511, 512, 513,
                             16383, 16384, 16385, 32768, 65536, 131072};
        for (int n : sizes) for (const char* kind : {"positive", "negative", "mixed", "zero", "ties", "distinct",
                                                   "residual", "near"}) {
            run_case(n, 1, 32, kind, false);
            run_case(n, 3, 32, kind, true);
        }
        for (int T : {1, 2, 31, 32, 33, 65}) {
            run_case(513, T, 32, "mixed", true);
            run_case((T + 3) / 4, T, 32, "ties", true, 4, 0);
        }
        for (int ratio : {1, 2, 7}) run_case(513, 33, 32, "mixed", true, ratio);
        for (int H : {16, 31, 64}) {
            run_case(513, 1, H, "mixed", false);
            run_case(513, 3, H, "mixed", true);
        }
        for (int top : {1, 511, 512, 513, 2048}) for (const char* kind : {"ties", "distinct"}) {
            run_case(16385, 1, 32, kind, false, 4, -1, top);
            run_case(16385, 3, 32, kind, true, 4, -1, top);
        }
        if (million) {
            size_t free_bytes = 0, total_bytes = 0;
            cu(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
            const size_t budget = size_t(512) * 1024 * 1024;
            if (free_bytes < budget) {
                std::printf("SKIP million rows: free=%zu required_headroom=%zu\n", free_bytes, budget);
                return 77;
            }
            for (const char* kind : {"mixed", "ties", "distinct", "residual"}) {
                run_case(1048576, 1, 32, kind, false);
                run_case(1048576, 2, 32, kind, true);
            }
        }
        std::printf("PASS score/fixture contracts cases=%d selection_rows_changed=%d membership_removed=%d strict_selection=%d\n", cases,
                    selection_rows_changed, selection_members_changed, int(strict_selection));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
#endif
