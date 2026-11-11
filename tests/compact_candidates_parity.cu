// Standalone CUDA parity. See candidates/compact-candidates/NOTES.md for build and sanitizer commands.
#include "strata/kernels/dsv41_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace ds = strata::kernels::dsv4;
namespace d41 = strata::kernels::dsv41;

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)
#define CUDA_OK(x) do { const cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

// Every device allocation has canaries on both sides. The row-layout checks also cover interior padding.
template<class T> struct Buffer {
    static constexpr size_t guard = 16;
    T *base = nullptr, *p = nullptr;
    size_t n;
    explicit Buffer(size_t count) : n(count) {
        REQUIRE(n > 0);
        CUDA_OK(cudaMalloc((void**) &base, (n + 2 * guard) * sizeof(T)));
        p = base + guard;
        poison();
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { cudaFree(base); }
    void poison() { CUDA_OK(cudaMemset(base, 0xa5, (n + 2 * guard) * sizeof(T))); }
    void put(const std::vector<T>& h) {
        REQUIRE(h.size() == n);
        CUDA_OK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> h(n);
        CUDA_OK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
    void check() const {
        std::vector<unsigned char> h(guard * sizeof(T));
        CUDA_OK(cudaMemcpy(h.data(), base, h.size(), cudaMemcpyDeviceToHost));
        for (auto v : h) REQUIRE(v == 0xa5);
        CUDA_OK(cudaMemcpy(h.data(), p + n, h.size(), cudaMemcpyDeviceToHost));
        for (auto v : h) REQUIRE(v == 0xa5);
    }
};

static bool same_float(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }
static void same_floats(const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    REQUIRE(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}
static std::vector<int> top_ids(const std::vector<float>& score, int top) {
    std::vector<int> ids(score.size());
    std::iota(ids.begin(), ids.end(), 0);
    top = std::min(top, (int) ids.size());
    std::partial_sort(ids.begin(), ids.begin() + top, ids.end(), [&](int a, int b) {
        return score[a] > score[b] || (score[a] == score[b] && a < b);
    });
    ids.resize(top);
    return ids;
}
static std::vector<int> block_reference(const std::vector<float>& sc) {
    std::vector<float> bs((sc.size() + 7) / 8, -INFINITY);
    for (size_t j = 0; j < sc.size(); ++j) bs[j / 8] = std::fmax(bs[j / 8], sc[j]);
    bs.back() = INFINITY;
    return top_ids(bs, 2048);
}
static std::vector<int> expand_reference(std::vector<int> blocks, int n) {
    std::sort(blocks.begin(), blocks.end());
    std::vector<int> rows;
    for (int b : blocks)
        for (int j = 8 * b; j < std::min(8 * (b + 1), n); ++j) rows.push_back(j);
    return rows;
}

__global__ void keys_kernel(uint16_t* K, int n, int kind) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * 128) return;
    const unsigned j = (unsigned) (i / 128), d = (unsigned) (i % 128);
    // kind 1 repeats every row, kind 2 yields all-zero scores through Q.
    const unsigned x = (kind == 1 ? 0u : j) * 1664525u + d * 1013904223u + 17u;
    const float v = (float) ((int) ((x >> 13) % 257u) - 128) / 97.0f;
    K[i] = __half_as_ushort(__float2half(v));
}

static void score_case(int n, int kind, cudaStream_t s) {
    constexpr int cap = 16384, heads = 32;
    const int B = (n + 7) / 8, top = std::min(n, 512), nb = std::min(B, 2048);
    Buffer<float> source(n), bscore(B), dense(n), compact(cap), fallback(n), iq(heads * 128), w(heads);
    Buffer<int> bsel(2048), bsel_bits(2048), rows(cap), nr(1), selected(top), expected(top);
    Buffer<unsigned char> keep(B);
    Buffer<uint32_t> bits((B + 31) / 32);
    Buffer<uint16_t> K((size_t) n * 128);
    std::vector<float> src(n);
    for (int j = 0; j < n; ++j)
        src[j] = kind == 1 ? 0.0f : kind == 3 ? -(float) ((j * 31u) % 1021u) : (float) ((j * 97u) % 997u);
    source.put(src);
    d41::candidates(source.p, n, 8, 2048, bscore.p, bsel.p, keep.p, s);
    d41::candidates_bitset(source.p, n, 8, 2048, bscore.p, bsel_bits.p, bits.p, s);
    d41::candidate_rows(bsel.p, n, 8, 2048, rows.p, nr.p, s);
    CUDA_OK(cudaStreamSynchronize(s));
    auto bs = bsel.get();
    bs.resize(nb);
    REQUIRE(bs == block_reference(src));
    auto bb = bsel_bits.get();
    bb.resize(nb);
    REQUIRE(bb == bs);
    const auto want_rows = expand_reference(bs, n);
    REQUIRE(nr.get()[0] == (int) want_rows.size());
    auto cr = rows.get();
    REQUIRE(std::equal(want_rows.begin(), want_rows.end(), cr.begin()));
    const auto km = keep.get();
    const auto bm = bits.get();
    for (int b = 0; b < B; ++b) REQUIRE((bool) km[b] == (bool) ((bm[b / 32] >> (b % 32)) & 1));
    if (B % 32) REQUIRE((bm.back() >> (B % 32)) == 0);
    REQUIRE(km.back() == 1);
    bsel.check();
    bscore.check();
    // Candidate scratch can be reused by other work while retained rows survive through layer 36.
    bsel.poison();
    bscore.poison();
    keys_kernel<<<(unsigned) (((size_t) n * 128 + 255) / 256), 256, 0, s>>>(K.p, n, kind);
    CUDA_OK(cudaGetLastError());
    for (int layer : {24, 28, 32, 36}) {
        std::vector<float> q(heads * 128), weights(heads);
        for (size_t i = 0; i < q.size(); ++i)
            q[i] = kind == 2 ? 0.0f : (float) ((int) ((i * 73 + layer * 11) % 521) - 260) / 123.0f;
        for (int h = 0; h < heads; ++h)
            weights[h] = kind == 3 ? -0.01f * (h + 1) : (float) ((h * 17 + layer) % 31 - 15) / 59.0f;
        iq.put(q);
        w.put(weights);
        ds::indexer_scores(iq.p, K.p, w.p, heads, n, dense.p, s);
        d41::indexer_candidate_scores(iq.p, K.p, w.p, heads, rows.p, nr.p, cap, compact.p, s);
        CUDA_OK(cudaStreamSynchronize(s));
        const auto full = dense.get(), small = compact.get();
        for (size_t i = 0; i < want_rows.size(); ++i) REQUIRE(same_float(small[i], full[want_rows[i]]));
        for (size_t i = want_rows.size(); i < small.size(); ++i) {
            uint32_t u;
            std::memcpy(&u, &small[i], 4);
            REQUIRE(u == 0xa5a5a5a5u);
        }
        d41::candidate_mask(dense.p, n, 8, keep.p, s);
        ds::indexer_select(dense.p, n, top, expected.p, s);
        d41::indexer_candidate_select(compact.p, rows.p, nr.p, n, top, fallback.p, selected.p, s);
        CUDA_OK(cudaStreamSynchronize(s));
        REQUIRE(selected.get() == expected.get());
        REQUIRE(selected.get() == top_ids(dense.get(), top));
        // Readers reuse the same absolute IDs. Selection leaves the retained domain untouched.
        REQUIRE(rows.get() == cr);
        CUDA_OK(cudaMemcpyAsync(fallback.p, full.data(), (size_t) n * sizeof(float), cudaMemcpyHostToDevice, s));
        d41::candidate_mask_bitset(fallback.p, n, 8, bits.p, s);
        CUDA_OK(cudaStreamSynchronize(s));
        same_floats(fallback.get(), dense.get());
    }
    source.check(); bscore.check(); dense.check(); compact.check(); fallback.check(); iq.check(); w.check();
    bsel.check(); bsel_bits.check(); rows.check(); nr.check(); selected.check(); expected.check();
    keep.check(); bits.check(); K.check();
}

static void empty_case(cudaStream_t s) {
    Buffer<int> rows(16384), nr(1), selected(512);
    Buffer<float> score(16384);
    const auto before = rows.get(), sel_before = selected.get();
    const auto score_before = score.get();
    d41::candidate_rows(nullptr, 0, 8, 2048, rows.p, nr.p, s);
    d41::indexer_candidate_scores(nullptr, nullptr, nullptr, 32, rows.p, nr.p, 0, score.p, s);
    d41::indexer_candidate_select(score.p, rows.p, nr.p, 0, 512, nullptr, selected.p, s);
    CUDA_OK(cudaStreamSynchronize(s));
    REQUIRE(nr.get()[0] == 0);
    REQUIRE(rows.get() == before);
    REQUIRE(selected.get() == sel_before);
    same_floats(score.get(), score_before);
    rows.check(); nr.check(); selected.check(); score.check();
}

static void exceptional_case(cudaStream_t s) {
    const int n = 32769, cap = 16384;
    Buffer<float> source(n), bscore((n + 7) / 8), compact(cap), masked(n), fallback(n);
    Buffer<int> bsel(2048), rows(cap), nr(1), got(512), want(512);
    Buffer<unsigned char> keep((n + 7) / 8);
    std::vector<float> src(n);
    std::iota(src.begin(), src.end(), 0.0f);
    source.put(src);
    d41::candidates(source.p, n, 8, 2048, bscore.p, bsel.p, keep.p, s);
    d41::candidate_rows(bsel.p, n, 8, 2048, rows.p, nr.p, s);
    CUDA_OK(cudaStreamSynchronize(s));
    REQUIRE(rows.get()[0] > 0);
    for (float value : {-INFINITY, INFINITY}) {
        compact.put(std::vector<float>(cap, value));
        masked.put(std::vector<float>(n, value));
        d41::candidate_mask(masked.p, n, 8, keep.p, s);
        ds::indexer_select(masked.p, n, 512, want.p, s);
        d41::indexer_candidate_select(compact.p, rows.p, nr.p, n, 512, fallback.p, got.p, s);
        CUDA_OK(cudaStreamSynchronize(s));
        const auto ids = got.get();
        REQUIRE(ids == want.get());
        if (value < 0) for (int i = 0; i < 512; ++i) REQUIRE(ids[i] == i);
    }
    // If enough older blocks are +inf too, the original tie rule can omit the newest forced block.
    source.put(std::vector<float>(n, INFINITY));
    d41::candidates(source.p, n, 8, 2048, bscore.p, bsel.p, keep.p, s);
    d41::candidate_rows(bsel.p, n, 8, 2048, rows.p, nr.p, s);
    CUDA_OK(cudaStreamSynchronize(s));
    REQUIRE(nr.get()[0] == cap);
    REQUIRE(rows.get().back() == cap - 1);
    REQUIRE(keep.get().back() == 0);
    source.check(); bscore.check(); compact.check(); masked.check(); fallback.check();
    bsel.check(); rows.check(); nr.check(); got.check(); want.check(); keep.check();
}

// Byte and bitset batches share original block maxima and top-block selection. Different row strides are deliberate.
static void batch_case(int max_ctx, int p0, int T, int ratio, int cand_min, cudaStream_t s) {
    const int nmax = (p0 + T) / ratio, ld = nmax + 13, bld = (nmax + 7) / 8 + 7;
    const size_t rb = d41::candidate_row_bytes(max_ctx, 8, 512, false);
    const size_t ri = d41::candidate_row_bytes(max_ctx, 8, 512, true);
    const int lb = (int) (rb / 4) + 8, li = (int) (ri / 4) + 12;
    Buffer<int> byte_rows((size_t) T * lb), bit_rows((size_t) T * li), serial_rows((size_t) T * li);
    Buffer<float> source((size_t) T * ld), byte_score((size_t) T * ld), bit_score((size_t) T * ld), bscore((size_t) T * bld);
    Buffer<int> bsel((size_t) T * 2048);
    std::vector<float> sc((size_t) T * ld);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < ld; ++j) sc[(size_t) t * ld + j] = t % 3 == 0 ? 0.0f : (float) ((j * 23u + t) % 113u) - 61.0f;
    source.put(sc);
    byte_score.put(sc);
    bit_score.put(sc);
    d41::candidates_batch(source.p, ld, ratio, p0, T, cand_min, 8, 2048, bscore.p, bld, bsel.p,
                          byte_rows.p, lb, 512, s);
    d41::candidates_batch_bitset(source.p, ld, ratio, p0, T, cand_min, 8, 2048, bscore.p, bld, bsel.p,
                                 bit_rows.p, li, 512, s);
    for (int t = 0; t < T; ++t) {
        const int n = (p0 + t + 1) / ratio;
        if (n > cand_min)
            d41::candidates_bitset(source.p + (size_t) t * ld, n, 8, 2048, bscore.p, bsel.p,
                                    reinterpret_cast<uint32_t*>(serial_rows.p + (size_t) t * li + 512), s);
    }
    CUDA_OK(cudaStreamSynchronize(s));
    REQUIRE(bit_rows.get() == serial_rows.get());
    const auto hb = byte_rows.get(), hi = bit_rows.get();
    for (int t = 0; t < T; ++t) {
        const int n = (p0 + t + 1) / ratio, B = (n + 7) / 8;
        const auto* kb = reinterpret_cast<const unsigned char*>(hb.data() + (size_t) t * lb + 512);
        const auto* ki = reinterpret_cast<const uint32_t*>(hi.data() + (size_t) t * li + 512);
        const size_t wb = n > cand_min ? B : 0, wi = n > cand_min ? (B + 31) / 32 * 4 : 0;
        if (n > cand_min) {
            for (int b = 0; b < B; ++b) REQUIRE((bool) kb[b] == (bool) ((ki[b / 32] >> (b % 32)) & 1));
            REQUIRE(kb[B - 1] == 1);
        }
        for (size_t j = wb; j < ((size_t) lb - 512) * 4; ++j) REQUIRE(kb[j] == 0xa5);
        const auto* kip = reinterpret_cast<const unsigned char*>(ki);
        for (size_t j = wi; j < ((size_t) li - 512) * 4; ++j) REQUIRE(kip[j] == 0xa5);
        for (int j = 0; j < 512; ++j) {
            REQUIRE((uint32_t) hb[(size_t) t * lb + j] == 0xa5a5a5a5u);
            REQUIRE((uint32_t) hi[(size_t) t * li + j] == 0xa5a5a5a5u);
        }
    }
    d41::candidate_mask_batch(byte_score.p, ld, ratio, p0, T, cand_min, 8, byte_rows.p, lb, 512, s);
    d41::candidate_mask_batch_bitset(bit_score.p, ld, ratio, p0, T, cand_min, 8, bit_rows.p, li, 512, s);
    CUDA_OK(cudaStreamSynchronize(s));
    same_floats(byte_score.get(), bit_score.get());
    // Original row selection consumes the new stride and leaves membership available for later sources.
    ds::indexer_batch_select(byte_score.p, ld, ratio, p0, T, 512, byte_rows.p, lb, s);
    ds::indexer_batch_select(bit_score.p, ld, ratio, p0, T, 512, bit_rows.p, li, s);
    CUDA_OK(cudaStreamSynchronize(s));
    const auto sb = byte_rows.get(), si = bit_rows.get();
    for (int t = 0; t < T; ++t) {
        const int n = (p0 + t + 1) / ratio;
        const int taken = std::min(n, 512);
        for (int j = 0; j < 512; ++j) REQUIRE(sb[(size_t) t * lb + j] == si[(size_t) t * li + j]);
        for (int j = taken; j < 512; ++j) REQUIRE(si[(size_t) t * li + j] == hi[(size_t) t * li + j]);
        REQUIRE(std::memcmp(si.data() + (size_t) t * li + 512, hi.data() + (size_t) t * li + 512,
                            ((size_t) li - 512) * sizeof(int)) == 0);
    }
    // Interior score padding and rows below the threshold are preserved by both masks.
    const auto masked = bit_score.get();
    for (int t = 0; t < T; ++t) {
        const int n = (p0 + t + 1) / ratio;
        for (int j = n > cand_min ? n : 0; j < ld; ++j) REQUIRE(same_float(masked[(size_t) t * ld + j], sc[(size_t) t * ld + j]));
    }
    byte_rows.check(); bit_rows.check(); serial_rows.check(); source.check();
    byte_score.check(); bit_score.check(); bscore.check(); bsel.check();
}

static void reused_membership_case(cudaStream_t s) {
    const int T = 9, ld = 131080, bld = (ld + 7) / 8;
    const int lb = (int) d41::candidate_row_bytes(1048576, 8, 512, false) / 4;
    const int li = (int) d41::candidate_row_bytes(1048576, 8, 512, true) / 4;
    Buffer<int> byte_rows((size_t) T * lb), bit_rows((size_t) T * li), bsel((size_t) T * 2048);
    Buffer<float> source((size_t) T * ld), a((size_t) T * ld), b((size_t) T * ld), bs((size_t) T * bld);
    for (int ratio : {1, 2}) {
        int epoch = 0;
        for (int depth : {131071, 65534, 16382, 131069, 131071}) {
            const int p0 = depth * ratio - T;
            std::vector<float> sc((size_t) T * ld);
            for (size_t i = 0; i < sc.size(); ++i) sc[i] = (float) ((i * 37 + epoch * 101) % 997) - 511.0f;
            source.put(sc); a.put(sc); b.put(sc);
            d41::candidates_batch(source.p, ld, ratio, p0, T, 16384, 8, 2048, bs.p, bld, bsel.p,
                                   byte_rows.p, lb, 512, s);
            if (epoch % 2 == 0)
                d41::candidates_batch_bitset(source.p, ld, ratio, p0, T, 16384, 8, 2048, bs.p, bld, bsel.p,
                                              bit_rows.p, li, 512, s);
            else
                for (int t = 0; t < T; ++t) {
                    const int n = (p0 + t + 1) / ratio;
                    if (n > 16384)
                        d41::candidates_bitset(source.p + (size_t) t * ld, n, 8, 2048, bs.p, bsel.p,
                                                reinterpret_cast<uint32_t*>(bit_rows.p + (size_t) t * li + 512), s);
                }
            d41::candidate_mask_batch(a.p, ld, ratio, p0, T, 16384, 8, byte_rows.p, lb, 512, s);
            d41::candidate_mask_batch_bitset(b.p, ld, ratio, p0, T, 16384, 8, bit_rows.p, li, 512, s);
            CUDA_OK(cudaStreamSynchronize(s));
            same_floats(a.get(), b.get());
            ++epoch;
        }
    }
    byte_rows.check(); bit_rows.check(); bsel.check(); source.check(); a.check(); b.check(); bs.check();
}

int main(int argc, char** argv) {
    const bool quick = argc == 2 && std::strcmp(argv[1], "--quick") == 0;
    REQUIRE(argc == 1 || quick);
    // The oracle is the pinned plain scorer, independently of any optional tensor-core arm.
    REQUIRE(setenv("STRATA_DS4_IDX", "1", 1) == 0);
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));
    int score_cases = 0, batch_cases = 0;
    std::vector<int> sizes = {1, 7, 8, 9, 511, 512, 513, 16383, 16384, 16385, 32768, 65536, 131072, 1048576};
    if (quick) sizes = {7, 513, 16383, 16384, 16385, 32769};
    for (int n : sizes)
        for (int kind = 0; kind < 4; ++kind) { score_case(n, kind, stream); ++score_cases; }
    empty_case(stream);
    exceptional_case(stream);
    reused_membership_case(stream);
    for (int ratio : {1, 2}) {
        for (int T : {1, 3, 9}) {
            for (int p0 : {0, 510 * ratio, 16382 * ratio, 32765 * ratio}) {
                batch_case(1048576, p0, T, ratio, 16384, stream);
                ++batch_cases;
            }
        }
        // Synthetic rebind/grow, prefix hit, and rewind: new row bases, changing batch widths and start positions.
        for (int n : (quick ? std::vector<int>{16385} : std::vector<int>{32768, 65536, 131072, 1048576 / ratio})) {
            batch_case(1048576, n * ratio - 7, 7, ratio, 16384, stream);
            batch_case(1048576, n * ratio - 9, 3, ratio, 16384, stream);
            batch_cases += 2;
        }
        // Small-domain candidate emitters are valid independently of the integration threshold.
        batch_case(1048576, 7, 9, ratio, 0, stream);
        ++batch_cases;
    }
    CUDA_OK(cudaStreamDestroy(stream));
    std::printf("PASS compact exact score/ID cases=%d (four late sources each), bitset batch/layout cases=%d, exceptional fallback=1\n",
                score_cases, batch_cases);
}
