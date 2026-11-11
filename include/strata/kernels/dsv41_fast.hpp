// include/strata/kernels/dsv41_fast.hpp - the deepseek41 (DeepSeek V4.1 Flash) decode kernels that differ from
// deepseek4's (dsv4_fast.hpp): the compressed rows of a KV-source layer (pairs pooled by a per-channel softmax gate,
// or one row a token) with their index keys, q's RoPE without the per-head norm, the candidate blocks, Engram's gate
// and the head's collapse.  The math is in Abyss-tools/worklog/2026-10-18_deepseek41-spec.md (sections 2, 3, 5, 6),
// transcribed from antirez's ds4 (ds4.c ds41_*, ds4_deepseek41_cuda.cuh) and llama.cpp's deepseek41.cpp.
#pragma once

#include "strata/kernels/dsv4_fast.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::dsv41 {

using dsv4::Rope;

/// One token at position p on a KV-source layer (spec 5.3).  v (and g, ratio 2) = the compressor's projections
/// (D = 512 each).  Ratio 2: an even p parks v and g in pend (2 x D) and writes nothing; an odd p pools the pair per
/// channel (softmax of the two gate values) into row p / 2, RoPE position p - 1.  Ratio 1: row p = v, RoPE position p.
/// A completed row c: c = rms(c) * norm; the index key ik = rms(idx_k . c) * idx_k_norm (BF16 idx_k [128][D], from
/// the UNROTATED c), RoPE on ik's last n_rot dims -> idx_cache row (FP16, 128); RoPE on c's last n_rot dims ->
/// comp_cache row (FP16, D).
struct PublishArgs {
    const float* v = nullptr;
    const float* g = nullptr;
    float* pend = nullptr;
    const float* norm = nullptr;
    const uint16_t* idx_k = nullptr;
    const float* idx_k_norm = nullptr;
    uint16_t* comp_cache = nullptr;
    uint16_t* idx_cache = nullptr;
    int ratio = 2, p = 0, D = 512, idx_dim = 128, n_rot = 64;
    Rope rope;
    float eps = 1e-20f;
};
void publish(const PublishArgs& a, cudaStream_t s);

/// publish for the T tokens p0 .. p0 + T - 1 of a prompt sub-batch: a.v / a.g their rows (D apart), a.pend the slot's
/// pend ring (ring entries of 2 x D by p / 2: a pair's even token from before the batch, and the last token's park).
void publish_batch(const PublishArgs& a, int p0, int T, int ring, cudaStream_t s);
/// q_rope for T tokens at p0 + t: q rows n_head * head_dim floats apart, iq rows idx_heads * idx_dim.
void q_rope_batch(float* q, int n_head, int head_dim, int n_rot, const Rope& rope, float* iq, int idx_heads,
                  int idx_dim, const Rope& comp_rope, int p0, int T, cudaStream_t s);

/// q (n_head x head_dim): RoPE on the last n_rot dims at p (V4.1 has no per-head q norm); iq (may be null): RoPE on
/// its last n_rot dims at p with comp_rope.
void q_rope(float* q, int n_head, int head_dim, int n_rot, const Rope& rope, float* iq, int idx_heads, int idx_dim,
            const Rope& comp_rope, int p, cudaStream_t s);

/// The candidate blocks (spec 2) at the candidate-source layer: bs[c] = max of score over rows [8c, 8c+8) < n_vis,
/// the newest block +inf, then the top min(top_blocks, B) blocks -> keep[c] = 1 (0 otherwise), B = ceil(n_vis / bs).
/// bscore (B floats) and bsel (top_blocks ints) are scratch.
void candidates(const float* score, int n_vis, int block, int top_blocks, float* bscore, int* bsel, unsigned char* keep,
                cudaStream_t s);
/// score[j] = -inf where keep[j / block] == 0, j < n_vis.
void candidate_mask(float* score, int n_vis, int block, const unsigned char* keep, cudaStream_t s);

/// The prompt path's candidates / candidate_mask over a group of T rows: row t is position p0 + t and sees
/// (p0 + t + 1) / ratio compressed rows of its score row (score + t * ld); only rows that see more than cand_min take
/// part.  Its keep bytes sit in its selection slot (sel + t * sel_ld, from int keep_off on); bscore (T x bld floats,
/// bld >= the blocks of the longest row) and bsel (T x top_blocks ints) are scratch.
void candidates_batch(const float* score, int ld, int ratio, int p0, int T, int cand_min, int block, int top_blocks,
                      float* bscore, int bld, int* bsel, int* sel, int sel_ld, int keep_off, cudaStream_t s);
void candidate_mask_batch(float* score, int ld, int ratio, int p0, int T, int cand_min, int block, const int* sel,
                          int sel_ld, int keep_off, cudaStream_t s);

/// Compact decode: bsel holds the original candidates' top min(top_blocks, ceil(n_vis / block), 2048) IDs.
/// Expand them in ascending absolute row order into rows (top_blocks * block ints), with the valid count in nrows.
/// Both outputs live through the last consumer. bsel remains scratch. The final partial block contributes valid rows.
void candidate_rows(const int* bsel, int n_vis, int block, int top_blocks, int* rows, int* nrows, cudaStream_t s);
/// The plain FP32 indexer arithmetic on rows[0, *nrows), idx_dim 128. capacity >= *nrows, idx_heads <= 64.
/// score holds capacity floats in compact ordinal order. This entry point always uses the plain warp arithmetic.
void indexer_candidate_scores(const float* iq, const uint16_t* K, const float* w, int idx_heads, const int* rows,
                              const int* nrows, int capacity, float* score, cudaStream_t s);
/// Select by score descending, then absolute row ID. rows is ascending and selection maps ordinals back to rows.
/// dense_scratch (n_vis floats, disjoint from score) supports the original masked order for nonfinite scores.
/// 1 <= top_k <= min(*nrows, 2048). sel holds top_k absolute IDs, ready for reuse by later layers.
void indexer_candidate_select(const float* score, const int* rows, const int* nrows, int n_vis, int top_k,
                              float* dense_scratch, int* sel, cudaStream_t s);

/// Prompt membership: one uint32_t word per 32 blocks, bit c % 32 of word c / 32 for block c.
/// The scalar buffer holds ceil(ceil(n_vis / block) / 32) words. Batch keep_off and sel_ld are in ints.
void candidates_bitset(const float* score, int n_vis, int block, int top_blocks, float* bscore, int* bsel,
                       uint32_t* keep, cudaStream_t s);
void candidate_mask_bitset(float* score, int n_vis, int block, const uint32_t* keep, cudaStream_t s);
void candidates_batch_bitset(const float* score, int ld, int ratio, int p0, int T, int cand_min, int block,
                             int top_blocks, float* bscore, int bld, int* bsel, int* sel, int sel_ld, int keep_off,
                             cudaStream_t s);
void candidate_mask_batch_bitset(float* score, int ld, int ratio, int p0, int T, int cand_min, int block,
                                 const int* sel, int sel_ld, int keep_off, cudaStream_t s);

/// Frozen process-wide prompt layout. STRATA_DS41_CANDIDATE_BITSET=1 opts in, default 0.
bool candidate_bitset_enabled();
/// Full-context row capacity, including the original extra block and 16-byte membership alignment. block > 0.
inline size_t candidate_keep_bytes(size_t max_ctx, int block, bool bitset) {
    const size_t blocks = (max_ctx + (size_t) block) / (size_t) block;
    const size_t bytes = bitset ? ((blocks + 31) / 32) * sizeof(uint32_t) : blocks;
    return (bytes + 15) / 16 * 16;
}
inline size_t candidate_row_bytes(size_t max_ctx, int block, int top_k, bool bitset) {
    return (size_t) top_k * sizeof(int) + candidate_keep_bytes(max_ctx, block, bitset);
}

/// Engram's add (spec 3.5) into the 4 streams of R (hc x n_embd) in place: kv = 5 x n_embd (4 keys, the value),
/// qn / kn = [4][n_embd] norm weights; per stream h: s = rms(R_h) * qn_h . rms(key_h) * kn_h / sqrt(n_embd),
/// g = sigmoid(sign(s) sqrt(max(|s|, 1e-6))), R_h += g * value.
void engram_add(float* R, const float* kv, const float* qn, const float* kn, int n_embd, float eps, cudaStream_t s);
/// engram_add for T tokens: R rows 4 x n_embd apart, kv rows 5 x n_embd apart.
void engram_add_rows(float* R, const float* kv, const float* qn, const float* kn, int n_embd, float eps, int T,
                     cudaStream_t s);
/// Waits on the device until the host-pinned *flag == want, then copies n floats from the pinned src to dst.
void engram_wait(const unsigned int* flag, unsigned int want, const float* src, float* dst, int n, cudaStream_t s);

/// The head's prologue (spec 6.2): h = sum_s pre_s R_s (the last sublayer's pre), x = rms(h) * norm_w -> x (f32) and
/// xq (q8_1); h (may be null): h itself, before the norm (the DSpark drafter's confidence reads it).
void head_prep(const float* R, const float* pre, const float* norm_w, float eps, int n_embd, float* x, void* xq,
               cudaStream_t s, float* h = nullptr);

/// rows rows of n floats (n % 32 == 0, any width - glmf::rms_q8 stops at 4096): x[r] = rms(in[r]) * w -> x (f32) and
/// xq (q8_1, n / 32 blocks a row).
void rms_rows(const float* in, int rows, const float* w, float eps, int n, float* x, void* xq, cudaStream_t s);

}  // namespace strata::kernels::dsv41
