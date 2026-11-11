// Host-only layout checks. compact_candidates_reference.py supplies a cudaStream_t header shim.
#include "strata/kernels/dsv41_fast.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace d41 = strata::kernels::dsv41;

int main() {
    const size_t contexts[] = {1, 7, 8, 9, 511, 512, 513, 16383, 16384, 16385,
                               32768, 65536, 131072, 1048576};
    for (size_t ctx : contexts) {
        const size_t blocks = (ctx + 8) / 8;
        const size_t byte_keep = ((blocks + 15) / 16) * 16;
        const size_t bit_keep = ((((blocks + 31) / 32) * 4 + 15) / 16) * 16;
        assert(d41::candidate_keep_bytes(ctx, 8, false) == byte_keep);
        assert(d41::candidate_keep_bytes(ctx, 8, true) == bit_keep);
        for (bool bits : {false, true}) {
            const size_t row = d41::candidate_row_bytes(ctx, 8, 512, bits);
            assert(row == 2048 + (bits ? bit_keep : byte_keep));
            assert(row % 16 == 0);
            // A grown or rebound chunk owns a new row base and the same context-derived stride.
            for (size_t T : {1u, 3u, 17u}) {
                const size_t guard = 64;
                std::vector<unsigned char> arena(2 * guard + T * row, 0xa5);
                for (size_t t = 0; t < T; ++t) {
                    unsigned char* dst = arena.data() + guard + t * row;
                    for (int i = 0; i < 512; ++i) {
                        const int id = i + (int) t;
                        std::memcpy(dst + i * sizeof(int), &id, sizeof(int));
                    }
                    const size_t n_vis = ctx - t % ctx;
                    const size_t B = (n_vis + 7) / 8;
                    unsigned char* keep = dst + 2048;
                    const size_t used = bits ? ((B + 31) / 32) * 4 : B;
                    std::memset(keep, 0, used);
                    for (size_t b = 0; b < B; ++b) {
                        const bool retained = b % 3 == 0 || b + 1 == B;
                        if (bits && retained) {
                            uint32_t word;
                            std::memcpy(&word, keep + (b / 32) * 4, 4);
                            word |= uint32_t(1) << (b % 32);
                            std::memcpy(keep + (b / 32) * 4, &word, 4);
                        } else if (!bits) keep[b] = retained;
                    }
                    for (size_t j = 0; j < n_vis; ++j) {
                        const size_t b = j / 8;
                        uint32_t word = 0;
                        if (bits) std::memcpy(&word, keep + (b / 32) * 4, 4);
                        const bool got = bits ? (word >> (b % 32)) & 1 : keep[b];
                        assert(got == (b % 3 == 0 || b + 1 == B));
                    }
                    for (size_t i = used; i < row - 2048; ++i) assert(keep[i] == 0xa5);
                    for (int i = 0; i < 512; ++i) {
                        int id;
                        std::memcpy(&id, dst + i * sizeof(int), sizeof(int));
                        assert(id == i + (int) t);
                    }
                }
                for (size_t i = 0; i < guard; ++i) {
                    assert(arena[i] == 0xa5);
                    assert(arena[guard + T * row + i] == 0xa5);
                }
            }
        }
    }
    const size_t old_row = d41::candidate_row_bytes(1048576, 8, 512, false);
    const size_t new_row = d41::candidate_row_bytes(1048576, 8, 512, true);
    assert(old_row == 133136);
    assert(new_row == 18448);
    std::printf("layout: old=%zu bitset=%zu bytes/row, T32768 old=%zu bitset=%zu bytes\n",
                old_row, new_row, old_row * size_t(32768), new_row * size_t(32768));
}
