#!/usr/bin/env python3
"""Offline contracts and the production header's host layout. No CUDA execution.

Run: python3 tests/compact_candidates_reference.py
Device score/order/mask checks: tests/compact_candidates_parity.cu.
"""
import heapq
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SIZES = (1, 2, 7, 8, 9, 511, 512, 513, 16383, 16384, 16385,
         32768, 65536, 131072, 1048576)


def top_ids(scores, count):
    return heapq.nsmallest(min(count, len(scores)), range(len(scores)),
                          key=lambda i: (-scores[i], i))


def candidate_blocks(scores, block=8, top=2048):
    maxima = [max(scores[i:i + block]) for i in range(0, len(scores), block)]
    if maxima:
        maxima[-1] = math.inf
    return top_ids(maxima, top)


def expand(blocks, n, block=8):
    return [j for b in sorted(blocks) for j in range(b * block, min((b + 1) * block, n))]


def dense_masked(scores, rows):
    masked = [-math.inf] * len(scores)
    for j in rows:
        masked[j] = scores[j]
    return masked


def compact_select(scores, rows, top=512):
    compact = [scores[j] for j in rows]
    # The device selection uses its original dense scratch for exceptional scores.
    if any(not math.isfinite(x) for x in compact):
        return top_ids(dense_masked(scores, rows), top)
    return [rows[i] for i in top_ids(compact, top)]


class CompactReference(unittest.TestCase):
    def test_production_header_layout(self):
        cxx = shutil.which(os.environ.get("CXX", "g++"))
        self.assertIsNotNone(cxx, "host C++ compiler required")
        scratch = Path(os.environ.get("TMPDIR", "/home/joe/.hermes/cache/scratch"))
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="compact-layout-", dir=scratch) as tmp:
            tmp = Path(tmp)
            # Only the host declarations need this type. This is not a CUDA compile.
            (tmp / "cuda_runtime.h").write_text(
                "#pragma once\n#include <cstddef>\nusing cudaStream_t = void*;\n")
            exe = tmp / "layout"
            cmd = [cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                   "-I", str(tmp), "-I", str(ROOT / "include"),
                   str(ROOT / "tests/compact_candidates_layout.cpp"), "-o", str(exe)]
            built = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            print(run.stdout.strip())

    def test_candidates_boundaries_and_absolute_ties(self):
        for n in SIZES:
            for kind in ("equal", "ties", "negative"):
                with self.subTest(n=n, kind=kind):
                    src = [0.0 if kind == "equal" else float((i * 97 + i // 8) % 101)
                           for i in range(n)]
                    if kind == "negative":
                        src = [-x - 1 for x in src]
                    blocks = candidate_blocks(src)
                    rows = expand(blocks, n)
                    self.assertEqual(len(blocks), min(2048, (n + 7) // 8))
                    self.assertIn((n - 1) // 8, blocks)
                    self.assertIn(n - 1, rows)
                    self.assertEqual(rows, sorted(set(rows)))
                    self.assertLessEqual(len(rows), 2048 * 8)
                    late = [0.0 if kind == "equal" else float((i * 31) % 67) - 33
                            for i in range(n)]
                    expected = top_ids(dense_masked(late, rows), 512)
                    self.assertEqual(compact_select(late, rows), expected)
                    if n <= 16384:
                        self.assertEqual(rows, list(range(n)))
                    if kind == "equal" and n > 16384:
                        self.assertEqual(blocks, [(n - 1) // 8] + list(range(2047)))
                        self.assertEqual(expected, list(range(512)))

    def test_bitonic_expansion_schedule(self):
        for n in SIZES:
            scores = [float((i * 37) % 997) for i in range(n)]
            blocks = candidate_blocks(scores)
            width = 1 << (len(blocks) - 1).bit_length()
            ids = blocks + [2**31 - 1] * (width - len(blocks))
            k = 2
            while k <= width:
                step = k // 2
                while step:
                    for tid in range(256):
                        for i in range(tid, width, 256):
                            j = i ^ step
                            if j > i and ((i & k == 0 and ids[i] > ids[j]) or
                                          (i & k != 0 and ids[i] < ids[j])):
                                ids[i], ids[j] = ids[j], ids[i]
                    step //= 2
                k *= 2
            self.assertEqual(ids[:len(blocks)], sorted(blocks))
            count = (len(blocks) - 1) * 8 + min(8, n - ids[len(blocks) - 1] * 8)
            rows = [ids[i // 8] * 8 + i % 8 for i in range(count)]
            self.assertEqual(rows, expand(blocks, n))

    def test_nonfinite_dense_fallback(self):
        n = 32769
        src = [float(i) for i in range(n)]
        rows = expand(candidate_blocks(src), n)
        self.assertNotIn(0, rows)
        for value in (-math.inf, math.inf):
            late = [value] * n
            self.assertEqual(compact_select(late, rows), top_ids(dense_masked(late, rows), 512))
        # Dense masking lets lower excluded IDs win a -inf tie.
        self.assertEqual(compact_select([-math.inf] * n, rows), list(range(512)))
        # +inf block ties can exclude the newest forced block. Expansion uses the selected IDs.
        blocks = candidate_blocks([math.inf] * n)
        self.assertNotIn((n - 1) // 8, blocks)
        self.assertEqual(len(expand(blocks, n)), 16384)

    def test_ratio_prefix_rollback_and_reuse_contract(self):
        # An independent token-order model. Full core prefix/rollback replay remains a device integration gate.
        for ratio in (1, 2):
            epoch = None
            for p in (16382, 16383, 16384, 32766, 32767, 32768, 65535, 131071, 1048575, 32767, 32768):
                n = (p + 1) // ratio
                self.assertEqual(n, len(range(ratio - 1, p + 1, ratio)))
                # Entry to the early layers invalidates even a repeated absolute position.
                epoch = None
                source = [float((i * 17 + p) % 29) for i in range(n)]
                rows = expand(candidate_blocks(source), n)
                saved = rows[:]
                epoch = (p, n, 3, 24)
                for layer in (24, 28, 32, 36):
                    self.assertEqual(epoch, (p, n, 3, layer))
                    self.assertNotEqual(epoch, (p - 1, n, 3, layer))
                    self.assertNotEqual(epoch, (p, n, 2, layer))
                    late = [float((i * (layer + 1)) % 37) for i in range(n)]
                    selected = compact_select(late, rows)
                    self.assertEqual(selected, top_ids(dense_masked(late, rows), 512))
                    # Scratch reuse changes the source list without mutating the retained domain.
                    scratch = [0] * 2048
                    self.assertEqual(rows, saved)
                    self.assertEqual(len(scratch), 2048)
                    epoch = (p, n, 3, layer + 4)
                self.assertEqual(epoch[-1], 40)
                # A rewind can revisit the same position with a different candidate source.
                changed = expand(candidate_blocks(list(reversed(source))), n)
                self.assertEqual(changed, sorted(set(changed)))

    def test_pinned_plain_arithmetic_and_original_functions(self):
        baseline = Path(os.environ.get("ABYSS_BASELINE", ROOT.parents[2] / "source-audit" /
            "Abyss-51ef4059389cb5a0f5af91c7ff353e4271634f78"))
        if not baseline.is_dir():
            self.skipTest("set ABYSS_BASELINE to the immutable pinned source")
        original = (baseline / "src/kernels/cuda/dsv41_fast.cu").read_text()
        updated = (ROOT / "src/kernels/cuda/dsv41_fast.cu").read_text()
        def body(text, name):
            start = text.index("{", text.index(name + "("))
            level = 1
            pos = start + 1
            while level:
                level += (text[pos] == "{") - (text[pos] == "}")
                pos += 1
            return text[start:pos]
        for name in ("warp_sum", "warp_max", "block_sum", "q8_1_store_warp", "bf", "rope_cs", "rope_pair",
                     "publish_row", "publish_kernel", "publish_batch_kernel", "q_rope_kernel", "q_rope_batch_kernel",
                     "block_max_kernel", "block_select_kernel", "keep_kernel", "mask_kernel", "block_max_batch_kernel",
                     "block_select_batch_kernel", "mask_batch_kernel", "engram_kernel", "head_kernel", "rms_rows_kernel",
                     "check", "publish", "q_rope", "publish_batch", "q_rope_batch", "candidates", "candidate_mask",
                     "candidates_batch", "candidate_mask_batch", "engram_wait_kernel", "engram_wait", "engram_add",
                     "engram_add_rows", "head_prep", "rms_rows"):
            self.assertEqual(body(updated, name), body(original, name), name)
        pinned = (baseline / "src/kernels/cuda/dsv4_fast.cu").read_text()
        plain = body(pinned, "idx_score_kernel")
        compact = body(updated, "candidate_score_kernel")
        begin, end = "    const uint16_t* kr =", "    if (lane == 0)"
        self.assertEqual(plain[plain.index(begin):plain.index(end)],
                         compact[compact.index(begin):compact.index(end)])
        self.assertEqual(body(pinned, "warp_sum"), body(updated, "warp_sum"))
        core = (ROOT / "src/core/dsv41_fast_path.cu").read_text()
        self.assertIn("if (il <= g.cand_layer || pf_sel_ != nullptr) F->v41_cpos = -1;", core)
        self.assertIn("F->v41_cpos == p64 && F->v41_cvis == n_vis && F->v41_cslot == Ly.slot", core)
        self.assertIn("il == F->v41_cnext && (il == 24 || il == 28 || il == 32 || il == 36)", core)
        self.assertIn("pf_sel_ == nullptr && F->v41_crows != nullptr", core)
        prefill = (ROOT / "src/core/glm_prefill.cu").read_text()
        self.assertIn("pf_keep_bits_ = nullptr;", prefill)
        self.assertIn("F->v41_cpos = -1;", prefill)
        for name in ("candidates_batch_bitset", "candidate_mask_batch_bitset", "candidates_bitset", "candidate_mask_bitset"):
            self.assertIn("d41::" + name + "(", prefill)

    def test_additive_api_surface(self):
        header = (ROOT / "include/strata/kernels/dsv41_fast.hpp").read_text()
        for name in ("candidate_rows", "indexer_candidate_scores", "indexer_candidate_select",
                     "candidates_bitset", "candidate_mask_bitset", "candidates_batch_bitset",
                     "candidate_mask_batch_bitset", "candidate_row_bytes"):
            self.assertIn(name + "(", header)


if __name__ == "__main__":
    unittest.main(verbosity=2)
