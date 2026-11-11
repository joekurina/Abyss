#!/usr/bin/env python3
"""Host-only layout, gate and additive-source checks for the idx32 candidate."""
import argparse
import difflib
from pathlib import Path
import re
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = None


def half(x):
    return struct.unpack("e", struct.pack("e", x))[0]


class Idx32SourceTest(unittest.TestCase):
    def test_optional_apis_and_dispatch(self):
        source = (ROOT / "src/kernels/cuda/dsv4_fast.cu").read_text()
        header = (ROOT / "include/strata/kernels/dsv4_fast.hpp").read_text()
        names = ("indexer_idx32_supported", "indexer_scores_idx32",
                 "indexer_batch_scores_idx32", "indexer_scores_reference",
                 "indexer_batch_scores_reference")
        for name in names:
            self.assertTrue(name + "(" in source, f"missing definition: {name}")
            self.assertTrue(name + "(" in header, f"missing declaration: {name}")
        self.assertIn('getenv("STRATA_DS41_IDX32")', source)
        self.assertIn('v[0] == \'1\' && v[1] == \'\\0\'', source)
        self.assertIn("idx_heads != IDX32_HEADS || idx_dim != IDX32_DIM", source)
        self.assertIn("major == 8 && minor == 0", source)
        self.assertIn("decode.ptxVersion >= 80 && batch.ptxVersion >= 80", source)
        for name in ("indexer", "indexer_scores", "indexer_batch", "indexer_batch_scores"):
            match = re.search(r"\nvoid " + name + r"\([^\n]*(?:\n[^{}]*)?\{", source)
            self.assertIsNotNone(match, name)
            start = match.end()
            end = source.index("\n}\n", start)
            body = source[start:end]
            self.assertIn("idx32_enabled()", body, name)
            self.assertIn("#if !defined(__HIPCC__) && !defined(STRATA_USE_HIP)", body, name)
            callee = "indexer_batch_scores_idx32(" if "batch" in name else "indexer_scores_idx32("
            self.assertIn(callee, body, name)

    def test_b_fragment_covers_32_by_128_once(self):
        seen = set()
        for i in range(4 * 8 * 32):
            lane, ks, nt = i & 31, (i >> 5) & 7, i >> 8
            h = nt * 8 + (lane >> 2)
            k = ks * 16 + (lane & 3) * 2
            for d in (k, k + 1, k + 8, k + 9):
                self.assertNotIn((h, d), seen)
                seen.add((h, d))
        self.assertEqual(seen, {(h, d) for h in range(32) for d in range(128)})

    def test_a_fragment_and_reduction_mapping(self):
        seen = set()
        out = {}
        for lane in range(32):
            g, t4 = lane >> 2, lane & 3
            for ks in range(8):
                k = ks * 16 + t4 * 2
                for r in (g, g + 8):
                    for d in (k, k + 1, k + 8, k + 9):
                        self.assertNotIn((r, d), seen)
                        seen.add((r, d))
            for nt in range(4):
                for r in (g, g + 8):
                    out.setdefault(r, []).extend((nt * 8 + t4 * 2, nt * 8 + t4 * 2 + 1))
        self.assertEqual(seen, {(r, d) for r in range(16) for d in range(128)})
        for heads in out.values():
            self.assertEqual(sorted(heads), list(range(32)))

    def test_causal_tiles_and_padded_rows(self):
        # Warp-uniform skipping still reaches the next block-wide barrier.
        for T in (1, 2, 31, 32, 33, 65):
            for ratio in (1, 2, 4, 7):
                for p0 in (0, 1, 31, 511 * ratio - 1):
                    maximum = (p0 + T) // ratio
                    written = set()
                    for t0 in range(0, T, 32):
                        tn = min(32, T - t0)
                        nrow = (p0 + t0 + tn) // ratio
                        for j0 in range(0, maximum, 256):
                            tf = 0
                            while tf < tn and (p0 + t0 + tf + 1) // ratio <= j0:
                                tf += 1
                            for ti in range(tf, tn):
                                t = t0 + ti
                                nvis = (p0 + t + 1) // ratio
                                for warp in range(8):
                                    wrow = j0 + warp * 32
                                    if wrow >= nvis:
                                        continue
                                    for r in range(wrow, wrow + 32):
                                        if r < nvis:
                                            self.assertLess(r, nrow)
                                            self.assertNotIn((t, r), written)
                                            written.add((t, r))
                    expected = {(t, r) for t in range(T) for r in range((p0 + t + 1) // ratio)}
                    self.assertEqual(written, expected)

    def test_two_halves_retain_small_residual(self):
        # The CUDA residual fixture needs the low part to preserve row ordering.
        q = 1.0001
        hi = half(q)
        lo = half(q - hi)
        self.assertNotEqual(lo, 0.0)
        self.assertLess(abs((hi + lo) - q), 1e-7)
        self.assertGreater(abs(hi - q), 9e-5)

    def test_original_lines_preserved(self):
        if BASELINE is None:
            self.skipTest("pass --baseline to verify additive edits")
        for name in ("src/kernels/cuda/dsv4_fast.cu", "include/strata/kernels/dsv4_fast.hpp"):
            before = (BASELINE / name).read_text().splitlines(keepends=True)
            after = (ROOT / name).read_text().splitlines(keepends=True)
            changes = difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes()
            self.assertTrue(all(tag in ("equal", "insert") for tag, *_ in changes), name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    args, extra = parser.parse_known_args()
    BASELINE = args.baseline
    unittest.main(argv=[__file__, *extra])
