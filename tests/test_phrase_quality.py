import unittest
from collections import Counter

import numpy as np

from keyloom_mania.phrase_quality import canonical_motif, diversity_summary, periodic_runs


class PhraseQualityTests(unittest.TestCase):
    def test_two_row_cycle_is_detected_while_pure_jack_is_separate(self):
        report = periodic_runs([9, 4] * 40)
        self.assertEqual(report[0]["longest_bars"], 5)
        self.assertEqual(periodic_runs([3] * 80)[0]["longest_bars"], 0)

    def test_silence_breaks_dense_repetition(self):
        report = periodic_runs([9, 4] * 16 + [0] * 16 + [9, 4] * 16)
        self.assertEqual(report[0]["longest_bars"], 2)

    def test_mirror_is_canonical_but_count_only_is_not_layout_identity(self):
        self.assertEqual(canonical_motif([1, 6]), canonical_motif([8, 6]))
        self.assertNotEqual(canonical_motif([1, 6]), canonical_motif([1, 3]))
        summary = diversity_summary(Counter(a=8, b=2))
        self.assertEqual(summary["top1_fraction"], 0.8)
        self.assertGreater(summary["effective_variants"], 1)


if __name__ == "__main__":
    unittest.main()
