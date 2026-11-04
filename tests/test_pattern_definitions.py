import unittest

import numpy as np

from keyloom_mania.evaluation.definitions import EVALUATION_LABELS, PRIORITY, RULES, bar_matches, dominant, row_context
from keyloom_mania.segment_labels import LABELS


def interior_labels(sequence, step=.25, bpm=180., repeat=16, timeline=None):
    """Dominant labels of the interior bars of a repeated fixture."""
    if timeline is None:
        rows = sequence * repeat
        beats = np.arange(len(rows), dtype=np.float64) * step
    else:
        rows, beats = [columns for _, columns in timeline], np.asarray([beat for beat, _ in timeline], dtype=np.float64)
    masks = np.asarray([sum(1 << column for column in columns) for columns in rows])
    context = row_context(beats, masks, np.full(len(beats), 60_000 / bpm), beats - np.floor(beats))
    bars = np.floor(beats / 4).astype(int)
    results = [bar_matches(context, np.flatnonzero(bars == bar)) for bar in np.unique(bars)[1:-1]]
    return [dominant(result) for result in results], results


class PatternDefinitionTests(unittest.TestCase):
    def assertDominant(self, label, sequence, **kwargs):
        labels, _ = interior_labels(sequence, **kwargs)
        self.assertTrue(labels and all(item == label for item in labels), (label, labels))

    def test_every_label_has_a_rule_and_a_priority(self):
        self.assertEqual(set(RULES), set(EVALUATION_LABELS))
        self.assertEqual(sorted(PRIORITY), sorted(EVALUATION_LABELS))
        # Tech is evaluator-only; the released planner vocabulary stays fifteen labels.
        self.assertEqual(EVALUATION_LABELS[:-1], LABELS)

    def test_light_jumpstream_and_handstream_are_recognised(self):
        # One jump or hand per beat on a 1/4 stream: textbook osu! wiki examples.
        self.assertDominant("Jumpstream", [[0, 2], [1], [3], [0], [1, 3], [0], [2], [1]])
        self.assertDominant("Handstream", [[0, 1, 3], [2], [1], [3], [0, 1, 2], [3], [0], [2]])

    def test_stream_requires_constant_intervals(self):
        sequence = [[0, 2], [1], [3], [0], [1, 3], [0], [2], [1]] * 16
        timeline, beat = [], 0.
        for index, columns in enumerate(sequence):
            timeline.append((beat, columns))
            beat += .25 if index % 2 == 0 else .75
        labels, _ = interior_labels(None, timeline=timeline)
        self.assertTrue(all(label is None for label in labels))

    def test_single_column_jacks_are_recognised(self):
        self.assertDominant("Minijacks", [[0], [1], [1], [2], [3], [2], [2], [1]])
        self.assertDominant("Jacks", [[0]] * 3 + [[2]] * 3 + [[1]] * 3 + [[3]] * 3, repeat=8)
        self.assertDominant("Longjacks", [[0]] * 8 + [[1], [2], [3], [2]], repeat=8)

    def test_minijacks_break_a_pure_stream(self):
        _, results = interior_labels([[0], [1], [1], [2], [3], [2], [2], [1]])
        self.assertFalse(any(result["Stream"] for result in results))

    def test_rolls_and_trills_by_shape(self):
        self.assertDominant("Rolls", [[0], [1], [2], [3]], repeat=32)
        self.assertDominant("Trills", [[0], [1]], repeat=64)
        self.assertDominant("Trills", [[0], [2]], repeat=64)
        self.assertDominant("Stream", [[0], [2], [1], [3], [2], [0], [3], [1]])

    def test_chord_trills_follow_hand_usage(self):
        self.assertDominant("Jumptrill", [[0, 1], [2, 3]], repeat=32)
        self.assertDominant("Split Trill", [[0, 2], [1, 3]], repeat=32)

    def test_jack_families(self):
        self.assertDominant("Chordjacks", [[0, 1, 2], [1, 2, 3], [0, 1, 2], [0, 2, 3]], step=.5)
        self.assertDominant("Gluts", [[0, 1], [1, 2], [2, 3], [0, 3]])
        self.assertDominant("Quadstream", [[0, 1, 2, 3], [1], [2], [0], [0, 1, 2, 3], [3], [1], [2]])

    def test_tech_needs_mixed_divisions_awkward_ratios_or_bursts(self):
        stream = [[0], [2], [1], [3], [2], [0], [3], [1]]
        poly = sorted([(bar * 4 + k * .25, stream[k]) for bar in range(16) for k in range(8)]
                      + [(bar * 4 + 2 + k / 3, stream[(k + 3) % 8]) for bar in range(16) for k in range(6)])
        labels, _ = interior_labels(None, timeline=poly)
        self.assertTrue(all(label == "Tech" for label in labels))
        burst = sorted([(bar * 4 + k * .25, stream[k % 8]) for bar in range(16) for k in range(16)]
                       + [(bar * 4 + 1.125, [1]) for bar in range(16)] + [(bar * 4 + 1.375, [0]) for bar in range(16)])
        labels, _ = interior_labels(None, timeline=burst)
        self.assertTrue(all(label == "Tech" for label in labels))
        _, results = interior_labels(stream)
        self.assertFalse(any(result["Tech"] for result in results))

    def test_sparse_bars_have_no_style(self):
        labels, _ = interior_labels([[0], [2], [1]], step=1.5, repeat=16)
        self.assertTrue(all(label is None for label in labels))


if __name__ == "__main__":
    unittest.main()
