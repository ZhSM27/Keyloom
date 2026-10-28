import itertools
import unittest

import numpy as np

from keyloom_mania.music_structure import MusicStructurePolicy, music_structure
from keyloom_mania.plan_compiler import compile_plan_v2
from keyloom_mania.role_style_plan import RoleStylePolicy, _schedule, active_style_plan
from keyloom_mania.segment_labels import LABELS
from keyloom_mania.style_hierarchy import compatible_styles


def example():
    evidence = np.full((12, len(LABELS)), 0.02, dtype=np.float32)
    evidence[:, LABELS.index("Jumpstream")] = 0.85
    evidence[4:8, LABELS.index("Jumpstream")] = 0.15
    evidence[4:8, LABELS.index("Handstream")] = 0.95
    audio = np.zeros((12, 3), dtype=np.float32)
    audio[4:8] = [2, -1, 1]
    return evidence, np.full(12, 3, dtype=np.float32), np.full(12, 1.5, dtype=np.float32), audio


def plan(evidence, notes, rows, **kwargs):
    return active_style_plan(evidence, notes, rows, np.full(len(LABELS), 0.5), {},
                             "Jumpstream", ["Handstream"], **kwargs)


class MusicStructureTests(unittest.TestCase):
    def test_sustained_learned_owner_change_can_refine_a_flat_audio_section(self):
        count = 12
        probabilities = np.zeros((count, 15), dtype=np.float32)
        owner = probabilities.copy()
        owner[:, LABELS.index("Jumpstream")] = .95
        owner[4:8, LABELS.index("Jumpstream")] = .05
        owner[4:8, LABELS.index("Handstream")] = .95
        result = music_structure(np.ones(count), np.ones(count), probabilities, np.zeros(count, dtype=bool),
                                 music=np.ones((count, 3)), owner_scores=owner)
        self.assertEqual([segment["start_bar"] for segment in result["segments"]], [0, 4, 8])
        self.assertEqual(result["segments"][1]["boundary_source"], "learned_owner")
    def test_flat_density_can_have_a_secondary_and_primary_reprise(self):
        evidence, notes, rows, audio = example()
        result = plan(evidence, notes, rows, music=audio)
        expected = ["Jumpstream"] * 4 + ["Handstream"] * 4 + ["Jumpstream"] * 4
        self.assertEqual([LABELS[index - 1] for index in result["style_ids"]], expected)
        self.assertEqual([(s["start_bar"], s["end_bar"]) for s in result["segments"]], [(0, 4), (4, 8), (8, 12)])
        np.testing.assert_array_equal(result["segment_ids"], result["phrase_ids"])
        self.assertEqual(result["repeat_group_ids"][0], result["repeat_group_ids"][8])
        self.assertNotEqual(result["repeat_group_ids"][0], result["repeat_group_ids"][4])
        self.assertGreater(result["segments"][2]["repeat_confidence"], 0.99)

    def test_audio_boundaries_are_not_fixed_four_bar_blocks(self):
        evidence = np.full((16, len(LABELS)), 0.1, dtype=np.float32)
        audio = np.zeros((16, 3), dtype=np.float32)
        audio[5:11] = 3
        result = plan(evidence, np.ones(16), np.ones(16), acoustic=audio)
        self.assertEqual([s["start_bar"] for s in result["segments"]], [0, 5, 11])

    def test_strong_section_is_allowed_to_use_explicit_secondary(self):
        evidence, notes, rows, audio = example()
        notes[4:8], rows[4:8] = 6, 3
        result = plan(evidence, notes, rows, music=audio)
        self.assertTrue(result["secondary_section"][4:8].all())

    def test_only_requested_and_directional_lower_styles_are_allowed(self):
        for label in LABELS:
            with self.subTest(label=label):
                evidence = np.full((10, len(LABELS)), 0.95, dtype=np.float32)
                result = active_style_plan(evidence, np.arange(1, 11), np.ones(10),
                                           np.full(len(LABELS), 0.5), {}, label)
                self.assertTrue(np.isin(result["style_ids"], [LABELS.index(name) + 1 for name in compatible_styles(label)]).all())
                self.assertTrue(result["main_section"].all())
                self.assertFalse(result["secondary_section"].any())

    def test_handstream_can_mix_jumpstream_but_reverse_is_forbidden(self):
        evidence = np.full((16, len(LABELS)), .02, dtype=np.float32)
        evidence[:, LABELS.index("Handstream")] = .8
        evidence[4:8, LABELS.index("Jumpstream")] = .98
        audio = np.zeros((16, 3), dtype=np.float32)
        audio[4:8] = 3
        args = (evidence, np.ones(16) * 4, np.ones(16) * 2, np.ones(len(LABELS)), {})
        hs = active_style_plan(*args, "Handstream", music=audio)
        self.assertTrue((hs["style_ids"][4:8] == LABELS.index("Jumpstream") + 1).all())
        self.assertTrue(hs["main_section"].all())
        self.assertFalse(hs["secondary_section"].any())
        js = active_style_plan(*args, "Jumpstream", music=audio)
        self.assertFalse(js["patterns"][:, LABELS.index("Handstream")].any())

    def test_chordjack_can_mix_minijack_but_reverse_is_forbidden(self):
        evidence = np.full((16, len(LABELS)), .02, dtype=np.float32)
        evidence[:, LABELS.index("Chordjacks")] = .8
        evidence[4:8, LABELS.index("Minijacks")] = .98
        audio = np.zeros((16, 3), dtype=np.float32)
        audio[4:8] = 3
        args = (evidence, np.ones(16) * 4, np.ones(16) * 2, np.ones(len(LABELS)), {})
        cj = active_style_plan(*args, "Chordjacks", music=audio)
        self.assertTrue(cj["patterns"][4:8, LABELS.index("Minijacks")].all())
        self.assertTrue(cj["main_section"].all())
        mj = active_style_plan(*args, "Minijacks", music=audio)
        self.assertFalse(mj["patterns"][:, LABELS.index("Chordjacks")].any())

    def test_same_family_secondary_only_appears_in_its_own_support_segment(self):
        evidence, notes, rows, audio = example()
        result = plan(evidence, notes, rows, music=audio)
        self.assertTrue((result["roles"][4:8] == "support").all())
        self.assertTrue((result["patterns"][result["main_section"].astype(bool), LABELS.index("Handstream")] == 0).all())
        self.assertTrue((result["patterns"].sum(axis=1) == 1).all())

    def test_silence_timing_boundaries_and_short_tails_are_preserved(self):
        evidence = np.full((9, len(LABELS)), 0.2, dtype=np.float32)
        notes, rows = np.ones(9), np.ones(9)
        notes[3:5], rows[3:5] = 0, 0
        sections = np.asarray([0] * 7 + [1] * 2)
        result = plan(evidence, notes, rows, music=np.zeros((9, 2)), timing_sections=sections)
        self.assertFalse(result["patterns"][3:5].any())
        self.assertTrue((result["segment_ids"][3:5] == -1).all())
        self.assertTrue((result["roles"][3:5] == "quiet").all())
        self.assertNotEqual(result["segment_ids"][6], result["segment_ids"][7])
        for segment in result["segments"]:
            self.assertEqual(len(set(sections[segment["start_bar"]:segment["end_bar"]])), 1)

    def test_all_quiet_single_bar_and_empty_secondary(self):
        evidence = np.ones((1, len(LABELS)), dtype=np.float32)
        result = plan(evidence, np.zeros(1), np.zeros(1))
        self.assertEqual(result["segments"], [])
        self.assertEqual(int(result["patterns"].sum()), 0)
        self.assertEqual(result["unused_secondary_styles"][0]["reason"], "insufficient_local_evidence")
        result = active_style_plan(evidence, np.ones(1), np.ones(1), np.ones(len(LABELS)), {}, "Minijacks")
        self.assertEqual(result["segments"][0]["style"], "Minijacks")

    def test_low_evidence_secondary_is_not_forced_to_appear(self):
        evidence, notes, rows, audio = example()
        evidence[:, LABELS.index("Handstream")] = 0.119
        result = plan(evidence, notes, rows, music=audio)
        self.assertFalse(result["secondary_section"].any())
        self.assertEqual(result["unused_secondary_styles"][0]["reason"], "insufficient_local_evidence")

    def test_density_and_inputs_remain_identical_to_old_compiler(self):
        evidence, notes, rows, audio = example()
        originals = [a.copy() for a in (evidence, notes, rows, audio)]
        result = plan(evidence, notes, rows, music=audio)
        raw = compile_plan_v2(evidence, notes, rows, np.full(len(LABELS), 0.5), {})
        for name in ("target_notes_per_beat", "target_rows_per_beat", "target_note_budget", "target_row_budget", "quiet"):
            np.testing.assert_array_equal(result[name], raw[name])
        for current, original in zip((evidence, notes, rows, audio), originals):
            np.testing.assert_array_equal(current, original)

    def test_constant_audio_has_no_invented_phrase_boundaries(self):
        evidence = np.full((20, len(LABELS)), 0.1, dtype=np.float32)
        result = plan(evidence, np.ones(20), np.ones(20), music=np.ones((20, 4)))
        self.assertEqual(len(result["segments"]), 1)

    def test_fallback_is_explicit_and_invalid_audio_is_rejected(self):
        evidence, notes, rows, audio = example()
        self.assertIn("planner_evidence_fallback", plan(evidence, notes, rows)["role_source"])
        for invalid in (audio[:-1], np.ones(12), np.full((12, 3), np.nan)):
            with self.assertRaises(ValueError):
                plan(evidence, notes, rows, music=invalid)
        with self.assertRaises(ValueError):
            plan(evidence, notes, rows, timing_sections=np.ones(11))
        with self.assertRaises(ValueError):
            plan(evidence, notes, rows, policy=RoleStylePolicy(minimum_segment_bars=0))

    def test_secondary_order_and_repeat_schedule_are_deterministic(self):
        evidence, notes, rows, audio = example()
        args = (evidence, notes, rows, np.ones(len(LABELS)), {}, "Jumpstream")
        first = active_style_plan(*args, ["Chordjacks", "Handstream"], music=audio)
        second = active_style_plan(*args, ["Handstream", "Chordjacks"], music=audio)
        np.testing.assert_array_equal(first["style_ids"], second["style_ids"])

    def test_dynamic_programming_matches_exhaustive_global_objective(self):
        emissions = np.asarray([[0.8, 0.4, 0.7], [0.2, 0.9, 0.8], [0.8, 0.7, 0.1], [0.5, 0.8, 0.9]])
        lengths = np.asarray([3, 2, 4, 3])
        intervals = np.asarray([[0, 3], [3, 5], [7, 11], [11, 14]])
        boundaries = np.asarray([1, .7, 1, .4])
        policy = RoleStylePolicy()
        def objective(choices):
            value = sum(emissions[i, choice] * lengths[i] for i, choice in enumerate(choices))
            for i in range(1, len(choices)):
                if intervals[i, 0] == intervals[i - 1, 1] and choices[i] != choices[i - 1]:
                    value -= policy.switch_penalty * (1 - .75 * boundaries[i])
            secondary = sum(length for choice, length in zip(choices, lengths) if choice)
            excess = max(0, secondary / lengths.sum() - (1 - policy.primary_target_fraction))
            return value - policy.primary_share_penalty * lengths.sum() * excess ** 2
        result = _schedule(emissions, lengths, boundaries, intervals, policy)
        best = max(objective(choices) for choices in itertools.product(range(3), repeat=4))
        self.assertAlmostEqual(objective(result), best)


if __name__ == "__main__":
    unittest.main()
