import unittest
import json
from unittest.mock import patch

import numpy as np
import torch

from keyloom_mania.lane_control import LaneControlConfig, LaneControlState
from keyloom_mania.main_style import STYLE_NAMES, one_hot_plan
from keyloom_mania.local_rhythm import LOCAL_WIDTH
from keyloom_mania.style_rhythm import RHYTHM_GRIDS, joint_mask_probability, rhythm_bar_metrics, rhythm_targets
from keyloom_mania.style_sequence import StyleSequenceGenerator, sequence_loss
from keyloom_mania.timing import BeatClock, Note, TimingSection
from keyloom_mania.tiny_data import SLOTS
from keyloom_mania.tiny_model import TinyModelConfig


class StyleSequenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = StyleSequenceGenerator(TinyModelConfig(width=12, heads=3, audio_layers=1, decoder_layers=1, feedforward=24))

    def inputs(self, length=192):
        return {"acoustic": torch.zeros(1, length, 128), "music": torch.zeros(1, length, 1024),
                "phase": torch.zeros(1, length, 5), "condition": torch.zeros(1, 5),
                "local": torch.zeros(1, length, LOCAL_WIDTH), "plan": torch.from_numpy(one_hot_plan(np.full((1, length), STYLE_NAMES.index("Handstream")))),
                "density": torch.ones(1, length, 2)}

    def test_rhythm_family_retains_triplet_and_fine_stream_grids(self):
        targets = np.zeros((4, 24), dtype=np.int64)
        targets[0, [SLOTS.index(tick) for tick in (0, 12, 24, 36)]] = 1
        targets[1, SLOTS.index(8)] = 1
        targets[2, SLOTS.index(6)] = 1
        targets[3, [SLOTS.index(6), SLOTS.index(8)]] = 1
        self.assertEqual(rhythm_targets(targets).flatten().tolist(), [0, 1, 2, 3])
        self.assertFalse(RHYTHM_GRIDS[0, SLOTS.index(8)])

    def test_count_history_is_causal_and_generation_accepts_partial_history(self):
        self.model.eval()
        memory = torch.randn(1, 192, 12)
        phase, condition = torch.randn(1, 192, 5), torch.zeros(1, 5)
        plan = torch.zeros(1, 192, 15)
        history = torch.zeros(1, 192, dtype=torch.long)
        first = self.model.decode_counts(memory, phase, condition, history, plan, True)
        history[:, 96:] = 3
        second = self.model.decode_counts(memory, phase, condition, history, plan, True)
        torch.testing.assert_close(first[:, :96], second[:, :96])
        self.assertEqual(self.model.decode_counts(memory, phase, condition, history[:, :2], plan, True).shape, (1, 2, 4))

    def test_joint_layout_does_not_read_teacher_current_count(self):
        self.model.eval()
        memory = torch.randn(1, 8, 12)
        phase = torch.randn(1, 8, 5)
        previous = torch.zeros(1, 8, dtype=torch.long)
        plan = torch.zeros(1, 8, 15)
        arguments = (memory, phase, torch.zeros(1, 5), previous, plan)
        _, first = self.model.decode_layout(*arguments, torch.ones_like(previous))
        _, second = self.model.decode_layout(*arguments, torch.full_like(previous, 3))
        torch.testing.assert_close(first, second)

    def test_joint_probabilities_and_loss_gradients_are_finite(self):
        inputs = self.inputs()
        targets = torch.zeros(1, 192, dtype=torch.long)
        positions = [beat * 24 + SLOTS.index(tick) for beat in range(8) for tick in (0, 12, 24, 36)]
        targets[0, positions] = torch.tensor([7, 8, 3, 12] * 8)
        previous = torch.cat((torch.full((1, 1), 16), targets[:, :-1]), dim=1)
        with patch.object(self.model, "encode_audio", return_value=torch.randn(1, 192, 12)):
            output = self.model(**inputs, previous=previous, current_counts=self.model.mask_counts[targets])
        joint = joint_mask_probability(output)
        torch.testing.assert_close(joint.sum(dim=-1), torch.ones(1, 192))
        losses = sequence_loss(output, targets, inputs["plan"], torch.ones_like(targets, dtype=torch.bool), torch.tensor(1.0))
        self.assertGreater(float(losses["sequence_motif_loss"].detach()), 0)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["loss"].backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in self.model.parameters()))

    def test_gap_only_never_rewards_repeats_and_style_limits_keep_jacks(self):
        plan = torch.from_numpy(one_hot_plan(np.asarray([STYLE_NAMES.index("Chordjacks")])))
        gap = LaneControlState(1, "cpu", LaneControlConfig(policy="gap-only"))
        style = LaneControlState(1, "cpu", LaneControlConfig(policy="style-aware", style_run_limits=(("Chordjacks", 8, 32, 8),)))
        for index in range(6):
            for state in (gap, style):
                state.advance(torch.tensor([3]), torch.tensor([index / 4], dtype=torch.float64))
        legal, penalty = gap.constraints(torch.tensor([1.5], dtype=torch.float64), plan)
        self.assertTrue(legal[0, 3])
        self.assertTrue((penalty == 0).all())
        self.assertTrue(style.constraints(torch.tensor([1.5], dtype=torch.float64), plan)[0][0, 3])
        self.assertFalse(style.constraints(torch.tensor([1.375], dtype=torch.float64), plan)[0][0, 3])

    def test_rhythm_evaluator_keeps_intrusive_notes_separate(self):
        clock = BeatClock([TimingSection(0, 480, 4)])
        notes = [Note(0, 0), Note(80, 1), Note(120, 2), Note(180, 3)]
        measured = rhythm_bar_metrics(clock, notes, [0], [1920])[0]
        json.dumps(measured)
        self.assertEqual(measured["off_sixteenth_rows"], 2)
        self.assertEqual(measured["row_gaps"]["twenty_fourth"], 1)
        self.assertEqual(measured["row_gaps"]["thirty_second"], 1)


    def test_density_conditions_softly_respect_regular_grid_capacity(self):
        inputs = self.inputs(24)
        density = torch.tensor([[[30.0, 10.0]]]).expand(1, 24, 2)
        projected = self.model.density_condition(density, inputs["plan"])
        self.assertTrue((projected[..., 1] < 5).all())
        self.assertTrue((projected[..., 0] < 11).all())

    def test_sequence_supervision_distinguishes_equal_histograms_with_bad_hs_order(self):
        positions = [beat * 24 + SLOTS.index(tick) for beat in range(8) for tick in (0, 12, 24, 36)]
        reference = torch.zeros(1, 192, dtype=torch.long)
        reference[0, positions] = torch.tensor([7, 8, 3, 12] * 8)
        plan = torch.from_numpy(one_hot_plan(np.full((1, 192), STYLE_NAMES.index("Handstream"))))

        def measured(masks):
            raw = torch.full((1, 192, 16), -20.0)
            raw.scatter_(-1, masks[..., None], 20.0)
            counts = self.model.mask_counts[masks]
            quantity = torch.full((1, 192, 4), -20.0)
            quantity.scatter_(-1, (counts - 1).clamp_min(0)[..., None], 20.0)
            output = {"onset_logits": torch.where(masks > 0, 20.0, -20.0), "conditional_count_logits": quantity,
                      "layout_logits": raw.masked_fill(self.model.mask_counts[None, None] != self.model.mask_counts[reference][..., None], -torch.inf),
                      "raw_layout_logits": raw, "rhythm_logits": torch.zeros(1, 8, 4)}
            return sequence_loss(output, reference, plan, torch.ones_like(reference, dtype=torch.bool), torch.tensor(1.0))

        good = measured(reference)
        bad = reference.clone()
        bad[0, positions] = torch.tensor([7, 3, 8, 12] * 8)
        bad_losses = measured(bad)
        self.assertAlmostEqual(float(good["bar_count_histogram_loss"]), float(bad_losses["bar_count_histogram_loss"]), places=5)
        self.assertGreater(float(bad_losses["sequence_motif_loss"]), float(good["sequence_motif_loss"]) + 5)


if __name__ == "__main__":
    unittest.main()
