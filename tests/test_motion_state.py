import unittest

import torch

from keyloom_mania.cadence_generator import CadenceGenerator
from keyloom_mania.motion_state import MOTION_WIDTH, MotionHistory, candidate_grid, motion_features
from keyloom_mania.natural_generator import NaturalLayoutGenerator
from keyloom_mania.playstyle_control import studio_playstyle_controls
from keyloom_mania.segment_labels import LABELS
from keyloom_mania.tiny_model import TinyModelConfig


class MotionFeaturesTests(unittest.TestCase):
    def test_hand_participation_matches_actual_past_and_has_no_future_input(self):
        previous = torch.tensor([[16, 1, 2, 1, 2, 1, 4, 8]])
        beats = torch.arange(8, dtype=torch.float64)[None] / 4
        plan = torch.zeros(1, 8, 15); plan[..., LABELS.index('Minijacks')] = 1
        feature = motion_features(previous, beats, plan, torch.full((1, 8), .5))
        self.assertEqual(feature.shape, (1, 8, MOTION_WIDTH))
        self.assertTrue(torch.isfinite(feature).all())
        self.assertEqual(float(feature[0, 5, 12]), 1.)
        self.assertEqual(float(feature[0, 5, 23]), 0.)
        self.assertEqual(float(feature[0, 6, 12]), 0.)
        changed = previous.clone(); changed[:, 6:] = 15
        alternate = motion_features(changed, beats, plan, torch.full((1, 8), .5))
        torch.testing.assert_close(feature[:, :6], alternate[:, :6])

    def test_seeded_window_matches_same_full_history_features(self):
        masks = torch.tensor([[1, 2, 3, 4, 8, 12, 9, 6, 1, 0, 2, 4, 8, 3, 12, 5]])
        beats = torch.arange(16, dtype=torch.float64)[None] / 4
        previous = torch.cat((torch.full((1, 1), 16), masks[:, :-1]), dim=1)
        plan = torch.zeros(1, 16, 15); plan[..., LABELS.index('Minijacks')] = 1
        full = motion_features(previous, beats, plan, torch.full((1, 16), .5))
        part_previous = torch.cat((torch.full((1, 1), 16), masks[:, 8:-1]), dim=1)
        part = motion_features(part_previous, beats[:, 8:], plan[:, 8:], torch.full((1, 8), .5), (masks[:, :8], beats[:, :8]))
        torch.testing.assert_close(part, full[:, 8:])

    def test_persisted_history_requires_original_overlap(self):
        state = MotionHistory()
        beats = torch.arange(8, dtype=torch.float64)[None] / 4
        masks = torch.tensor([[1, 2, 3, 4, 8, 12, 9, 6]])
        state.update(masks, beats, None)
        next_beats = torch.arange(6, 14, dtype=torch.float64)[None] / 4
        seed = state.before(next_beats, masks[:, -2:])
        self.assertEqual(seed[0].tolist(), masks[:, :6].tolist())
        with self.assertRaisesRegex(ValueError, 'replace'):
            state.before(next_beats, torch.zeros(1, 2, dtype=torch.long))


class NaturalGeneratorTests(unittest.TestCase):
    def test_zero_adapter_preserves_parent_raw_logits_and_only_adds_one_parameter(self):
        torch.manual_seed(41)
        config = TinyModelConfig(width=12, heads=3, audio_layers=1, decoder_layers=1, feedforward=24)
        parent = CadenceGenerator(config).eval()
        model = NaturalLayoutGenerator(config).eval()
        missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
        self.assertEqual(missing, ['motion_projection.weight']); self.assertEqual(unexpected, [])
        memory, phase = torch.randn(1, 8, 12), torch.randn(1, 8, 5)
        phase[..., 4] = .5
        previous = torch.tensor([[16, 3, 12, 0, 5, 10, 1, 2]])
        plan = torch.zeros(1, 8, 15)
        counts = torch.full((1, 8), 2)
        arguments = (memory, phase, torch.zeros(1, 5), previous, plan, counts)
        torch.testing.assert_close(model.decode_layout(*arguments)[1], parent.decode_layout(*arguments)[1])
        model.decode_layout(*arguments)[1].sum().backward()
        self.assertGreater(float(model.motion_projection.weight.grad.abs().sum()), 0)

    def test_generated_overlap_carries_motion_without_replaying_it(self):
        torch.set_num_threads(2)
        model = NaturalLayoutGenerator(TinyModelConfig(width=12, heads=3, audio_layers=1, decoder_layers=1, feedforward=24)).eval()
        inputs = {'acoustic': torch.zeros(1, 48, 128), 'music': torch.zeros(1, 48, 1024), 'phase': torch.zeros(1, 48, 5),
                  'local': torch.zeros(1, 48, 360), 'condition': torch.zeros(1, 5), 'density': torch.ones(1, 48, 2), 'plan': torch.zeros(1, 48, 15)}
        inputs['phase'][..., 4] = .5
        inputs['plan'][..., LABELS.index('Minijacks')] = 1
        lane, phrase = studio_playstyle_controls('cpu', {'Minijacks'})
        state = MotionHistory()
        first = model.generate(**inputs, candidate_beats=candidate_grid(1, 48, 'cpu'), lane_state=lane, phrase_state=phrase, motion_state=state)
        second = model.generate(**inputs, prefix_masks=first['masks'][:, -24:], candidate_beats=candidate_grid(1, 48, 'cpu') + 1,
                                lane_state=lane, phrase_state=phrase, motion_state=state)
        self.assertEqual(state.masks.shape[1], 72)
        torch.testing.assert_close(second['masks'][:, :24], first['masks'][:, -24:])
        self.assertIsNone(model._motion_seed)


if __name__ == '__main__':
    unittest.main()
