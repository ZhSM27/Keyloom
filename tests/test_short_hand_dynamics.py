import unittest
import numpy as np
import torch
from keyloom_mania.short_hand_dynamics import ordered_hand_context, numpy_signature, hand_allocation, prefix_distance
from keyloom_mania.hand_dynamics_generator import HandDynamicsGenerator
from keyloom_mania.expression_generator import ExpressionGenerator
from keyloom_mania.motion_state import candidate_grid
from keyloom_mania.tiny_model import TinyModelConfig


class FiniteHandDynamicsTest(unittest.TestCase):
    def context(self, masks, times, now=5.):
        return ordered_hand_context(torch.tensor([[16]]), torch.tensor([[now]], dtype=torch.float64),
            torch.tensor([[.4]]), (torch.tensor([masks]), torch.tensor([times], dtype=torch.float64)))

    def test_old_pressure_and_left_boundary_are_forgotten(self):
        a = self.context([3, 3, 3, 12], [0., 1., 2., 4.75])
        b = self.context([12, 0, 3, 12], [0., 1., 2., 4.75])
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_four_beat_rest_recovers_both_hands(self):
        bins, scalar = self.context([3, 12], [0., 1.])
        self.assertEqual(float(bins.sum()), 0.)
        torch.testing.assert_close(scalar[0, 0, :4], torch.tensor([1., 1., 0., 0.]))

    def test_order_is_not_reduced_to_totals(self):
        a, _ = self.context([3, 12], [2., 4.75])
        b, _ = self.context([12, 3], [2., 4.75])
        torch.testing.assert_close(a.sum(-2), b.sum(-2))
        self.assertFalse(torch.equal(a, b))

    def test_future_notes_do_not_change_past_features(self):
        times = candidate_grid(1, 96, 'cpu').double()
        previous = torch.zeros(1, 96, dtype=torch.long); previous[:, 0] = 16
        a = ordered_hand_context(previous, times, torch.full_like(times, .4))
        previous[:, 50:] = 3
        b = ordered_hand_context(previous, times, torch.full_like(times, .4))
        for first, second in zip(a, b):
            torch.testing.assert_close(first[:, :50], second[:, :50])

    def test_numpy_matching_and_model_see_identical_prefix(self):
        masks = [3, 5, 12, 9, 3]; times = [1., 1.125, 2., 4.75, 4.875]
        bins, scalar = self.context(masks, times)
        actual = np.r_[bins.flatten().numpy(), scalar.flatten().numpy()]
        np.testing.assert_allclose(actual, numpy_signature(masks, times, 5., 150), atol=1e-7)

    def test_matching_does_not_prefer_the_quieter_hand(self):
        left = numpy_signature([3, 3], [4.5, 4.75], 5., 150)
        right = numpy_signature([12, 12], [4.5, 4.75], 5., 150)
        self.assertEqual(prefix_distance(left, left), 0.)
        self.assertAlmostEqual(prefix_distance(left, right), prefix_distance(right, left))

    def test_joint_allocation_keeps_double_shapes_distinct(self):
        self.assertEqual(hand_allocation(np.array([3, 5, 12])).tolist(), [6, 4, 2])

    def test_seed_and_in_window_history_agree(self):
        masks = torch.tensor([[3, 0, 12, 5]])
        beats = torch.tensor([[4., 4.25, 4.5, 4.75]])
        previous = torch.tensor([[16, 3, 0, 12, 5]])
        full = ordered_hand_context(previous, torch.tensor([[4., 4.25, 4.5, 4.75, 5.]]), torch.full((1, 5), .4))
        seeded = ordered_hand_context(torch.tensor([[16]]), torch.tensor([[5.]]), torch.tensor([[.4]]), (masks, beats))
        for first, second in zip(full, seeded):
            torch.testing.assert_close(first[:, -1:], second)


class LayoutInitializationTest(unittest.TestCase):
    def test_zero_projection_preserves_baseline_and_source_bank(self):
        torch.manual_seed(7); torch.set_num_threads(2)
        cfg = TinyModelConfig(width=24, heads=3, audio_layers=1, decoder_layers=1, feedforward=48)
        base = ExpressionGenerator(cfg, reference_count_fallback=True, reference_layout_fallback=True).eval()
        model = HandDynamicsGenerator(cfg, reference_count_fallback=True, reference_layout_fallback=True).eval()
        missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(n.startswith(('short_encoder.', 'short_projection.', 'hand_allocation_head.', 'protected_layout_decoder.')) for n in missing))
        model.protected_layout_decoder.load_state_dict(base.decoder.state_dict())
        memory = torch.randn(1, 12, 24); phase = torch.randn(1, 12, 5); phase[..., 4] = .4
        condition = torch.randn(1, 5); previous = torch.tensor([[16, 3, 12, 5, 0, 3, 12, 6, 0, 3, 0, 12]])
        plan = torch.zeros(1, 12, 15); plan[..., 2] = 1; counts = torch.full((1, 12), 2)
        with torch.no_grad():
            a = base.decode_layout(memory, phase, condition, previous, plan, counts)[1]
            b = model.decode_layout(memory, phase, condition, previous, plan, counts)[1]
        torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-6)
        model._reference_pass = True; base._reference_pass = True
        with torch.no_grad():
            model.short_projection.bias.fill_(5)
            for parameter in model.decoder.layers[-1].parameters():
                parameter.add_(.01)
            a = base.decode_layout(memory, phase, condition, previous, plan, counts)[1]
            b = model.decode_layout(memory, phase, condition, previous, plan, counts)[1]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        base._reference_pass = model._reference_pass = False
        plan.zero_(); plan[..., 6] = 1  # Jumpstream, protected cut branch.
        with torch.no_grad():
            a = base.decode_layout(memory, phase, condition, previous, plan, counts)[1]
            b = model.decode_layout(memory, phase, condition, previous, plan, counts)[1]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
