import unittest

import numpy as np
import torch

from keyloom_mania.arrangement_model import ArrangementPlanner, ArrangementPlannerConfig
from keyloom_mania.section_density import SectionDensityAdapter, density_context, section_density_loss


class SectionDensityTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(42)
        base = torch.tensor([3., 1.5]).expand(2, 8, 2).clone()
        inputs = {"memory": torch.randn(2, 8, 16), "section_memory": torch.randn(2, 8, 16),
                  "global_memory": torch.randn(2, 8, 16), "base_density": base,
                  "beat_activity": torch.randn(2, 8, 4, 6), "quiet": torch.zeros(2, 8, dtype=torch.bool)}
        return SectionDensityAdapter(16), inputs

    def test_initialization_preserves_feasible_base_and_beat_mean(self):
        model, inputs = self.fixture()
        output = model(**inputs)
        torch.testing.assert_close(output["bar_density"], inputs["base_density"])
        torch.testing.assert_close(output["beat_density"].mean(-2), output["bar_density"])

    def test_quiet_and_row_note_feasibility_survive_extreme_updates(self):
        model, inputs = self.fixture()
        inputs["quiet"][0, 2] = True
        inputs["base_density"][1] = torch.tensor([.1, 10.])
        with torch.no_grad():
            model.local_head[-1].bias.copy_(torch.tensor([-10., 10.]))
        output = model(**inputs)
        self.assertEqual(output["bar_density"][0, 2].sum().item(), 0)
        beats = output["beat_density"]
        self.assertTrue((beats[..., 0] >= beats[..., 1] - 1e-5).all())
        self.assertTrue((beats[..., 0] <= 4 * beats[..., 1] + 1e-5).all())

    def test_section_loss_detects_flattened_contrast_and_backpropagates(self):
        model, inputs = self.fixture()
        truth = inputs["base_density"][:, :, None].expand(2, 8, 4, 2).clone()
        truth[:, :4] *= .5
        truth[:, 4:] *= 1.5
        batch = {"beat_targets": truth, "padding_mask": torch.zeros(2, 8, dtype=torch.bool),
                 "segment_ids": torch.tensor([[0] * 4 + [1] * 4] * 2)}
        output = model(**inputs)
        control = section_density_loss(output, batch, structured=False)
        candidate = section_density_loss(output, batch, structured=True)
        self.assertGreater(candidate["section_gap_loss"].item(), 0)
        self.assertGreater(candidate["loss"].item(), control["loss"].item())
        candidate["loss"].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.section_head[-1].weight.grad.abs().sum().item(), 0)

    def test_density_context_respects_noncontiguous_segment_ids_and_quiet(self):
        memory = np.arange(24, dtype=np.float32).reshape(6, 4)
        ids = np.asarray([0, 0, -1, 3, 3, 3])
        section, global_memory = density_context(memory, ids)
        np.testing.assert_allclose(section[0], memory[:2].mean(axis=0))
        np.testing.assert_array_equal(section[2], memory[2])
        np.testing.assert_allclose(section[4], memory[3:].mean(axis=0))
        np.testing.assert_allclose(global_memory, memory.mean(axis=0))

    def test_returning_frozen_memory_does_not_change_planner_predictions(self):
        torch.manual_seed(42)
        model = ArrangementPlanner(ArrangementPlannerConfig(width=16, heads=2, layers=1, feedforward=32, max_bars=8)).eval()
        inputs = [torch.randn(1, 6, width) for width in (1024, 128, 40, 7, 17)] + [torch.zeros(1, 6, dtype=torch.bool)]
        with torch.inference_mode():
            baseline, exposed = model(*inputs), model(*inputs, return_memory=True)
        for name in baseline:
            torch.testing.assert_close(baseline[name], exposed[name], rtol=0, atol=0)
        self.assertEqual(exposed["memory"].shape, (1, 6, 16))


if __name__ == "__main__":
    unittest.main()
