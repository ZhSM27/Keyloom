import unittest

import torch

from keyloom_mania.playstyle_control import PlaystyleControlConfig, studio_playstyle_controls
from keyloom_mania.segment_labels import LABELS


def fourth_hit_legal(playstyle):
    lane, _ = studio_playstyle_controls("cpu", {"Chordjacks"}, playstyle=playstyle)
    plan = torch.zeros(1, 15)
    plan[0, LABELS.index("Chordjacks")] = 1
    for row in range(3):
        beat = torch.tensor([row / 4], dtype=torch.float64)
        lane.constraints(beat, plan)
        lane.advance(torch.tensor([0b0011]), beat)
    legal, penalty = lane.constraints(torch.tensor([.75], dtype=torch.float64), plan)
    return bool(legal[0, 0b0001]), float(penalty[0, 0b0001])


class JackCeilingTests(unittest.TestCase):
    def test_default_keeps_the_fourth_ordinary_hit(self):
        legal, penalty = fourth_hit_legal(PlaystyleControlConfig())
        self.assertTrue(legal)
        self.assertLessEqual(penalty, 0)          # no run penalty before the fifth hit

    def test_style_ceiling_stops_chordjack_chains_at_three(self):
        legal, _ = fourth_hit_legal(PlaystyleControlConfig(jack_style_ceilings=(("Chordjacks", 3, 0.),)))
        self.assertFalse(legal)
        legal, penalty = fourth_hit_legal(PlaystyleControlConfig(jack_style_ceilings=(("Chordjacks", 3, 1.),)))
        self.assertTrue(legal)                   # the drawn extension allows a fourth hit, now penalised
        self.assertGreater(penalty, 0)

    def test_invalid_ceilings_are_rejected(self):
        for value in (("Minijacks", 3, 0.), ("Chordjacks", 5, 0.), ("Chordjacks", 3, 1.5), ("Stream", 3, 0.)):
            with self.assertRaises(ValueError):
                PlaystyleControlConfig(jack_style_ceilings=(value,))


if __name__ == "__main__":
    unittest.main()
