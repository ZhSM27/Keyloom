import unittest

import torch

from keyloom_mania.phrase_control import PhraseControlState


class PhraseControlTests(unittest.TestCase):
    def test_periodic_layout_changes_after_grace_without_illegal_choice(self):
        state = PhraseControlState(1, "cpu")
        generated = []
        for index in range(48):
            preferred, alternative = (9, 3) if index % 2 == 0 else (4, 2)
            logits = torch.full((1, 16), -torch.inf)
            logits[0, preferred], logits[0, alternative] = 2, 0
            beats = torch.tensor([index / 4], dtype=torch.float64)
            mask = state.choose(logits, beats, True)
            state.advance(mask, beats, True)
            generated.append(int(mask))
        self.assertEqual(generated[:32], [9, 4] * 16)
        self.assertNotEqual(generated[32:34], [9, 4])
        self.assertGreater(state.summary()["changed_layout_rows"], 0)

    def test_constant_jack_is_preserved_and_replay_is_rejected(self):
        state = PhraseControlState(1, "cpu")
        for index in range(48):
            logits = torch.full((1, 16), -torch.inf)
            logits[0, 3] = 1
            beat = torch.tensor([index / 4], dtype=torch.float64)
            predicted = state.choose(logits, beat, True)
            state.advance(predicted, beat, True)
            self.assertEqual(int(predicted), 3)
        with self.assertRaises(ValueError):
            state.choose(logits, beat, True)


if __name__ == "__main__":
    unittest.main()
