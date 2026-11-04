import unittest

import numpy as np

from keyloom_mania.tiny_data import SLOTS
from keyloom_mania.accent_features import (ACCENT_LOCAL_WIDTH, ACCENT_WIDTH, CHANNELS, OUTPUT_HOP_MS, SAMPLE_RATE,
                                          accent_frames, sample_accents)


def at(seconds):
    return int(seconds * SAMPLE_RATE)


def fixture():
    """Kick at 1 s, hi-hat at 2 s, a sustained 440 Hz tone from 3 s and a pitch change to 660 Hz at 4 s."""
    rng = np.random.default_rng(1)
    audio = np.zeros(at(6), dtype=np.float32) + 1e-4 * rng.standard_normal(at(6)).astype(np.float32)
    time = np.arange(2880) / SAMPLE_RATE
    audio[at(1):at(1) + 2880] += np.sin(2 * np.pi * 55 * time) * np.exp(-time * 30)
    noise = rng.standard_normal(720).astype(np.float32)
    audio[at(2):at(2) + 720] += .5 * np.diff(noise, prepend=0) * np.exp(-np.arange(720) / SAMPLE_RATE * 120)
    tone = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    audio[at(3):at(3) + SAMPLE_RATE] += .3 * np.sin(2 * np.pi * 440 * tone)
    audio[at(4):at(4) + SAMPLE_RATE] += .3 * np.sin(2 * np.pi * 660 * tone)
    return audio


SLOT_BEATS = [slot / 48 for slot in SLOTS]


class AccentFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = accent_frames(fixture())

    def peak(self, channel, seconds):
        column = CHANNELS.index(channel)
        center = int(seconds * 1000 / OUTPUT_HOP_MS)
        return float(self.frames[center - 6:center + 6, column].max())

    def test_shape_and_finite(self):
        self.assertEqual(self.frames.shape[1], ACCENT_WIDTH)
        self.assertTrue(np.isfinite(self.frames).all())
        self.assertAlmostEqual(len(self.frames) * OUTPUT_HOP_MS / 1000, 6, delta=.05)

    def test_percussive_bands_separate_kick_and_hat(self):
        self.assertGreater(self.peak("kick_flux", 1), self.peak("kick_flux", 2))
        self.assertGreater(self.peak("hat_flux", 2), self.peak("hat_flux", 1))

    def test_tone_onsets_and_pitch_changes_are_harmonic(self):
        self.assertGreater(self.peak("harmonic_flux", 3), self.peak("harmonic_flux", 2))
        # The pitch change keeps loudness, so pitch flux rises where the hat does not.
        self.assertGreater(self.peak("pitch_flux", 4), self.peak("pitch_flux", 2))

    def test_sampling_follows_local_offsets(self):
        sampled = sample_accents(self.frames, np.asarray([1000., 2000., 5900.]))
        self.assertEqual(sampled.shape, (3, ACCENT_LOCAL_WIDTH))
        self.assertTrue(np.isfinite(sampled).all())


class AccentCountTests(unittest.TestCase):
    def test_untrained_accent_model_decodes_like_its_parent(self):
        import torch
        from keyloom_mania.accent_count import AccentCountGenerator
        from keyloom_mania.music_count_natural import MusicCountNaturalGenerator
        from keyloom_mania.tiny_model import TinyModelConfig

        torch.manual_seed(0)
        config = TinyModelConfig(width=32, heads=2, audio_layers=1, decoder_layers=1, feedforward=64)
        parent = MusicCountNaturalGenerator(config).eval()
        child = AccentCountGenerator(config).eval()
        missing, unexpected = child.load_state_dict(parent.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(name.startswith(("accent_count_adapter.", "support_head.")) for name in missing))
        length = 48
        memory = torch.randn(1, length, 32)
        phase = torch.randn(1, length, 5)
        condition = torch.zeros(1, 5)
        previous = torch.randint(0, 5, (1, length))
        plan = torch.zeros(1, length, 15)
        plan[..., 14] = 1
        density = torch.ones(1, length, 2)
        for model in (parent, child):
            model._count_training_density = density
        child._accent = torch.randn(1, length, ACCENT_LOCAL_WIDTH)
        with torch.no_grad():
            expected = parent.decode_counts(memory, phase, condition, previous, plan, True)
            actual = child.decode_counts(memory, phase, condition, previous, plan, True)
            support = child.support_logits(memory, phase, plan)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(support, torch.zeros_like(support)))

    def test_stream_rows_need_constant_linked_gaps_for_one_beat(self):
        import torch
        from keyloom_mania.accent_count import stream_rows

        decided = torch.zeros(1, 192, dtype=torch.bool)
        decided[0, [0, 6, 12, 18, 24, 30, 36, 42]] = True      # sixteenths over two beats
        decided[0, [96, 102, 108]] = True                       # three sixteenths: half a beat only
        decided[0, [144, 168]] = True                           # isolated quarter-beat rows
        stream = stream_rows(decided)[0]
        self.assertTrue(stream[[0, 6, 12, 18, 24, 30, 36, 42]].all())
        self.assertFalse(stream[[96, 102, 108, 144, 168]].any())

    def test_support_decoding_drops_unsupported_rows_outside_streams_and_counts_them(self):
        from unittest import mock

        import torch
        from keyloom_mania.accent_count import AccentCountGenerator
        from keyloom_mania.music_count_natural import MusicCountNaturalGenerator
        from keyloom_mania.tiny_model import TinyModelConfig

        config = TinyModelConfig(width=32, heads=2, audio_layers=1, decoder_layers=1, feedforward=64)
        model = AccentCountGenerator(config).eval()
        logits = torch.full((1, 192), -5.)
        logits[0, [0, 6, 12, 18, 24, 30]] = 5.     # stream rows, all weakly supported
        logits[0, [72, 120, 168]] = 5.             # off-stream rows
        support = torch.ones(1, 192)
        support[0, [0, 6, 12, 18, 24, 30, 120, 168]] = .01
        plan = torch.zeros(1, 192, 15)
        plan[..., 0] = 1
        model.support_drop_thresholds = torch.full((15,), .1)
        model._support, model._prefix_slots = support, 144
        model._support_stats = {"decided_rows": 0, "stream_protected_rows": 0, "dropped_rows": 0, "dropped_slots": [],
                                "demoted_rows": 0, "demoted_notes": 0}
        with mock.patch.object(MusicCountNaturalGenerator, "generation_onset_logits", return_value=logits.clone()):
            decoded = model.generation_onset_logits(None, None, plan)
        dropped = torch.nonzero(decoded[0] < 0).flatten().tolist()
        self.assertIn(168, dropped)                  # unsupported, off-stream, after the prefix
        self.assertNotIn(120, dropped)               # inside the continuation prefix: never changed
        self.assertNotIn(72, dropped)                # supported
        self.assertFalse(set(dropped) & {0, 6, 12, 18, 24, 30})
        self.assertEqual(model._support_stats["dropped_slots"], [168])
        self.assertEqual(model._support_stats["decided_rows"], 1)

    def test_chordjack_rows_are_kept_and_sized_by_the_human_position_prior(self):
        from unittest import mock

        import torch
        from keyloom_mania.accent_count import AccentCountGenerator
        from keyloom_mania.music_count_natural import MusicCountNaturalGenerator
        from keyloom_mania.segment_labels import LABELS
        from keyloom_mania.tiny_model import TinyModelConfig

        config = TinyModelConfig(width=32, heads=2, audio_layers=1, decoder_layers=1, feedforward=64)
        model = AccentCountGenerator(config).eval()
        onset = torch.full((1, 192), -5.)
        onset[0, [0, 6, 72]] = 5.                  # beat 0, its quarter (tick 12), beat 3
        support = torch.full((1, 192), .99)
        support[0, 6] = .1
        plan = torch.zeros(1, 192, 15)
        plan[..., LABELS.index("Chordjacks")] = 1
        # Human prior: large chords on supported beats, jumps on weak quarter beats.
        log_prior = torch.full((4, 2, 4), -3.)
        log_prior[0, 1, 3] = 0.                     # on the beat, strong support: four notes
        log_prior[2, 0, 1] = 0.                     # quarter beat, weak support: two notes
        model._accent = torch.zeros(1, 192, ACCENT_LOCAL_WIDTH)
        model.support_drop_thresholds = torch.full((15,), .5)
        model.support_chord_thresholds = torch.full((15,), .95)
        model.support_size_styles = ("Chordjacks",)
        model.size_prior = {"schema": "test", "support_edges": [.5], "log_prior": log_prior.tolist()}
        model.size_prior_weight = 1.
        model._support, model._prefix_slots = support, 0
        model._support_stats = {"decided_rows": 0, "stream_protected_rows": 0, "dropped_rows": 0, "dropped_slots": [],
                                **{f"{name}_{unit}": 0 for name in ("demoted", "size_lowered", "size_raised") for unit in ("rows", "notes")}}
        with mock.patch.object(MusicCountNaturalGenerator, "generation_onset_logits", return_value=onset.clone()):
            decoded = model.generation_onset_logits(None, None, plan)
        self.assertTrue((decoded[0, [0, 6, 72]] > 0).all())        # weak Chordjacks rows are never dropped
        model._reference_anchor = torch.zeros(1)
        base = torch.tensor([0., 0., 1., 0.])                       # the free-running model prefers triads everywhere
        for position, expected in ((0, 3), (6, 1), (72, 3)):
            logits = base.expand(1, position + 1, 4).clone()
            with mock.patch.object(MusicCountNaturalGenerator, "generation_count_logits", return_value=logits):
                shaped = model.generation_count_logits(None, None, None, torch.zeros(1, position + 1, dtype=torch.long), None, plan, True)
            self.assertEqual(int(shaped[0, -1].argmax()), expected)
        stats = model._support_stats
        self.assertEqual((stats["size_raised_rows"], stats["size_raised_notes"]), (2, 2))     # two beats: triad to quad
        self.assertEqual((stats["size_lowered_rows"], stats["size_lowered_notes"]), (1, -1))  # weak quarter: triad to jump
        self.assertEqual(stats["demoted_rows"], 0)                                          # Chordjacks never demoted


class DecodingPolicyTests(unittest.TestCase):
    def test_checkpoint_policy_configures_decoding_and_absence_keeps_it_off(self):
        from keyloom_mania.accent_count import DECODING_SCHEMA, AccentCountGenerator
        from keyloom_mania.playstyle_control import PlaystyleControlConfig
        from keyloom_mania.segment_labels import LABELS
        from keyloom_mania.tiny_model import TinyModelConfig

        config = TinyModelConfig(width=32, heads=2, audio_layers=1, decoder_layers=1, feedforward=64)
        model = AccentCountGenerator(config)
        model.configure_decoding(None)
        self.assertIsNone(model.support_drop_thresholds)
        self.assertEqual(model.motif_strength, 0.)
        policy = {"schema": DECODING_SCHEMA, "residual_scale": 0., "drop_thresholds": {label: .1 for label in LABELS},
                  "chord_thresholds": {label: .9 for label in LABELS}, "size_styles": ["Chordjacks"],
                  "size_prior": {"support_edges": [.5], "log_prior": []}, "size_prior_weight": 3., "size_quad_bonus": 1.5,
                  "layout_mask_bias": {"Jumpstream": [0.] * 16}, "motif_styles": ["Jumpstream"], "motif_strength": 3.,
                  "jack_style_ceilings": [["Chordjacks", 3, .25], ["Gluts", 3, 0.]]}
        model.configure_decoding(policy)
        self.assertEqual(model.accent_residual_scale, 0.)
        self.assertEqual(model.support_size_styles, ("Chordjacks",))
        self.assertAlmostEqual(float(model.support_chord_thresholds[0]), .9)
        self.assertEqual(model.motif_styles, ("Jumpstream",))
        # What the pipeline hands to the lane controls must be a valid configuration.
        PlaystyleControlConfig(jack_style_ceilings=model.jack_style_ceilings)
        with self.assertRaises(ValueError):
            model.configure_decoding({**policy, "schema": "unknown"})


class MotifReuseTests(unittest.TestCase):
    def build(self):
        import torch
        from keyloom_mania.accent_count import AccentCountGenerator
        from keyloom_mania.segment_labels import LABELS
        from keyloom_mania.tiny_model import TinyModelConfig

        config = TinyModelConfig(width=32, heads=2, audio_layers=1, decoder_layers=1, feedforward=64)
        model = AccentCountGenerator(config).eval()
        model.motif_styles, model.motif_strength = ("Jumpstream",), 2.
        model._support_stats = {"motif_scored_rows": 0, "motif_matched_rows": 0}
        length = 192
        beats = (torch.arange(8, dtype=torch.float64)[:, None] + torch.tensor(SLOT_BEATS, dtype=torch.float64)[None]).reshape(1, -1) + 8    # window starts at bar 2
        plan = torch.zeros(1, length, 15)
        plan[..., LABELS.index("Jumpstream")] = 1
        onset = torch.zeros(1, length)
        onset[0, ::6] = 1.                                                    # sixteenths
        model._layout_previous = torch.full((1, 1), 16)
        return model, beats, plan, onset, torch.ones(1, length, 32)

    def scores(self, model, position, beats, plan, onset, memory):
        from unittest import mock

        import torch
        from keyloom_mania.music_count_natural import MusicCountNaturalGenerator

        logits = torch.zeros(1, 16)
        logits[0, 0] = -torch.inf
        with mock.patch.object(MusicCountNaturalGenerator, "generation_layout_scores", return_value=logits):
            return model.generation_layout_scores(logits, memory, None, None, None, torch.ones(1, dtype=torch.long), plan, position,
                                                  beats, None, onset)

    def test_layouts_overlapping_the_matching_earlier_bar_score_higher(self):
        import torch
        model, beats, plan, onset, memory = self.build()
        for bar in (0, 1):
            model._motif_rows[bar] = {tick: (0b0011 if bar == 1 else 0b1100) for tick in range(0, 192, 12)}
            model._motif_bars[bar] = (int(plan[0, 0].argmax()), memory[0, 0] / memory[0, 0].norm())
        scored = self.scores(model, 0, beats, plan, onset, memory)
        self.assertEqual(model._motif_choice[2], 1)                # nearest of two equally good bars
        self.assertGreater(float(scored[0, 0b0011]), float(scored[0, 0b1100]))
        self.assertGreater(float(scored[0, 0b0011]), float(scored[0, 0b0001]))
        self.assertTrue(torch.isneginf(scored[0, 0]))              # illegal stays illegal
        self.assertEqual(model._support_stats["motif_scored_rows"], 1)

    def test_layout_bias_applies_to_its_style_and_keeps_illegal_layouts(self):
        import torch
        model, beats, plan, onset, memory = self.build()
        model.motif_strength = 0.
        model.layout_mask_bias = {"Jumpstream": [0.] * 3 + [1.] + [0.] * 12, "Stream": [5.] * 16}
        scored = self.scores(model, 0, beats, plan, onset, memory)
        self.assertEqual(float(scored[0, 3]), 1.)
        self.assertEqual(float(scored[0, 5]), 0.)
        self.assertTrue(torch.isneginf(scored[0, 0]))

    def test_no_reference_without_a_matching_rhythm_or_style(self):
        import torch
        model, beats, plan, onset, memory = self.build()
        model._motif_rows[1] = {tick: 0b0011 for tick in range(0, 192, 48)}   # quarter notes only
        model._motif_bars[1] = (int(plan[0, 0].argmax()), memory[0, 0] / memory[0, 0].norm())
        model._motif_rows[0] = {tick: 0b0011 for tick in range(0, 192, 12)}
        model._motif_bars[0] = (0, memory[0, 0] / memory[0, 0].norm())        # another style
        scored = self.scores(model, 0, beats, plan, onset, memory)
        self.assertIsNone(model._motif_choice[2])
        self.assertTrue(torch.equal(scored[0, 1:], torch.zeros(15)))


if __name__ == "__main__":
    unittest.main()
