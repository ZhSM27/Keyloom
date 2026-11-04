import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from keyloom_mania.arrangement_model import density_quantile_loss, planner_loss, section_contrast_loss
from keyloom_mania.difficulty_placement import difficulty_placement, mania_strains, strain_section_starts
from keyloom_mania.peak_planner import (
    load_calibration, section_local_stars, shape_peak_anchored_density, validate_star_condition, window_star,
)
from keyloom_mania.studio.export import write_full_osu
from keyloom_mania.timing import BeatClock, TimingSection
from tests.test_peak_planner import example, flat_density

BAR_MS = 2000.
BAR_STARTS = 1000 + np.arange(96) * BAR_MS


def chart(dense_bars):
    """Eighth notes everywhere, sixteenths in the dense bars."""
    notes = []
    for bar in range(96):
        step = 125 if bar in dense_bars else 250
        for index, time in enumerate(np.arange(BAR_STARTS[bar], BAR_STARTS[bar] + BAR_MS, step)):
            notes.append((int(time), (index * 3 + bar) % 4))
    directory = Path(tempfile.mkdtemp())
    path = directory / 'chart.osu'
    write_full_osu(path, {'metadata': {}}, notes, BeatClock([TimingSection(1000., 500., 4)]), 'audio.mp3', 'test')
    return path, [time for time, _ in notes]


class DifficultyPlacementTests(unittest.TestCase):
    def placement(self, dense_bars):
        path, times = chart(set(dense_bars))
        strains, length = mania_strains(path)
        return difficulty_placement(strains, length, times, BAR_STARTS, example(), np.full(96, 4.))

    def test_top_strains_follow_the_dense_section(self):
        chorus = self.placement(range(32, 64))
        self.assertEqual(chorus['peak_bar_share'], round(32 / 96, 4))
        self.assertGreaterEqual(chorus['top_strain_peak_share'], .9)
        self.assertAlmostEqual(chorus['lower_to_peak_exported_density_median'], .5)
        self.assertLess(chorus['lower_to_peak_strain_p90_median'], 1)
        # A chart whose verse carries the difficulty is reported as misallocated.
        verse = self.placement(range(0, 32))
        self.assertLessEqual(verse['top_strain_peak_share'], .1)
        self.assertGreater(verse['lower_to_peak_strain_p90_median'], 1)
        json.dumps(chorus, allow_nan=False)

    def test_excluded_style_bars_leave_every_metric(self):
        path, times = chart(set(range(0, 32)))
        strains, length = mania_strains(path)
        styles = ['Stream'] * 32 + ['Jumpstream'] * 64
        report = difficulty_placement(strains, length, times, BAR_STARTS, example(), np.full(96, 4.), bar_styles=styles)
        self.assertEqual(report['excluded_styles'], ['Stream'])
        self.assertEqual(report['excluded_bars'], 32)
        self.assertGreater(report['excluded_strain_sections'], 0)
        self.assertTrue(report['sections'][0]['excluded'])
        self.assertEqual(report['top_sections'], 20)
        self.assertTrue(all(item['bar'] > 32 for item in report['top_strains']))
        self.assertEqual(report['peak_bar_share'], .5)
        # Without styles nothing is excluded, so the dense verse still dominates.
        self.assertLessEqual(difficulty_placement(strains, length, times, BAR_STARTS, example(), np.full(96, 4.))['top_strain_peak_share'], .1)
        json.dumps(report, allow_nan=False)

    def test_a_fully_excluded_chart_reports_no_shares(self):
        path, times = chart(set())
        strains, length = mania_strains(path)
        report = difficulty_placement(strains, length, times, BAR_STARTS, example(), np.full(96, 4.), bar_styles=['Stream'] * 96)
        self.assertEqual(report['top_sections'], 0)
        self.assertIsNone(report['top_strain_peak_share'])
        self.assertIsNone(report['lower_to_peak_exported_density_median'])
        json.dumps(report, allow_nan=False)

    def test_strain_sections_align_to_absolute_multiples(self):
        starts = strain_section_starts([1000, 1125, 1250], 3, 400.)
        np.testing.assert_array_equal(starts, [800, 1200, 1600])
        self.assertEqual(len(strain_section_starts([1000], 3, 400.)), 0)

    def test_rejects_misaligned_bars(self):
        with self.assertRaises(ValueError):
            difficulty_placement([1., 2.], 400., [0, 1], BAR_STARTS[:10], example(), np.full(96, 4.))


class LocalStarTests(unittest.TestCase):
    def test_peak_keeps_target_and_lower_sections_follow_the_star_density_slope(self):
        _, shaping = shape_peak_anchored_density(flat_density(), example(), 6.)
        stars, report = section_local_stars(shaping, 96, 6.)
        slope = load_calibration()['star_from_peak_nps']['slope']
        np.testing.assert_array_equal(stars[32:64], 6.)
        multiplier = shaping['sections'][0]['multiplier']
        np.testing.assert_allclose(stars[:32], 6 + slope * np.log(multiplier), atol=1e-3)
        np.testing.assert_array_equal(stars[:32], stars[64:])
        self.assertEqual([s['tier'] for s in report['sections']], ['reduced', 'peak', 'reduced'])

    def test_drop_is_bounded_and_floored(self):
        density = flat_density()
        density[:32] *= .3
        density[64:] *= .3
        _, shaping = shape_peak_anchored_density(density, example(), 6.)
        stars, _ = section_local_stars(shaping, 96, 6., maximum_drop=1.)
        np.testing.assert_allclose(stars[:32], 5.)
        stars, _ = section_local_stars(shaping, 96, 2.5)
        self.assertTrue(np.all(stars >= 2.))

    def test_window_star_uses_only_new_slots(self):
        stars = np.array([4., 4., 6., 6.])
        beats = np.arange(6, 14, .25)
        self.assertAlmostEqual(window_star(stars, beats[8:]), 6.)
        self.assertAlmostEqual(window_star(stars, beats), 5.5)  # 8 slots of bar 1, 24 of bars 2-3
        with self.assertRaises(ValueError):
            window_star(stars, [])

    def test_option_requires_peak_anchored_structure(self):
        self.assertEqual(validate_star_condition('section', 'peak-anchored'), 'section')
        self.assertEqual(validate_star_condition('global', 'hierarchical'), 'global')
        for value, structure in (('section', 'hierarchical'), ('section', 'legacy'), ('local', 'peak-anchored')):
            with self.assertRaises(ValueError):
                validate_star_condition(value, structure)
        with self.assertRaises(ValueError):
            section_local_stars({'version': 'section-phrase-hierarchy-v1', 'sections': []}, 4, 5.)


def planner_batch(notes, section_ids=None):
    notes = torch.as_tensor(notes, dtype=torch.float32)[None]
    length = notes.shape[1]
    batch = {'padding_mask': torch.zeros(1, length, dtype=torch.bool), 'notes_per_beat': notes,
             'rows_per_beat': notes / 2, 'rows': (notes > 0).float(), 'chord': torch.full((1, length, 4), .25),
             'pattern_available': torch.ones(1, length, dtype=torch.bool), 'pattern': torch.zeros(1, length, 15),
             'acoustic': torch.randn(1, length, 128, generator=torch.Generator().manual_seed(0)),
             'rhythm': torch.randn(1, length, 40, generator=torch.Generator().manual_seed(1))}
    # Timing-section starts are always present in training data; they must not create sections.
    batch['is_section_start'] = torch.zeros(1, length, dtype=torch.bool)
    batch['is_section_start'][0, 0] = True
    if section_ids is not None:
        batch['section_ids'] = torch.as_tensor(section_ids, dtype=torch.long)[None]
    return batch


def planner_output(log_notes):
    log_notes = torch.as_tensor(log_notes, dtype=torch.float32)[None].requires_grad_()
    length = log_notes.shape[1]
    density = torch.stack((log_notes, log_notes - .3), dim=-1)
    return {'density_log': density, 'chord_logits': torch.zeros(1, length, 4), 'pattern_logits': torch.zeros(1, length, 15)}, log_notes


class PlannerContrastLossTests(unittest.TestCase):
    notes = np.r_[np.full(16, 2.), np.full(16, 6.), np.full(16, 2.)]

    def test_quantiles_penalize_flattened_density(self):
        batch = planner_batch(self.notes)
        exact, _ = planner_output(np.log1p(self.notes))
        flat, parameters = planner_output(np.full(48, np.log1p(self.notes).mean()))
        mask = ~batch['padding_mask']
        self.assertAlmostEqual(float(density_quantile_loss(exact, batch, mask)), 0., places=6)
        loss = density_quantile_loss(flat, batch, mask)
        self.assertGreater(float(loss), .05)
        loss.backward()
        self.assertTrue(torch.isfinite(parameters.grad).all() and parameters.grad.abs().sum() > 0)

    def test_section_rank_penalizes_reversed_contrast(self):
        # The reprise shares the verse's audio section id; bar 47 has no section.
        batch = planner_batch(self.notes, np.r_[np.zeros(16), np.ones(16), np.zeros(15), -1])
        mask = ~batch['padding_mask']
        right, _ = planner_output(np.log1p(self.notes))
        reversed_, _ = planner_output(np.log1p(8 - self.notes))
        gap, rank = section_contrast_loss(right, batch, mask)
        self.assertAlmostEqual(float(gap), 0., places=6)
        self.assertAlmostEqual(float(rank), 0., places=6)
        gap, rank = section_contrast_loss(reversed_, batch, mask)
        self.assertGreater(float(gap), 0)
        self.assertGreater(float(rank), 0)
        no_sections = planner_batch(self.notes)
        self.assertEqual([float(v) for v in section_contrast_loss(reversed_, no_sections, mask)], [0., 0.])

    def test_default_objective_is_unchanged(self):
        batch = planner_batch(self.notes, np.repeat([0, 1, 2], 16))
        output, _ = planner_output(np.full(48, 1.))
        base = planner_loss(output, batch, {'pattern_positive': 1., 'pattern_mean': 1.})
        zeroed = planner_loss(output, batch, {'pattern_positive': 1., 'pattern_mean': 1., 'density_quantile': 0., 'section_contrast': 0.})
        self.assertEqual(float(base['loss'].detach()), float(zeroed['loss'].detach()))
        enabled = planner_loss(output, batch, {'pattern_positive': 1., 'pattern_mean': 1., 'density_quantile': 1., 'section_contrast': 1.})
        expected = base['loss'] + enabled['density_quantile_loss'] + enabled['section_contrast_gap_loss'] + enabled['section_contrast_rank_loss']
        self.assertAlmostEqual(float(enabled['loss']), float(expected), places=5)


if __name__ == '__main__':
    unittest.main()
