from tests.helpers import portable_generator
from unittest.mock import patch
import copy
import unittest
from pathlib import Path

import numpy as np

from keyloom_mania.hierarchical_planner import section_phrase_structure
from keyloom_mania.peak_planner import (
    RANK_GRID, PeakPolicy, calibrate, load_calibration, ratio_curve, section_tiers,
    shape_peak_anchored_density, validate_calibration, validate_structure_id,
)


def example():
    # A quiet-energy verse, a loud chorus and a reprised verse.
    music = np.zeros((96, 3), np.float32)
    music[:, 0] = np.tile([0, 1, 2, 1, 0, -1, -2, -1], 12) * 2
    music[32:64, 1:] = 8
    acoustic = np.zeros((96, 4), np.float32)
    acoustic[32:64] = 6
    return section_phrase_structure(np.zeros(96, bool), music=music, acoustic=acoustic)


def flat_density(notes=5., rows=3.):
    density = np.empty((96, 4, 2), np.float32)
    density[..., 0], density[..., 1] = notes, rows
    return density


class PeakPlannerTests(unittest.TestCase):
    def test_packaged_calibration_is_a_decreasing_ratio_curve(self):
        calibration = load_calibration()
        self.assertGreaterEqual(len(calibration['buckets']), 3)
        for star in (2, 5, 9):
            curve = ratio_curve(calibration, star)
            self.assertEqual(curve.shape, (len(RANK_GRID),))
            self.assertTrue(np.all((curve > 0) & (curve <= 1)))
            self.assertTrue(np.all(np.diff(curve) <= 1e-9))

    def test_peak_kept_lower_sections_capped_never_raised(self):
        structure = example()
        density = flat_density()
        original = density.copy()
        shaped, report = shape_peak_anchored_density(density, structure, 6.)
        np.testing.assert_array_equal(density, original)
        tiers = {(s['start_bar'], s['end_bar']): s['tier'] for s in report['sections']}
        self.assertEqual(tiers[(32, 64)], 'peak')
        np.testing.assert_array_equal(shaped[32:64], density[32:64])
        self.assertLess(shaped[:32, :, 0].mean(), 5)
        self.assertLess(shaped[64:, :, 0].mean(), 5)
        # The reprise repeats the verse and must receive the same treatment.
        self.assertEqual(structure['sections'][0]['repeat_group'], structure['sections'][2]['repeat_group'])
        np.testing.assert_array_equal(shaped[:32], shaped[64:])
        self.assertTrue(np.all(shaped <= density + 1e-6))
        np.testing.assert_allclose(shaped[..., 0] / shaped[..., 1], 5 / 3, atol=1e-6)
        self.assertTrue(np.all(np.array(report['multipliers']) >= PeakPolicy().minimum_multiplier - 1e-9))
        self.assertLess(report['note_budget_change'], 0)
        self.assertEqual(report['peak_reference_notes_per_beat'], 5)

    def test_long_section_crossing_the_peak_fraction_counts_as_peak(self):
        sections = [{'start_bar': a, 'end_bar': b, 'energy_level': energy, 'repeat_group': group}
                    for group, (a, b, energy) in enumerate(((0, 8, -1.), (8, 16, 2.), (16, 85, 1.), (85, 93, -1.)))]
        tiers = section_tiers({'sections': sections}, np.full(93, 4.))
        self.assertEqual(tiers['peak'], {1, 2})

    def test_higher_star_allows_more_contrast(self):
        structure = example()
        low = shape_peak_anchored_density(flat_density(), structure, 2.5)[0]
        high = shape_peak_anchored_density(flat_density(), structure, 8.5)[0]
        np.testing.assert_array_equal(low[32:64], high[32:64])
        self.assertGreater(low[64:, :, 0].mean(), high[64:, :, 0].mean())

    def test_sections_already_below_ratio_are_untouched(self):
        structure = example()
        density = flat_density()
        density[:32] *= .3
        density[64:] *= .3
        shaped, report = shape_peak_anchored_density(density, structure, 6.)
        np.testing.assert_array_equal(shaped, density)
        self.assertEqual(report['changed_bars'], 0)
        self.assertEqual({s['tier'] for s in report['sections']}, {'peak', 'within_ratio'})

    def test_flat_audio_single_section_unchanged(self):
        structure = section_phrase_structure(np.zeros(32, bool), music=np.ones((32, 3)))
        density = np.full((32, 4, 2), 2., np.float32)
        shaped, report = shape_peak_anchored_density(density, structure, 7.)
        np.testing.assert_array_equal(shaped, density)
        self.assertEqual([s['tier'] for s in report['sections']], ['peak'])

    def test_quiet_and_zero_budgets_preserved(self):
        quiet = np.zeros(96, bool)
        quiet[30:34] = True
        music = np.zeros((96, 3), np.float32)
        music[:, 0] = np.tile([0, 1, 2, 1, 0, -1, -2, -1], 12) * 2
        music[34:64, 1:] = 8
        acoustic = np.zeros((96, 4), np.float32)
        acoustic[34:64] = 6
        structure = section_phrase_structure(quiet, music=music, acoustic=acoustic)
        density = flat_density()
        density[quiet] = .1
        density[5] = 0
        shaped, _ = shape_peak_anchored_density(density, structure, 6.)
        np.testing.assert_array_equal(shaped[quiet], density[quiet])
        np.testing.assert_array_equal(shaped[5], density[5])

    def test_rejects_invalid_inputs_and_mismatched_calibration(self):
        structure = example()
        with self.assertRaises(ValueError):
            shape_peak_anchored_density(np.full((95, 4, 2), 1., np.float32), structure, 5.)
        with self.assertRaises(ValueError):
            shape_peak_anchored_density(flat_density(), structure, float('nan'))
        with self.assertRaises(ValueError):
            shape_peak_anchored_density(flat_density(), structure, 5., PeakPolicy(minimum_multiplier=0))
        stale = copy.deepcopy(load_calibration())
        stale['version'] = 'peak-anchored-density-v0'
        with self.assertRaises(ValueError):
            validate_calibration(stale)
        rising = copy.deepcopy(load_calibration())
        rising['buckets'][0]['ratio_median'][0] = 1.5
        with self.assertRaises(ValueError):
            shape_peak_anchored_density(flat_density(), structure, 5., calibration=rising)
        self.assertEqual(validate_structure_id('peak-anchored'), 'peak-anchored')
        for value in (None, 'unknown'):
            with self.assertRaises(ValueError):
                validate_structure_id(value)

    def test_calibration_uses_density_sorted_blocks_per_star_bucket(self):
        def chart(star, densities, bpm=None):
            return {'star': star, 'bpm': bpm, 'blocks': [{'scalars': {'notes': 16 * value, 'active_bars': 16}} for value in densities]}
        # Peak 10 notes per bar; star follows 1.5 + 2 ln(notes per second) exactly (all 5-6 stars).
        tempos = np.linspace(150, 190, 30)
        profile = {'schema': 'human-chart-reference-v1', 'feature_schema': 'fixture', 'corpus_sha256': '0' * 64,
                   'filters': {'manifest_splits': ['train']},
                   'charts': [chart(1.5 + 2 * np.log(10 * bpm / 240), [10, 10, 10, 5, 5, 5, 5, 5, 5, 5], bpm) for bpm in tempos]
                   + [chart(None, [1, 2, 3, 4]), chart(3.5, [1, 2, 3])]}
        result = calibrate(profile)
        validate_calibration(result)
        self.assertEqual([bucket['charts'] for bucket in result['buckets']], [30])
        np.testing.assert_allclose(result['buckets'][0]['ratio_median'], [.5] * len(RANK_GRID))
        self.assertAlmostEqual(result['star_from_peak_nps']['slope'], 2, places=3)
        self.assertAlmostEqual(result["star_from_peak_nps"]["intercept"], 1.5, places=3)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_queue_accepts_peak_anchored_structure(self):
        import json
        import tempfile
        from keyloom_mania.generation_studio import StudioApp
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            source.write_bytes(b'fixture')
            app = StudioApp(root, 0)
            job_id = 'e' * 32
            app.jobs[job_id] = {'directory': str(root), 'source_path': str(source), 'filename': source.name, 'status': 'ready'}
            with patch('keyloom_mania.generation_studio.threading.Thread'):
                app.generate(job_id, {'main_style': 'Jumpstream', 'structure_id': 'peak-anchored'})
            self.assertEqual(json.loads((root / 'status.json').read_text(encoding='utf-8'))['settings']['structure_id'], 'peak-anchored')
            app.generation_lock.release()


if __name__ == '__main__':
    unittest.main()
