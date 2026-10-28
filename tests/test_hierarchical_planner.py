from tests.helpers import portable_generator
from unittest.mock import patch
import copy
import importlib.util
import unittest
from pathlib import Path

import numpy as np

from keyloom_mania.hierarchical_planner import (
    section_phrase_structure, shape_hierarchical_density, validate_hierarchy,
    validate_structure_id,
)
from keyloom_mania.role_style_plan import active_style_plan
from keyloom_mania.segment_labels import LABELS
from keyloom_mania.style_hierarchy import compatible_styles


def example():
    # Three long sections; the first returns after a contrasting middle.
    music = np.zeros((96, 3), np.float32)
    music[:, 0] = np.tile([0, 1, 2, 1, 0, -1, -2, -1], 12) * 2
    music[32:64, 1:] = 8
    acoustic = np.zeros((96, 4), np.float32)
    acoustic[32:64] = 6
    return section_phrase_structure(np.zeros(96, bool), music=music, acoustic=acoustic)


class HierarchicalPlannerTests(unittest.TestCase):
    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_sections_contain_multiple_phrases_and_reprise(self):
        result = example()
        validate_hierarchy(result, result['quiet'])
        self.assertEqual([(s['start_bar'], s['end_bar']) for s in result['sections']],
                         [(0, 32), (32, 64), (64, 96)])
        self.assertTrue(all(len(s['phrase_ids']) > 1 for s in result['sections']))
        self.assertEqual(result['sections'][0]['repeat_group'], result['sections'][2]['repeat_group'])
        self.assertNotEqual(result['sections'][0]['repeat_group'], result['sections'][1]['repeat_group'])

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_flat_audio_does_not_invent_regular_phrases_or_energy(self):
        result = section_phrase_structure(np.zeros(32, bool), music=np.ones((32, 3)))
        self.assertEqual(len(result['sections']), 1)
        self.assertEqual(len(result['segments']), 1)
        self.assertEqual(result['segments'][0]['function_cue'], 'neutral')
        density = np.full((32, 4, 2), 2., np.float32)
        shaped, _ = shape_hierarchical_density(density, result)
        np.testing.assert_array_equal(shaped, density)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_timing_and_silence_are_hard_boundaries(self):
        quiet = np.zeros(32, bool)
        quiet[7:10] = True
        timing = np.zeros(32, int)
        timing[19:] = 1
        result = section_phrase_structure(quiet, music=np.ones((32, 3)), timing_sections=timing)
        validate_hierarchy(result, quiet)
        self.assertEqual([(s['start_bar'], s['end_bar']) for s in result['sections']], [(0, 7), (10, 19), (19, 32)])
        self.assertTrue((result['phrase_ids'][quiet] == -1).all())

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_build_and_release_cues_follow_audio_energy(self):
        from keyloom_mania.hierarchical_planner import HierarchyPolicy
        for values, expected in ((np.arange(8), 'build'), (np.arange(8)[::-1], 'release')):
            result = section_phrase_structure(np.zeros(8, bool), acoustic=values[:, None],
                policy=HierarchyPolicy(phrase_minimum_bars=8))
            self.assertEqual(result['segments'][0]['function_cue'], expected)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_rejects_missing_nonfinite_or_misaligned_evidence(self):
        for kwargs in ({}, {'music': np.ones((3, 2))}, {'music': np.full((4, 2), np.nan)}):
            with self.assertRaises(ValueError):
                section_phrase_structure(np.zeros(4, bool), **kwargs)
        for value in (None, [], 'unknown'):
            with self.assertRaises(ValueError):
                validate_structure_id(value)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_crossing_parent_and_frozen_silence_rejected(self):
        result = example()
        result['segments'][0]['section_id'] = 1
        with self.assertRaises(ValueError):
            validate_hierarchy(result, result['quiet'])
        with self.assertRaises(ValueError):
            validate_hierarchy(example(), np.ones(96, bool))
        result = example()
        result['sections'][0]['phrase_ids'] = []
        with self.assertRaises(ValueError):
            validate_hierarchy(result, result['quiet'])

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_density_conserves_notes_and_chord_ratio_with_bounded_contrast(self):
        result = example()
        density = np.empty((96, 4, 2), np.float32)
        density[..., 0], density[..., 1] = 5, 3
        original = density.copy()
        shaped, report = shape_hierarchical_density(density, result)
        np.testing.assert_array_equal(density, original)
        self.assertAlmostEqual(float(shaped[..., 0].sum()), float(density[..., 0].sum()), places=3)
        np.testing.assert_allclose(shaped[..., 0] / shaped[..., 1], 5 / 3, atol=1e-6)
        self.assertTrue(np.all(np.abs(np.array(report['multipliers']) - 1) <= .120001))
        self.assertGreater(shaped[32:64, :, 0].mean(), shaped[:32, :, 0].mean())

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_zero_budgets_and_physical_capacity(self):
        result = example()
        density = np.full((96, 4, 2), 2., np.float32)
        density[0] = 0
        density[32:64, :, 0], density[32:64, :, 1] = 96, 24
        shaped, _ = shape_hierarchical_density(density, result)
        np.testing.assert_array_equal(shaped[0], density[0])
        self.assertLessEqual(shaped[..., 0].max(), 96)
        self.assertLessEqual(shaped[..., 1].max(), 24)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_all_quiet_and_nonzero_quiet_targets_are_preserved(self):
        for quiet in (np.ones(8, bool), np.array([True] * 2 + [False] * 6)):
            h = section_phrase_structure(quiet, acoustic=np.arange(8)[:, None])
            density = np.full((8, 4, 2), .1, np.float32)
            shaped, _ = shape_hierarchical_density(density, h)
            np.testing.assert_array_equal(shaped[quiet], density[quiet])
            probabilities = np.full((8, len(LABELS)), .8)
            result = active_style_plan(probabilities, np.where(quiet, .1, 4.), np.where(quiet, .1, 2.),
                np.full(len(LABELS), .5), {}, 'Jumpstream', [], structure_override=h)
            self.assertTrue((result['style_ids'][quiet] == 0).all())

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_role_constraints_hold_for_every_primary(self):
        hierarchy = example()
        frozen = copy.deepcopy(hierarchy)
        evidence = np.full((96, len(LABELS)), .8, np.float32)
        for primary in LABELS:
            result = active_style_plan(evidence, np.full(96, 4.), np.full(96, 2.),
                np.full(len(LABELS), .5), {}, primary, [], structure_override=hierarchy)
            chosen = {LABELS[i - 1] for i in result['style_ids'] if i}
            self.assertTrue(chosen <= set(compatible_styles(primary)))
            self.assertTrue(all('style_intent' in s for s in result['sections']))
        self.assertEqual(hierarchy['sections'], frozen['sections'])

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_explicit_upper_secondary_and_order_invariance(self):
        h = example()
        evidence = np.full((96, len(LABELS)), .02, np.float32)
        evidence[:, LABELS.index('Jumpstream')] = .9
        evidence[32:64, LABELS.index('Jumpstream')] = .05
        evidence[32:64, LABELS.index('Handstream')] = .95
        def run(secondary):
            return active_style_plan(evidence, np.full(96, 4.), np.full(96, 2.),
                np.full(len(LABELS), .5), {}, 'Jumpstream', secondary, structure_override=h)
        a, b = run(['Handstream', 'Stream']), run(['Stream', 'Handstream'])
        np.testing.assert_array_equal(a['style_ids'], b['style_ids'])
        self.assertIn(LABELS.index('Handstream') + 1, a['style_ids'])

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_queue_validates_and_persists_structure(self):
        import json
        import tempfile
        from unittest.mock import patch
        from keyloom_mania.generation_studio import StudioApp
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            source.write_bytes(b'fixture')
            app = StudioApp(root, 0)
            job_id = 'd' * 32
            for mode in ('legacy', 'hierarchical', 'unknown'):
                app.jobs[job_id] = {'directory': str(root), 'source_path': str(source), 'filename': source.name, 'status': 'ready'}
                with patch('keyloom_mania.generation_studio.threading.Thread') as worker:
                    if mode == 'unknown':
                        with self.assertRaises(ValueError):
                            app.generate(job_id, {'main_style': 'Jumpstream', 'structure_id': mode})
                        worker.assert_not_called()
                    else:
                        app.generate(job_id, {'main_style': 'Jumpstream', 'structure_id': mode})
                        self.assertEqual(json.loads((root / 'status.json').read_text(encoding='utf-8'))['settings']['structure_id'], mode)
                        app.generation_lock.release()



if __name__ == '__main__':
    unittest.main()
