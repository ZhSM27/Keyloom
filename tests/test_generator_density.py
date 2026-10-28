from tests.helpers import portable_generator
from unittest.mock import patch
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from keyloom_mania.generator_density import (ExecutionLedger, execution_diagnostics,
    generator_capabilities, native_density_condition, prepare_density_budget, validate_budget)
from keyloom_mania.segment_labels import LABELS
from keyloom_mania.style_sequence import StyleSequenceGenerator
from keyloom_mania.tiny_model import TinyModelConfig


class GeneratorDensityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = StyleSequenceGenerator(TinyModelConfig(width=24, heads=4, audio_layers=1, decoder_layers=1, feedforward=48)).eval()
        cls.capabilities = generator_capabilities(cls.model, {'checkpoint_sha256': 'test-only', 'bpm': 120})

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_capability_uses_native_count_support_and_marks_calibration_unavailable(self):
        self.assertEqual(self.capabilities['count_head_support']['Stream'], [1])
        self.assertEqual(self.capabilities['count_head_support']['Handstream'], [1, 2, 3])
        self.assertEqual(self.capabilities['attainable_density_calibration'], 'unavailable')

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_baseline_is_bit_identical_and_inputs_are_immutable(self):
        d = np.array([[[5., 1.5]] * 4, [[0., 0.]] * 4], dtype=np.float32)
        before = d.copy()
        result, report = prepare_density_budget(d, np.array([1, 0]), [False, True], self.capabilities)
        np.testing.assert_array_equal(result, before)
        np.testing.assert_array_equal(d, before)
        self.assertEqual(report['changed_beats'], 0)
        self.assertEqual(report['conflicting_beats'], 4)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_research_preserves_notes_quiet_and_directional_style_ids(self):
        d = np.array([[[4., 2.]] * 4, [[4., 2.]] * 4, [[.1, .01]] * 4])
        styles = np.array([LABELS.index('Stream') + 1, LABELS.index('Handstream') + 1, 0])
        before = styles.copy()
        result, report = prepare_density_budget(d, styles, [False, False, True], self.capabilities, 'feasible')
        np.testing.assert_array_equal(result[:, :, 0], d[:, :, 0])
        np.testing.assert_array_equal(result[2], d[2])
        np.testing.assert_array_equal(styles, before)
        np.testing.assert_allclose(result[0, :, 1], 2.5)
        np.testing.assert_array_equal(result[1], d[1])
        self.assertEqual(report['changed_beats'], 4)
        self.assertEqual(report['unresolved_beats'], 4)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_budget_can_reduce_rows_without_reducing_notes_and_handles_integer_input(self):
        d = np.array([[[2, 4]] * 4])
        result, report = prepare_density_budget(d, np.array([1]), [False], self.capabilities, 'feasible')
        np.testing.assert_array_equal(result[0, :, 0], [2] * 4)
        np.testing.assert_array_equal(result[0, :, 1], [3] * 4)
        zero, _ = prepare_density_budget(np.array([[[1, 0]] * 4]), np.array([1]), [False], self.capabilities, 'feasible')
        self.assertEqual(zero[0, 0, 1], .25)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_unknown_or_density_unsupported_models_do_not_borrow_limits(self):
        capability = copy.deepcopy(self.capabilities)
        capability['accepts_density'] = False
        d = np.array([[[4., 2.]] * 4])
        for field in ('accepts_density', 'count_head_support'):
            candidate = copy.deepcopy(self.capabilities)
            candidate[field] = False if field == 'accepts_density' else {}
            result, _ = prepare_density_budget(d, np.array([1]), [False], candidate, 'feasible')
            np.testing.assert_array_equal(result, d)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_native_preprocessing_is_reported_separately_from_desired(self):
        d = np.array([[[9., 6.]] * 4], dtype=np.float32)
        pattern = np.eye(15, dtype=np.float32)[[LABELS.index('Jumpstream')]]
        original = d.copy()
        effective = native_density_condition(self.model, d, pattern)
        np.testing.assert_array_equal(d, original)
        # Native preprocessing retains 10% of the difference from its feasible reference.
        np.testing.assert_allclose(effective, [[[8.1, 4.2]] * 4], atol=1e-6)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_invalid_budget_and_density_are_rejected(self):
        for value in (None, [], False, 'unknown'):
            with self.assertRaises(ValueError):
                validate_budget(value)
        for d in (np.zeros((1, 2)), np.full((1, 4, 2), np.nan), -np.ones((1, 4, 2))):
            with self.assertRaises(ValueError):
                prepare_density_budget(d, np.array([1]), [False], self.capabilities)

    def diagnostic(self, ledger, exported, end=1216.3, start=216.3):
        d = np.ones((1, 4, 2))
        return execution_diagnostics(d, d, d, [{'start_ms': start, 'end_ms': end, 'style': 'Jumpstream', 'quiet': False}],
                                     250, ledger, exported, {})

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_counts_gross_additions_losses_and_chord_histograms_without_net_cancellation(self):
        ledger = ExecutionLedger()
        ledger.record(216, 3, 2, 1)  # one note removed in preparation, one in layout
        ledger.record(300, 1, 2, 3)  # preparation adds one: net zero does not hide losses
        result = self.diagnostic(ledger, [(216, 0), (300, 0)])
        summary = result['summary']
        self.assertEqual(summary['prepare_removed_notes'], 1)
        self.assertEqual(summary['prepare_added_notes'], 1)
        self.assertEqual(summary['layout_removed_notes'], 1)
        self.assertEqual(summary['export_removed_notes'], 1)
        beat = result['bars'][0]['beats'][0]
        self.assertEqual(beat['music_chord_histogram'], [1, 0, 1, 0])
        total_delta = np.sum(list(beat['signed_deltas'].values()), axis=0)
        np.testing.assert_allclose(total_delta, np.asarray(beat['stage_counts']['export']) - beat['desired'])

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_rounded_half_open_edges_partial_beats_and_no_prefix_double_counting(self):
        ledger = ExecutionLedger()
        for stamp in (216, 466, 716, 966):
            ledger.record(stamp, 1, 1, 1)
        result = self.diagnostic(ledger, [(t, 0) for t in ledger.events], end=850)
        self.assertEqual(result['summary']['full_beats'], 2)
        self.assertEqual(result['summary']['notes_beat_mae'], 0)
        self.assertEqual(result['summary']['unattributed_music_notes'], 1)
        self.assertFalse(result['bars'][0]['beats'][2]['complete'])
        with self.assertRaises(ValueError):
            ledger.record(466, 1, 1, 1)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_row_drop_is_visible_at_each_stage_and_no_empty_histogram_rows(self):
        ledger = ExecutionLedger()
        ledger.record(216, 1, 0, 0)
        ledger.record(466, 1, 1, 0)
        ledger.record(716, 1, 1, 1)
        ledger.record(966, 0, 0, 0)
        result = self.diagnostic(ledger, [])
        for name in ('prepare', 'layout', 'export'):
            self.assertEqual(result['summary'][name + '_dropped_rows'], 1)
        self.assertEqual(result['bars'][0]['beats'][3]['music_chord_histogram'], [0] * 4)
        json.dumps(result, allow_nan=False)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_budget_queue_validation_and_persistence(self):
        from keyloom_mania.generation_studio import StudioApp
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            source.write_bytes(b'fixture')
            app = StudioApp(root, 0)
            job = 'c' * 32
            app.jobs[job] = {'directory': str(root), 'source_path': str(source), 'filename': 'source.wav', 'status': 'ready'}
            with patch('keyloom_mania.generation_studio.threading.Thread') as worker:
                with self.assertRaises(ValueError):
                    app.generate(job, {'budget_id': 'unknown'})
                with self.assertRaisesRegex(ValueError, 'Stream'):
                    app.generate(job, {'budget_id': 'feasible'})
                worker.assert_not_called()
                app.generate(job, {'budget_id': 'legacy', 'main_style': 'Handstream'})
                worker.return_value.start.assert_called_once()
            self.assertEqual(json.loads((root / 'status.json').read_text(encoding='utf-8'))['settings']['budget_id'], 'legacy')


if __name__ == '__main__':
    unittest.main()
