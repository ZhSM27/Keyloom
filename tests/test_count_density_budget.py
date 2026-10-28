from tests.helpers import portable_generator
from unittest.mock import patch
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from keyloom_mania.count_density_budget import enable_count_density_reference, bounded_count_distribution
from keyloom_mania.expression_generator import ExpressionGenerator
from keyloom_mania.tiny_model import TinyModelConfig
from keyloom_mania.segment_labels import LABELS
from keyloom_mania.local_rhythm import LOCAL_WIDTH
from keyloom_mania.playstyle_control import studio_playstyle_controls


class CountDensityReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(81)
        self.model = ExpressionGenerator(TinyModelConfig(width=24, heads=4, audio_layers=1, decoder_layers=1, feedforward=48)).eval()
        self.model.count_policy = 'music-density-sample'

    def batch(self, style):
        length = 192
        plan = torch.zeros(1, length, 15)
        plan[..., LABELS.index(style)] = 1
        return {'acoustic': torch.randn(1, length, 128), 'music': torch.randn(1, length, 1024),
            'local': torch.randn(1, length, LOCAL_WIDTH), 'phase': torch.zeros(1, length, 5),
            'condition': torch.zeros(1, 5), 'plan': plan,
            'density': torch.tensor([4., 2.]).expand(1, length, 2).clone(),
            'candidate_beats': torch.arange(length)[None] / 24,
            'prefix_masks': torch.empty(1, 0, dtype=torch.long), 'fast_empty': True}

    def run_model(self, model, batch, **extra):
        styles = {LABELS[i] for i in batch['plan'].amax((0, 1)).nonzero().flatten().tolist()}
        lane, phrase = studio_playstyle_controls('cpu', styles)
        return model.generate(**batch, lane_state=lane, phrase_state=phrase, **extra)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_mix_zero_is_bit_identical_to_native_generation(self):
        model = copy.deepcopy(self.model)
        enable_count_density_reference(model, 0)
        batch = self.batch('Jumpstream')
        with torch.inference_mode():
            old, new = self.run_model(self.model, batch), self.run_model(model, batch)
        for key in ('masks', 'music_counts', 'onset_probability'):
            torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_non_js_hs_types_and_neural_inputs_remain_identical(self):
        for style in ('Stream', 'Minijacks', 'Handstream'):
            model = copy.deepcopy(self.model)
            enable_count_density_reference(model)
            batch = self.batch(style)
            density = batch['density'].clone()
            with torch.inference_mode():
                old, new = self.run_model(self.model, batch), self.run_model(model, batch)
            for key in ('masks', 'music_counts', 'onset_probability'):
                torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
            torch.testing.assert_close(batch['density'], density, rtol=0, atol=0)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_js_onsets_are_identical_and_prefix_counts_are_committed(self):
        model = copy.deepcopy(self.model)
        enable_count_density_reference(model)
        batch = self.batch('Jumpstream')
        # Half a beat of committed silence must override recomputed prefix logits.
        batch['prefix_masks'] = torch.zeros(1, 12, dtype=torch.long)
        with torch.inference_mode():
            old, new = self.run_model(self.model, batch), self.run_model(model, batch)
        torch.testing.assert_close(old['onset_probability'], new['onset_probability'], rtol=0, atol=0)
        torch.testing.assert_close(new['masks'][:, :12], batch['prefix_masks'], rtol=0, atol=0)
        first = model.count_density_windows[0]['beats'][0]
        self.assertEqual(first['selected_music_rows'], int((new['onset_probability'][0, 12:24] >= .5).sum()))
        self.assertIsNone(model._count_density_prefix)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_reference_pass_and_unsupported_policies_are_protected(self):
        self.model.count_policy = 'argmax'
        with self.assertRaises(ValueError):
            enable_count_density_reference(self.model)
        self.model.count_policy = 'music-density-sample'
        enable_count_density_reference(self.model)
        with self.assertRaises(ValueError):
            enable_count_density_reference(self.model)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_sampling_stays_in_native_support_and_bounds_adjacent_count_odds(self):
        probability = torch.tensor([[.3, .7, 0., 0.], [.2, .3, .5, 0.], [0., 1., 0., 0.]])
        target = torch.tensor([1.1, 2.8, 1.])
        result = bounded_count_distribution(probability, target)
        torch.testing.assert_close(result.sum(-1), torch.ones(3))
        self.assertTrue((result[probability == 0] == 0).all())
        for row in range(2):
            count = int((probability[row] > 0).sum())
            old = torch.log(probability[row, 1:count] / probability[row, :count - 1])
            new = torch.log(result[row, 1:count] / result[row, :count - 1])
            self.assertLessEqual(float((new - old).abs().max()), .50001)
        values = torch.arange(1, 5)
        self.assertLess(float(((result * values).sum(-1)[:2] - target[:2]).abs().sum()),
                        float(((probability * values).sum(-1)[:2] - target[:2]).abs().sum()))
        torch.testing.assert_close(result[2], probability[2])
        torch.testing.assert_close(result[1], probability[1])

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_sampling_rejects_invalid_inputs(self):
        for probability in (torch.zeros(1, 4), torch.full((1, 4), float('nan')), torch.tensor([[-1., 2., 0., 0.]])):
            with self.assertRaises(ValueError):
                bounded_count_distribution(probability, torch.ones(1))

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_committed_music_prefix_takes_precedence_over_dropped_layout_rows(self):
        from keyloom_mania.motion_state import MotionHistory
        model = self.model
        enable_count_density_reference(model)
        batch = self.batch('Jumpstream')
        batch['prefix_masks'] = torch.zeros(1, 12, dtype=torch.long)
        history = MotionHistory()
        history.expression_music = (torch.ones(1, 12, dtype=torch.long), batch['candidate_beats'][:, :12].clone())
        with torch.inference_mode():
            result = self.run_model(model, batch, motion_state=history)
        first = model.count_density_windows[0]['beats'][0]
        self.assertEqual(first['selected_music_rows'], 12 + int((result['onset_probability'][0, 12:24] >= .5).sum()))

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_native_learned_mini_density_features_remain_unchanged(self):
        from keyloom_mania.music_count_natural import MusicCountNaturalGenerator
        model = MusicCountNaturalGenerator(self.model.config).eval()
        model.count_policy = 'music-density-sample'
        with torch.no_grad():
            model.music_count_adapter[-1].weight.normal_(std=.05)
        candidate = copy.deepcopy(model)
        enable_count_density_reference(candidate)
        batch = self.batch('Minijacks')
        with torch.inference_mode():
            old, new = self.run_model(model, batch), self.run_model(candidate, batch)
        for key in ('masks', 'music_counts', 'onset_probability'):
            torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_existing_native_instance_generate_hook_is_retained(self):
        from types import MethodType
        model = self.model
        native = model.generate
        def native_hook(self, *args, **kwargs):
            return native(*args, **kwargs)
        model.generate = MethodType(native_hook, model)
        del native_hook
        enable_count_density_reference(model)
        with torch.inference_mode():
            result = self.run_model(model, self.batch('Jumpstream'))
        self.assertEqual(result['masks'].shape, (1, 192))

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_disabled_policy_is_bit_identical(self):
        candidate = copy.deepcopy(self.model)
        enable_count_density_reference(candidate, enabled=False)
        batch = self.batch('Handstream')
        with torch.inference_mode():
            old, new = self.run_model(self.model, batch), self.run_model(candidate, batch)
        torch.testing.assert_close(old['masks'], new['masks'], rtol=0, atol=0)
        self.assertEqual(candidate.count_density_windows, [])

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_reviewed_budget_queue_support_and_provenance_settings(self):
        from keyloom_mania.generator_density import BUDGET_OPTIONS
        from keyloom_mania.generation_studio import StudioApp
        option = next(item for item in BUDGET_OPTIONS if item['id'] == 'count-reference')
        with tempfile.TemporaryDirectory() as directory, patch.dict(option, {'available': True}):
            root = Path(directory); source = root / 'source.wav'; source.write_bytes(b'fixture')
            app = StudioApp(root, 0); job = 'b' * 32
            app.jobs[job] = {'directory': str(root), 'source_path': str(source), 'filename': 'source.wav', 'status': 'ready'}
            with patch('keyloom_mania.generation_studio.threading.Thread') as worker:
                with self.assertRaises(ValueError):
                    app.generate(job, {'budget_id': 'count-reference', 'model_id': 'unsupported-model'})
                with self.assertRaises(ValueError):
                    app.generate(job, {'budget_id': 'count-reference', 'main_style': 'Handstream'})
                worker.assert_not_called()
                app.generate(job, {'budget_id': 'count-reference', 'main_style': 'Jumpstream'})
                worker.return_value.start.assert_called_once()
            self.assertEqual(json.loads((root / 'status.json').read_text(encoding='utf-8'))['settings']['budget_id'], 'count-reference')


if __name__ == '__main__':
    unittest.main()
