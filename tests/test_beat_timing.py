import unittest
import numpy as np
from keyloom_mania.beat_timing import fit_beat_grid
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from types import ModuleType
from unittest.mock import Mock, patch


class GridFitTests(unittest.TestCase):
    def test_downbeat_phase_is_preserved_with_missing_beats(self):
        beats = .173 + np.arange(120) * (60 / 137.5)
        downbeats = beats[2::4]
        result = fit_beat_grid(np.delete(beats, [17, 38, 70]), downbeats)
        self.assertAlmostEqual(result['bpm'], 137.5, places=2)
        self.assertAlmostEqual(result['offset_ms'], 173 + 2 * 60_000 / 137.5, delta=.2)

    def test_drifting_tempo_is_not_reported_as_a_stable_grid(self):
        beats = .2 + np.cumsum(np.linspace(.4, .6, 120))
        result = fit_beat_grid(beats, beats[::4])
        self.assertEqual(result['status'], 'uncertain')
        self.assertGreater(result['grid_error_p90_ms'], 45)

    def test_actual_detector_entry_applies_pinned_estimator_calibration(self):
        from keyloom_mania import beat_timing
        beats=.2039+np.arange(120)*(60/137.5)
        tracker = Mock()
        inference = ModuleType('beat_this.inference')
        inference.Audio2Beats = tracker
        optional_package = ModuleType('beat_this')
        optional_package.__path__ = []
        optional_package.inference = inference
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'weights';checkpoint.write_bytes(b'tracker fixture')
            calibration=Path(directory)/'phase.json';calibration.write_text(json.dumps({'estimator_phase_bias_ms':30.9}),encoding='utf-8')
            with patch.object(beat_timing,'CHECKPOINT',checkpoint),patch.object(beat_timing,'CHECKPOINT_SHA256',hashlib.sha256(checkpoint.read_bytes()).hexdigest()), \
                 patch.object(beat_timing,'PHASE_CALIBRATION',calibration),patch.object(beat_timing,'PHASE_CALIBRATION_SHA256',hashlib.sha256(calibration.read_bytes()).hexdigest()), \
                 patch.dict(sys.modules, {'beat_this': optional_package, 'beat_this.inference': inference}), \
                 patch('torch.cuda.is_available',return_value=False):
                tracker.return_value.return_value=(beats,beats[::4])
                result=beat_timing.detect_timing(np.ones(24000,dtype=np.float32))
                tracker.assert_called_once_with(checkpoint_path=str(checkpoint), device='cpu', dbn=False)
                self.assertAlmostEqual(result['offset_ms'],173.,delta=.2)
                self.assertEqual(result['fit_policy'],'absolute-consensus-v4')
                self.assertEqual(result['phase_calibration_sha256'],hashlib.sha256(calibration.read_bytes()).hexdigest())
                calibration.write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError,'校验失败'):beat_timing.detect_timing(np.ones(24000,dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
