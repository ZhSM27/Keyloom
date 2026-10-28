import unittest

import numpy as np

from keyloom_mania.robust_beat_grid import fit_robust_grid


class RobustBeatGridTests(unittest.TestCase):
    def test_late_extra_and_missing_beats_do_not_corrupt_absolute_indices(self):
        rng=np.random.default_rng(19);true=.173+np.arange(400)*60/173
        beats=np.delete(true,[17,38,70,201,250]);beats=np.sort(np.r_[beats,true[120]+.17,true[180]+.16])
        beats=beats+rng.normal(0,.002,len(beats))
        result=fit_robust_grid(beats,true[::4])
        self.assertAlmostEqual(result['bpm'],173,delta=.02)
        self.assertAlmostEqual(result['offset_ms'],173,delta=2)

    def test_calibration_is_applied_to_estimator_only(self):
        beats=.23+np.arange(80)*.5;copy=beats.copy()
        a=fit_robust_grid(beats,beats[::4]);b=fit_robust_grid(beats,beats[::4],30)
        self.assertAlmostEqual(a['offset_ms']-b['offset_ms'],30,delta=.1)
        original={c['bpm']:c for c in a['alternatives']}
        stable=[c for c in b['alternatives'] if c['status']=='estimated']
        self.assertTrue(stable)
        for candidate in stable:
            self.assertEqual(candidate['estimator_phase_bias_applied_ms'],30)
            period=240000/candidate['bpm']
            self.assertAlmostEqual((original[candidate['bpm']]['offset_ms']-candidate['offset_ms'])%period,30,delta=.2)
        np.testing.assert_array_equal(beats,copy)

    def test_variable_tempo_is_uncertain_and_does_not_receive_bias(self):
        beats=.2+np.cumsum(np.linspace(.4,.6,120))
        result=fit_robust_grid(beats,beats[::4],30)
        self.assertEqual(result['status'],'uncertain')
        self.assertEqual(result['estimator_phase_bias_applied_ms'],0)


if __name__=='__main__':unittest.main()
