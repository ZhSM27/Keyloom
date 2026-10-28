import unittest
import torch
from keyloom_mania.count_continuation import CountContinuationGenerator
from keyloom_mania.tiny_model import TinyModelConfig
from keyloom_mania.segment_labels import LABELS


class CountContinuationTest(unittest.TestCase):
    def model(self):
        torch.set_num_threads(2)
        m=CountContinuationGenerator(TinyModelConfig(width=24,heads=3,audio_layers=1,decoder_layers=2,feedforward=48),reference_count_fallback=True,reference_layout_fallback=True).eval()
        m.expression_decoding=True;m.learned_count_lookahead_enabled=True
        return m
    def test_source_pass_and_nonmini_bypass_new_lookahead(self):
        m=self.model();x=torch.zeros(1,16);plan=torch.zeros(1,3,len(LABELS));plan[:,:,LABELS.index('Longjacks')]=1
        m._reference_pass=True
        result=m.generation_layout_scores(x,None,None,None,None,torch.tensor([2]),plan,0,torch.tensor([[0.,.25,.5]]),None,torch.ones(1,3))
        self.assertTrue(torch.equal(x,result))
    def test_no_extra_trainable_state(self):
        m=self.model()
        self.assertTrue(m.learned_music_counts)
        self.assertEqual(m.configuration()['pressure_windows_beats'],[2.,4.])
        self.assertFalse(any('lookahead' in k for k in m.state_dict()))
    def test_mini_uses_learned_future_count_without_editing_legal_mask(self):
        from keyloom_mania.playstyle_control import PlaystyleLaneControlState
        from keyloom_mania.lane_control import LaneControlConfig
        from keyloom_mania.expression_decoder import next_event_viability
        m=self.model();lane=PlaystyleLaneControlState(1,'cpu',LaneControlConfig(policy='style-aware'))
        plan=torch.zeros(1,3,len(LABELS));plan[:,:,LABELS.index('Minijacks')]=1
        m._reference_anchor=torch.tensor([[[3.,0.,0.,0.]]*3]);m._expression_stats={'viability_scored_rows':0}
        m.generation_count_logits=lambda *args:torch.tensor([[[0.,3.,0.,0.]]*2])
        x=torch.full((1,16),-torch.inf);x[:,[3,5,6,9,10,12]]=0.
        beats=torch.tensor([[0.,.25,.5]])
        result=m.generation_layout_scores(x,None,None,None,torch.tensor([[5]]),torch.tensor([2]),plan,0,beats,lane,torch.ones(1,3))
        expected=x+next_event_viability(lane,beats[:,0],beats[:,1],torch.tensor([2]),plan[:,0])
        torch.testing.assert_close(result,expected)
        self.assertTrue(torch.equal(torch.isfinite(x),torch.isfinite(result)))
        self.assertEqual(m._expression_stats['learned_count_lookahead_changed_rows'],1)

if __name__=='__main__':unittest.main()
