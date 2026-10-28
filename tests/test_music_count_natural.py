import unittest
import torch
from keyloom_mania.music_count_natural import MusicCountNaturalGenerator,count_events
from keyloom_mania.tiny_model import TinyModelConfig
from keyloom_mania.segment_labels import LABELS

class LearnedCountTest(unittest.TestCase):
    def test_teacher_and_generation_add_the_residual_once(self):
        from keyloom_mania.phrase_natural import PhraseNaturalGenerator
        torch.set_num_threads(2);torch.manual_seed(42)
        cfg=TinyModelConfig(width=24,heads=3,audio_layers=1,decoder_layers=2,feedforward=48)
        base=PhraseNaturalGenerator(cfg,reference_count_fallback=True,reference_layout_fallback=True).eval()
        model=MusicCountNaturalGenerator(cfg,reference_count_fallback=True,reference_layout_fallback=True).eval()
        model.load_state_dict(base.state_dict(),strict=False)
        memory=torch.randn(1,4,24);phase=torch.randn(1,4,5);condition=torch.randn(1,5);previous=torch.tensor([[5,1,2,1]])
        plan=torch.zeros(1,4,len(LABELS));plan[:,:,LABELS.index('Chordjacks')]=1
        model._music_density=torch.ones(1,4,2)
        with torch.no_grad():
            original=base.decode_counts(memory,phase,condition,previous,plan,True)
            torch.testing.assert_close(model.decode_counts(memory,phase,condition,previous,plan,True),original)
            model.music_count_adapter[-1].bias.fill_(.2)
            model._reference_anchor=original.clone()
            teacher=model.decode_counts(memory,phase,condition,previous,plan,True)
            generated=model.generation_count_logits(memory,phase,condition,previous,None,plan,True)
            torch.testing.assert_close(teacher,generated)
            model._reference_pass=True
            torch.testing.assert_close(model.decode_counts(memory,phase,condition,previous,plan,True),original)

    def test_count_context_has_no_layout_or_bos_leak(self):
        v=count_events(torch.tensor([[5,0,2,0,1]]))
        self.assertEqual(v.shape,(1,5,12));self.assertEqual(float(v[0,0].sum()),0)
        self.assertEqual(float(v[0,-1,:4].argmax()),0)
    def test_zero_adapter_preserves_logits_and_nonjack_is_protected(self):
        torch.set_num_threads(2)
        cfg=TinyModelConfig(width=24,heads=3,audio_layers=1,decoder_layers=2,feedforward=48)
        model=MusicCountNaturalGenerator(cfg,reference_count_fallback=True,reference_layout_fallback=True).eval()
        memory=torch.randn(1,4,24);phase=torch.randn(1,4,5);condition=torch.randn(1,5)
        previous=torch.tensor([[5,1,2,1]]);plan=torch.zeros(1,4,len(LABELS));plan[:,:,LABELS.index('Chordjacks')]=1
        model._count_training_density=torch.ones(1,4,2);base=torch.randn(1,4,4)
        self.assertTrue((model.count_residual(base,memory,phase,condition,previous,plan)==0).all())
        with torch.no_grad():model.music_count_adapter[-1].bias.fill_(1)
        self.assertTrue((model.count_residual(base,memory,phase,condition,previous,plan)>0).all())
        plan.zero_();plan[:,:,LABELS.index('Longjacks')]=1
        self.assertTrue((model.count_residual(base,memory,phase,condition,previous,plan)==0).all())
if __name__=='__main__':unittest.main()
