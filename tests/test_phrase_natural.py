import unittest
import numpy as np
import torch
from keyloom_mania.phrase_natural import PhraseNaturalGenerator,musical_plan_context
from keyloom_mania.expression_generator import ExpressionGenerator
from keyloom_mania.tiny_model import TinyModelConfig
from keyloom_mania.phrase_decoding import sample_legal_layout,LongjackSamplingPhrase


class MusicalPlanTest(unittest.TestCase):
    def test_long_sampling_is_not_copied_into_the_musical_shadow(self):
        import copy
        from keyloom_mania.playstyle_control import PlaystylePhraseControlState,PlaystyleControlConfig
        state=PlaystylePhraseControlState(1,'cpu',PlaystyleControlConfig())
        proxy=LongjackSamplingPhrase(state,'cpu')
        state._longjack_proxy=proxy
        shadow=copy.deepcopy(proxy)
        self.assertIsInstance(shadow,PlaystylePhraseControlState)
        self.assertNotIsInstance(shadow,LongjackSamplingPhrase)
        self.assertIsNot(shadow.generator,state.generator)
        self.assertFalse(hasattr(shadow,'_longjack_proxy'))

    def test_sampling_preserves_the_legal_mask_and_music_count(self):
        logits=torch.full((256,16),-torch.inf);logits[:,[3,5,6,9,10,12]]=0
        chosen=sample_legal_layout(logits,torch.Generator().manual_seed(42))
        self.assertGreater(len(set(chosen.tolist())),2)
        self.assertTrue(all(int(m).bit_count()==2 for m in chosen))
        self.assertTrue(torch.isfinite(logits.gather(1,chosen[:,None])).all())

    def test_future_type_change_is_visible_without_note_history(self):
        a=torch.zeros(1,192,15);a[:,:,2]=1
        b=a.clone();b[:,96:,2]=0;b[:,96:,6]=1
        self.assertFalse(torch.equal(musical_plan_context(a)[:,0],musical_plan_context(b)[:,0]))
        self.assertTrue(torch.isfinite(musical_plan_context(b)).all())

    def test_zero_new_branches_preserve_baseline_and_protected_cut(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        cfg=TinyModelConfig(width=24,heads=3,audio_layers=1,decoder_layers=2,feedforward=48)
        base=ExpressionGenerator(cfg,reference_count_fallback=True,reference_layout_fallback=True).eval()
        model=PhraseNaturalGenerator(cfg,reference_count_fallback=True,reference_layout_fallback=True).eval()
        missing,unexpected=model.load_state_dict(base.state_dict(),strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(n.startswith(('short_','hand_allocation_head.','protected_layout_decoder.','phrase_plan_projection.')) for n in missing))
        model.protected_layout_decoder.load_state_dict(base.decoder.state_dict())
        memory=torch.randn(1,12,24);phase=torch.randn(1,12,5);phase[...,4]=.4
        condition=torch.randn(1,5);previous=torch.tensor([[16,3,12,5,0,3,12,6,0,3,0,12]])
        plan=torch.zeros(1,12,15);plan[:,:,2]=1;counts=torch.full((1,12),2)
        with torch.no_grad():
            a=base.decode_layout(memory,phase,condition,previous,plan,counts)[1]
            b=model.decode_layout(memory,phase,condition,previous,plan,counts)[1]
        torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)
        with torch.no_grad():
            model.phrase_plan_projection.bias.fill_(1)
            for layer in model.decoder.layers:
                for parameter in layer.parameters():parameter.add_(.01)
            for reference_pass in (True,False):
                base._reference_pass=model._reference_pass=reference_pass
                plan.zero_();plan[:,:,6]=1
                a=base.decode_layout(memory,phase,condition,previous,plan,counts)[1]
                b=model.decode_layout(memory,phase,condition,previous,plan,counts)[1]
                torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertFalse(model.short_projection._forward_hooks)


if __name__=='__main__':unittest.main()
