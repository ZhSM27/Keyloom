import copy
import unittest
import torch
from keyloom_mania.natural_release import context, release_scores, NaturalReleasePhrase, SCHEMA
from keyloom_mania.playstyle_control import PlaystylePhraseControlState, PlaystyleControlConfig
from keyloom_mania.segment_labels import LABELS


class NaturalReleaseTest(unittest.TestCase):
    def logits(self):
        x=torch.full((16,),-torch.inf);x[[3,5,6,9,10,12]]=0.;x[3]=2.
        return x

    def reference(self,history,t):
        coarse,fine=context(history,t,0,'Jacks',2)[:2]
        entry={'counts':[50,40,10],'groups':30}
        return {'schema':SCHEMA,'split':'train','test_used':False,'coarse':{coarse:entry},'fine':{fine:entry}}

    def test_only_two_and_four_beat_load_history(self):
        h=[(i/4,3) for i in range(12)]
        self.assertEqual(context(h,3.,0,'Jacks',2),context([(-50.,3),(-10.,3),*h],3.,0,'Jacks',2))
        self.assertIsNone(context(h,4.,0,'Jacks',2))

    def test_release_corrects_supported_overcontinuation_without_touching_legal_set(self):
        h=[(i/4,3) for i in range(16)];x=self.logits()
        y,n=release_scores(x,h,4.,'Jacks',self.reference(h,4.))
        self.assertGreater(n,0);self.assertLess(float(y[3]-y[12]),float(x[3]-x[12]))
        self.assertTrue(torch.equal(torch.isfinite(x),torch.isfinite(y)))
        self.assertLessEqual(float((y[torch.isfinite(x)]-x[torch.isfinite(x)]).abs().max()),.35*.7+1e-6)

    def test_aab_after_two_rows_is_not_penalized(self):
        h=[(0.,3),(.25,3)];x=self.logits();ref={'coarse':{},'fine':{}}
        y,n=release_scores(x,h,.5,'Jacks',ref)
        self.assertEqual(n,0);self.assertTrue(torch.equal(x,y))

    def test_mirror_equivariance(self):
        h=[(i/4,3) for i in range(16)];ref=self.reference(h,4.);x=self.logits()
        mirror=torch.tensor([((i&3)<<2)|((i&12)>>2) for i in range(16)])
        y,_=release_scores(x,h,4.,'Jacks',ref)
        z,_=release_scores(x[mirror],[(t,12) for t,m in h],4.,'Jacks',ref)
        torch.testing.assert_close(y[mirror],z)

    def test_no_impossible_release_or_count_change(self):
        h=[(i/4,3) for i in range(6)];x=torch.full((16,),-torch.inf);x[15]=0
        ref=self.reference(h,1.5);y,_=release_scores(x,h,1.5,'Jacks',ref)
        self.assertEqual(int(y.argmax()),15);self.assertTrue(torch.isfinite(y[15]))

    def test_dense_short_phrase_is_not_a_full_bar(self):
        h=[(i/4,3) for i in range(9)];x=self.logits();ref=self.reference(h,2.25)
        y,n=release_scores(x,h,2.25,'Jacks',ref)
        self.assertEqual(n,0);self.assertTrue(torch.equal(x,y))

    def test_no_period_ablation_keeps_the_ordinary_rng(self):
        from dataclasses import replace
        cfg=PlaystyleControlConfig(jack_period_strength=0.)
        direct=PlaystylePhraseControlState(1,'cpu',cfg)
        proxy=NaturalReleasePhrase(PlaystylePhraseControlState(1,'cpu',replace(cfg,jack_period_strength=2.5)),'cpu')
        plan=torch.zeros(1,len(LABELS));plan[0,LABELS.index('Jacks')]=1
        for i in range(30):
            beat=torch.tensor([i*.25]);x=self.logits()[None]
            a=direct.choose(x,beat,True,plan);b=proxy.choose(x,beat,True,plan)
            self.assertTrue(torch.equal(a,b))
            direct.advance(a,beat,True,plan);proxy.advance(b,beat,True,plan)

    def test_longjacks_receive_no_release_score(self):
        h=[(i/4,3) for i in range(16)];x=self.logits()
        y,n=release_scores(x,h,4.,'Longjacks',self.reference(h,4.))
        self.assertEqual(n,0);self.assertTrue(torch.equal(x,y))

    def test_source_shadow_and_cut_are_preserved(self):
        state=PlaystylePhraseControlState(1,'cpu',PlaystyleControlConfig())
        proxy=NaturalReleasePhrase(state,'cpu');state._natural_release_proxy=proxy
        shadow=copy.deepcopy(proxy)
        self.assertIsInstance(shadow,PlaystylePhraseControlState)
        self.assertFalse(hasattr(shadow,'_natural_release_proxy'))
        plan=torch.zeros(1,len(LABELS));plan[0,LABELS.index('Jacks')]=1
        for i in range(5):
            logits=self.logits()[None];beat=torch.tensor([i*.25])
            chosen=proxy.choose(logits,beat,True,plan)
            self.assertTrue(torch.isfinite(logits[0,chosen[0]]))
            proxy.advance(chosen,beat,True,plan)
        self.assertNotIn('jack_scores',state.__dict__)
        self.assertEqual(state.jack_cycle_adjustments,0)


if __name__=='__main__':unittest.main()
