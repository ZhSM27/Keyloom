import copy
import unittest
import torch
from keyloom_mania.cross_double_control import cross_double_scores, CrossDoublePhrase, CROSS_DOUBLES
from keyloom_mania.playstyle_control import PlaystylePhraseControlState, PlaystyleControlConfig


class CrossDoubleTest(unittest.TestCase):
    def logits(self):
        x=torch.full((16,),-torch.inf);x[[3,5,6,9,10,12]]=0.;return x

    def test_all_four_identical_triples_are_discouraged(self):
        for mask in CROSS_DOUBLES:
            x=self.logits();y,changed=cross_double_scores(x,[(0.,mask),(.25,mask)],.5,'Jacks')
            self.assertTrue(changed);self.assertLess(float(y[mask]),float(x[mask]))
            self.assertTrue(torch.equal(torch.isfinite(x),torch.isfinite(y)))

    def test_second_hit_aab_and_switches_are_retained(self):
        for history in ([(0.,5)],[(0.,3),(.25,3)],[(0.,5),(.25,10)],[(0.,9),(.25,6)]):
            x=self.logits();y,changed=cross_double_scores(x,history,.5,'Jacks')
            self.assertFalse(changed);self.assertTrue(torch.equal(x,y))

    def test_longjacks_and_pause_bypass(self):
        x=self.logits();history=[(0.,5),(.25,5)]
        for beat,style in ((.5,'Longjacks'),(1.,'Chordjacks')):
            y,changed=cross_double_scores(x,history,beat,style)
            self.assertFalse(changed);self.assertTrue(torch.equal(x,y))

    def test_no_alternative_does_not_drop_a_row(self):
        x=torch.full((16,),-torch.inf);x[5]=0
        y,changed=cross_double_scores(x,[(0.,5),(.25,5)],.5,'Handstream')
        self.assertFalse(changed);self.assertEqual(int(y.argmax()),5)

    def test_older_history_does_not_affect_penalty(self):
        h=[(0.,5),(.25,5)];x=self.logits()
        a,_=cross_double_scores(x,h,.5,'Jacks');b,_=cross_double_scores(x,[(-8.,5),*h],.5,'Jacks')
        self.assertTrue(torch.equal(a,b))

    def test_musical_shadow_is_pristine(self):
        state=PlaystylePhraseControlState(1,'cpu',PlaystyleControlConfig())
        proxy=CrossDoublePhrase(state,'cpu');state._natural_release_proxy=proxy
        shadow=copy.deepcopy(proxy)
        self.assertIsInstance(shadow,PlaystylePhraseControlState)
        self.assertFalse(hasattr(shadow,'_natural_release_proxy'))


if __name__=='__main__':unittest.main()
