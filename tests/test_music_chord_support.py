import unittest
import torch
from keyloom_mania.music_chord_support import supported_counts
from keyloom_mania.segment_labels import LABELS

class ChordSupportTest(unittest.TestCase):
    def inputs(self,label='Chordjacks'):
        p=torch.tensor([[[.45,.40,.10,.05]]*4]).log()
        onset=torch.ones(1,4);plan=torch.zeros(1,4,len(LABELS));plan[:,:,LABELS.index(label)]=1
        density=torch.tensor([[[6.,3.]]*4]);beats=torch.tensor([[0.,.25,.5,.75]])
        return p,onset,plan,density,beats
    def test_supported_doubles_use_only_available_music_budget(self):
        args=self.inputs();r=supported_counts(*args)
        self.assertEqual(int(r.sum()),6);self.assertEqual(int((r==2).sum()),2)
    def test_confident_singles_are_not_forced_into_chords(self):
        args=list(self.inputs());args[0]=torch.tensor([[[.9,.08,.01,.01]]*4]).log()
        self.assertTrue((supported_counts(*args)==1).all())
    def test_no_onsets_created(self):
        args=list(self.inputs());args[1].zero_();self.assertEqual(int(supported_counts(*args).sum()),0)
    def test_long_and_cut_are_unchanged(self):
        for label in ('Longjacks','Handstream'):
            self.assertTrue((supported_counts(*self.inputs(label))==1).all())
    def test_no_underfill_or_sparse_music_means_no_addition(self):
        for notes,rows in ((4.,2.),(6.,4.)):
            args=list(self.inputs());args[3][:,:,0]=notes;args[3][:,:,1]=rows
            self.assertTrue((supported_counts(*args)==1).all())
    def test_runtime_hook_does_not_override_hs_sampling(self):
        from keyloom_mania.music_chord_support import enable_music_chord_support
        class Dummy:
            _reference_pass=False
            def generation_count_choice(self,counts,*args):return counts
            def generation_onset_logits(self,*args):return torch.zeros(1,1)
            def generate(self,*args,**kwargs):return {}
        model=Dummy();enable_music_chord_support(model);model._supported_music_counts=torch.tensor([[2]])
        plan=torch.zeros(1,1,len(LABELS));plan[:,:,LABELS.index('Handstream')]=1
        args=(torch.tensor([1]),None,None,None,None,None,None,plan,0,None,None,False)
        self.assertEqual(int(model.generation_count_choice(*args)[0]),1)
        plan.zero_();plan[:,:,LABELS.index('Chordjacks')]=1
        self.assertEqual(int(model.generation_count_choice(*args)[0]),2)
if __name__=='__main__':unittest.main()
