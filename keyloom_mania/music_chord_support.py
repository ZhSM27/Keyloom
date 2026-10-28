"""Restore ambiguous musical doubles only inside a supported beat-note deficit."""
import types
import torch
from .segment_labels import LABELS
from .playstyle_control import JACK_STYLES

SCHEMA='music-chord-support-v1'


def supported_counts(log_probability,onsets,plan,density,beats):
    probability=log_probability.softmax(-1)
    original=(probability.argmax(-1)+1)*(onsets>=.5)
    result=original.clone()
    active=plan[:,:,[LABELS.index(s) for s in JACK_STYLES]].amax(-1)>.5
    eligible=active&(original==1)&(probability[:,:,1]>=.65*probability[:,:,0])
    # Preserve musical single accents with high model confidence. A density
    # request alone never overrides a confident single, onset, or Long passage.
    for b in range(len(result)):
        ids=torch.floor(beats[b]+1e-6).long()
        for beat in ids.unique():
            members=ids==beat
            notes=density[b,members,0].mean()
            rows=density[b,members,1].mean().clamp_min(.1)
            if notes/rows<1.6:continue
            budget=max(0,int(torch.round(notes))-int(original[b,members].sum()))
            positions=(members&eligible[b]).nonzero().flatten()
            if budget==0 or not len(positions):continue
            score=(probability[b,positions,1]/probability[b,positions,0].clamp_min(1e-8))
            chosen=positions[torch.argsort(score,descending=True,stable=True)[:budget]]
            result[b,chosen]=2
    return result


def enable_music_chord_support(model):
    if getattr(model,'music_chord_support_enabled',False):return
    model.music_chord_support_enabled=True;model.music_chord_support_schema=SCHEMA
    model._supported_music_counts=None
    onset=model.generation_onset_logits;choice=model.generation_count_choice;generate=model.generate
    def onset_logits(this,memory,condition,plan):
        logits=onset(memory,condition,plan)
        if not this._reference_pass and this._reference_anchor is not None:
            this._supported_music_counts=supported_counts(this._reference_anchor,logits.sigmoid(),plan,this._music_density,this._density_beats)
        return logits
    def count_choice(this,counts,conditional,memory,phase,condition,previous,generated_counts,plan,position,candidate_beats,lane_state,constrained_counts):
        result=choice(counts,conditional,memory,phase,condition,previous,generated_counts,plan,position,candidate_beats,lane_state,constrained_counts)
        if this._reference_pass or constrained_counts or this._supported_music_counts is None:return result
        target=this._supported_music_counts[:,position]
        jack=plan[:,position,[LABELS.index(s) for s in JACK_STYLES]].amax(-1)>.5
        return torch.where(jack&(result==1)&(target==2),target,result)
    def wrapped(this,*args,**kwargs):
        try:return generate(*args,**kwargs)
        finally:this._supported_music_counts=None
    model.generation_onset_logits=types.MethodType(onset_logits,model)
    model.generation_count_choice=types.MethodType(count_choice,model)
    model.generate=types.MethodType(wrapped,model)
