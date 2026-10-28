"""Discourage identical cross-hand double triples outside explicit Longjacks."""
import types

import torch

from .natural_release import NaturalReleasePhrase
from .segment_labels import LABELS

SCHEMA = 'cross-double-repeat-v1'
CROSS_DOUBLES = (5, 10, 9, 6)  # 13, 24, 14, 23; 12/34 are intentionally absent.


def cross_double_scores(logits, history, beat, style, strength=2.5):
    """Soft correction only; all existing -inf/count decisions are preserved."""
    if style == 'Longjacks' or strength == 0 or not history:
        return logits, False
    last = history[-1][1]
    if last not in CROSS_DOUBLES:
        return logits, False
    run = 0; previous = beat
    for t, mask in reversed(history):
        if t < beat-4-1e-6 or previous-t > .5+1e-6 or mask != last:
            break
        run += 1; previous = t
    if run < 2 or not torch.isfinite(logits[last]):
        return logits, False
    alternatives = [m for m in range(16) if m != last and m.bit_count() == 2 and torch.isfinite(logits[m])]
    if not alternatives:
        return logits, False
    result = logits.clone()
    result[last] -= min(4., strength*(run-1))
    return result, True


class CrossDoublePhrase(NaturalReleasePhrase):
    def __init__(self, state, device, reference=None, strength=.35, repeat_strength=2.5):
        super().__init__(state,device,reference,strength)
        self.repeat_strength = repeat_strength; self.cross_scored_rows = 0

    def choose(self, logits, beats, fixed_grid, plan=None):
        scores = logits.clone()
        if plan is not None:
            for i in range(len(logits)):
                if not bool(plan[i].sum()):continue
                style = LABELS[int(plan[i].argmax())]
                scores[i],changed = cross_double_scores(logits[i],self.events[i],float(beats[i]),style,self.repeat_strength)
                self.cross_scored_rows += int(changed)
        return super().choose(scores,beats,fixed_grid,plan=plan)

    def summary(self):
        return {**super().summary(),'cross_double_schema':SCHEMA,'cross_double_scored_rows':self.cross_scored_rows,
                'cross_double_policy':'Soft penalty for identical 13/24/14/23 third+ occupied rows at gaps <= half beat; Longjacks bypass; alternatives required; no note/count deletion.'}


def enable_cross_double_control(model, reference=None, strength=.35, repeat_strength=2.5):
    if getattr(model,'cross_double_repeat_enabled',False):return
    model.cross_double_repeat_enabled = True; model.cross_double_schema = SCHEMA
    original = model.generate
    def generate(this,*args,**kwargs):
        state = kwargs.get('phrase_state')
        if state is not None and not isinstance(state,CrossDoublePhrase):
            proxy = getattr(state,'_natural_release_proxy',None)
            if not isinstance(proxy,CrossDoublePhrase):
                phase = kwargs.get('phase',args[2] if len(args)>2 else None)
                proxy = CrossDoublePhrase(state,phase.device,reference,strength,repeat_strength)
                state._natural_release_proxy = proxy
            kwargs['phrase_state'] = proxy
        return original(*args,**kwargs)
    model.generate = types.MethodType(generate,model)
