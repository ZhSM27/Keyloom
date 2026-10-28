"""Music-only count adaptation; fixed onset/layout and legacy musical shadow."""
import torch
from torch import nn
from .phrase_natural import PhraseNaturalGenerator
from .playstyle_control import JACK_STYLES
from .segment_labels import LABELS

SCHEMA='real-musical-count-v1'


def count_events(previous):
    positions=torch.arange(previous.shape[1],device=previous.device)[None].expand_as(previous)
    occupied=(previous>0)&(previous<5)
    last=torch.where(occupied,positions,-1).cummax(1).values
    index=last;values=[]
    for _ in range(3):
        counts=previous.gather(1,index.clamp_min(0))
        counts=torch.where(index>=0,counts,0).clamp_max(4)
        values.append(nn.functional.one_hot(counts,5)[...,1:].float())
        before=last.gather(1,(index-1).clamp_min(0))
        index=torch.where(index>0,before,-1)
    return torch.cat(values,-1)


class MusicCountNaturalGenerator(PhraseNaturalGenerator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # 192 music memory + phase5 + condition5 + type15 + density2 +
        # original count probabilities4 + last-three occupied counts12.
        width=self.config.width+5+5+len(LABELS)+2+4+12
        self.music_count_adapter=nn.Sequential(nn.Linear(width,128),nn.GELU(),nn.Linear(128,4))
        nn.init.zeros_(self.music_count_adapter[-1].weight);nn.init.zeros_(self.music_count_adapter[-1].bias)
        self.learned_music_counts=True
        self._count_training_density=None

    def count_residual(self,logits,memory,phase,condition,previous_counts,plan):
        length=previous_counts.shape[1]
        density=self._count_training_density if self._count_training_density is not None else self._music_density
        if density is None:raise ValueError('Learned counts require explicit musical density conditions.')
        features=torch.cat((memory[:,:length].detach(),phase[:,:length],condition[:,None].expand(-1,length,-1),
            plan[:,:length],density[:,:length],logits.detach().softmax(-1),count_events(previous_counts)),-1)
        residual=1.5*self.music_count_adapter(features).tanh()
        active=plan[:,:length,[LABELS.index(s) for s in JACK_STYLES]].amax(-1)>.5
        return torch.where(active[...,None],residual,torch.zeros_like(residual))

    def decode_counts(self,memory,phase,condition,previous_counts,plan,count_plan_enabled):
        base=super().decode_counts(memory,phase,condition,previous_counts,plan,count_plan_enabled)
        if self._reference_pass or not count_plan_enabled:return base
        return base+self.count_residual(base,memory,phase,condition,previous_counts,plan)

    def generation_count_logits(self,memory,phase,condition,previous_counts,previous,plan,count_plan_enabled):
        base=super().generation_count_logits(memory,phase,condition,previous_counts,previous,plan,count_plan_enabled)
        if self._reference_pass or self._reference_anchor is None or not count_plan_enabled:return base
        # Expression's canonical anchor replaces non-HS/JS logits, so the new
        # jack residual must be applied after that replacement at inference.
        return base+self.count_residual(base,memory,phase,condition,previous_counts,plan)

    def configuration(self):
        return {**super().configuration(),'architecture':'music-count-v1','music_count_schema':SCHEMA,
                'learned_music_count_styles':list(JACK_STYLES),'count_residual_bound':1.5,
                'onsets_frozen':True,'layout_weights_frozen':True,'count_inputs_include_hand_pressure':False}
