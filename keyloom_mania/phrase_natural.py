"""Real phrase adaptation with separate musical-plan and finite hand observations."""
import torch
from torch import nn
from .hand_dynamics_generator import HandDynamicsGenerator
from .tiny_model import TinyModelConfig

PLAN_SCHEMA = 'future-style-plan-v1-2-4-8'


def musical_plan_context(plan):
    """Upcoming type proportions/changes; no past hand loads or target note masks."""
    batch, length, width = plan.shape
    prefix = torch.cat((plan.new_zeros(batch, 1, width), plan.cumsum(1)), 1)
    start = torch.arange(length, device=plan.device)
    blocks = []
    for beats in (2, 4, 8):
        end = (start + beats*24).clamp_max(length)
        blocks.append((prefix[:, end]-prefix[:, start])/(end-start)[None,:,None])
    known = plan.sum(-1)>0
    ids = plan.argmax(-1)
    changed = torch.cat((torch.zeros(batch,1,device=plan.device,dtype=torch.bool),
                         (ids[:,1:]!=ids[:,:-1]) | (known[:,1:]!=known[:,:-1])),1)
    positions = start[None].expand(batch,-1)
    before = torch.where(changed,positions,0).cummax(1).values
    after = torch.where(changed,positions,torch.full_like(positions,length)).flip(1).cummin(1).values.flip(1)
    return torch.cat((*blocks,(positions-before).float()[...,None]/192,
                      (after-positions).float()[...,None]/192),-1)


class PhraseNaturalGenerator(HandDynamicsGenerator):
    def __init__(self, config=TinyModelConfig(), **kwargs):
        super().__init__(config,**kwargs)
        self.phrase_plan_projection=nn.Linear(47,config.width)
        nn.init.zeros_(self.phrase_plan_projection.weight)
        nn.init.zeros_(self.phrase_plan_projection.bias)

    def decode_layout(self,memory,phase,condition,previous,plan,current_counts):
        if self._reference_pass:
            return super().decode_layout(memory,phase,condition,previous,plan,current_counts)
        # Inject only into the new layout branch. The musical source and protected
        # cut decoder do not call short_projection and therefore stay unchanged.
        length=previous.shape[1]
        hint=self.phrase_plan_projection(musical_plan_context(plan)[:,:length]).reshape(-1,self.config.width)
        handle=self.short_projection.register_forward_hook(lambda module,args,result:result+hint)
        try:
            return super().decode_layout(memory,phase,condition,previous,plan,current_counts)
        finally:
            handle.remove()

    def configuration(self):
        return {**super().configuration(),'architecture':'phrase-natural-v8','musical_plan_schema':PLAN_SCHEMA,
                'musical_plan_horizon_beats':[2,4,8], 'generated_prefix_training':False,
                'layout_adaptation':'real complete paths; final two layout decoder layers; future type plan; finite hand context'}
