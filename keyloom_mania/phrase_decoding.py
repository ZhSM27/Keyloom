"""Probability sampling inside an already count/legality-masked layout set."""
import torch
import copy
from .segment_labels import LABELS


def sample_legal_layout(logits,generator,temperature=.8,logit_window=4.):
    if temperature<=0 or logit_window<=0:raise ValueError('Positive sampling parameters required.')
    if not torch.isfinite(logits).any(-1).all():raise ValueError('No legal layout; do not invent or delete a note here.')
    values=logits.masked_fill(logits<logits.max(-1,keepdim=True).values-logit_window,-torch.inf)
    return torch.multinomial((values/temperature).softmax(-1),1,generator=generator).squeeze(-1)


class LongjackSamplingPhrase:
    """A main-layout-only proxy; musical shadow deepcopy returns pristine state."""
    def __init__(self,state,device):
        self.state=state
        self.generator=torch.Generator(device=device).manual_seed(state.playstyle.seed+1000000)
        self.long_samples=0

    def __getattr__(self,name):
        if name=='state':raise AttributeError(name)
        return getattr(self.state,name)

    def __deepcopy__(self,memo):
        shadow=type(self.state).__new__(type(self.state))
        memo[id(self)]=shadow;memo[id(self.state)]=shadow
        shadow.__dict__.update({k:copy.deepcopy(v,memo) for k,v in self.state.__dict__.items() if k!='_longjack_proxy'})
        return shadow

    def choose(self,logits,beats,fixed_grid,plan=None):
        result=self.state.choose(logits,beats,fixed_grid,plan=plan)
        if plan is not None:
            active=plan[:,LABELS.index('Longjacks')]>.5
            if active.any():
                result[active]=sample_legal_layout(logits[active],self.generator)
                self.long_samples+=int(active.sum())
        return result

    def advance(self,*args,**kwargs):return self.state.advance(*args,**kwargs)
    def summary(self):return {**self.state.summary(),'longjack_sampled_rows':self.long_samples}


def enable_longjack_sampling(model):
    """Install a reversible decoder-only option; weights/count choices stay intact."""
    import types
    if getattr(model,'longjack_sampling_enabled',False):return
    model.longjack_sampling_enabled=True
    original=model.generate
    def generate(this,*args,**kwargs):
        state=kwargs.get('phrase_state')
        if state is not None and not isinstance(state,LongjackSamplingPhrase):
            # The caller persists its underlying state across windows. Persist
            # the independent Long sampler there too, without sharing the cut RNG.
            proxy=getattr(state,'_longjack_proxy',None)
            if proxy is None:
                phase=kwargs.get('phase',args[2] if len(args)>2 else None)
                proxy=LongjackSamplingPhrase(state,phase.device);state._longjack_proxy=proxy
            kwargs['phrase_state']=proxy
        return original(*args,**kwargs)
    model.generate=types.MethodType(generate,model)
