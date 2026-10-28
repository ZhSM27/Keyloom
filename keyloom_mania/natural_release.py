"""TRAIN-calibrated local release, with no hand quota or AAB penalty.

Reference contexts use only occupied events inside the last 2/4 beats (inclusive
left boundary). A release adjustment requires four beats of observed continuity.
The decoder retains the upstream count and legality mask and its own musical RNG.
"""
import copy
import math
import types

import torch

from .segment_labels import LABELS
from .playstyle_control import JACK_STYLES

SCHEMA = 'natural-release-v2-2-4-full-bar'
STYLES = JACK_STYLES


def bucket(run):
    return next((v for v in (0, 1, 2, 3, 4, 5, 6, 8, 12, 16) if run <= v), 16)


def context(history, beat, hand, style, count):
    # Reconstruct rather than retain cumulative runs: no pre-four-beat debt.
    recent = [(t, m) for t, m in history if beat-4-1e-6 <= t < beat]
    gap = beat-recent[-1][0] if recent else math.inf
    if gap > .5+1e-6:
        return None
    flag = 3 if hand == 0 else 12
    runs = []; spans = []
    for window in (2., 4.):
        run = 0; last = beat; oldest = beat
        for t, m in reversed(recent):
            if t < beat-window-1e-6 or last-t > .5+1e-6 or not (m & flag):
                break
            run += 1; last = t; oldest = t
        runs.append(run); spans.append(min(window, beat-oldest))
    previous = (recent[-1][1] & flag).bit_count()
    coarse = f'{style}:{count}:{int(gap > .25+1e-6)}:{previous}:{int(spans[1]>=4-1e-6)}'
    fine = coarse+f':{bucket(runs[0])}:{bucket(runs[1])}:{round(spans[0]*4)}:{round(spans[1]*4)}'
    return coarse, fine, runs[1], spans[1]


def release_scores(logits, history, beat, style, reference, strength=.35):
    """A bounded posterior correction only after a full bar of participation.

    Estimate marginal hand-note probabilities within the CURRENT legal set;
    impossible releases are never rewarded and remain impossible. Reference
    outcomes are symmetric (both hands pooled), not a desired left/right ratio.
    """
    if style not in STYLES or strength == 0:
        return logits, 0
    finite = torch.isfinite(logits)
    masks = torch.arange(16, device=logits.device)
    counts = torch.tensor([i.bit_count() for i in range(16)], device=logits.device)
    available = counts[finite].unique()
    if len(available) != 1 or int(available[0]) == 0:
        return logits, 0
    count = int(available[0]); probability = logits.softmax(-1)
    adjustment = torch.zeros_like(logits); used = 0
    for hand, flag in enumerate((3, 12)):
        ctx = context(history, beat, hand, style, count)
        if ctx is None or ctx[3] < 4-1e-6:
            continue
        coarse, fine = ctx[:2]
        prior = reference['coarse'].get(coarse)
        if prior is None or prior['groups'] < 20:
            continue
        target = torch.tensor(prior['counts'], device=logits.device, dtype=logits.dtype)+1.
        target /= target.sum()
        entry = reference['fine'].get(fine)
        if entry is not None and entry['groups'] >= 8:
            n = torch.tensor(entry['counts'], device=logits.device, dtype=logits.dtype)
            target = (n+16*target)/(n.sum()+16)
        allocations = torch.tensor([(i & flag).bit_count() for i in range(16)], device=logits.device)
        supported = torch.stack([(finite & (allocations == k)).any() for k in range(3)])
        target = target*supported; target /= target.sum()
        predicted = torch.stack([probability[allocations == k].sum() for k in range(3)])
        correction = (target.clamp_min(1e-6).log()-predicted.clamp_min(1e-6).log()).clamp(-.7, .7)
        adjustment += correction[allocations]; used += 1
    return logits+strength*adjustment/max(1, used), used


class NaturalReleasePhrase:
    def __init__(self, state, device, reference=None, strength=.35):
        if reference is not None and (reference.get('schema') != SCHEMA or reference.get('split') != 'train' or reference.get('test_used') is not False):
            raise ValueError('Release calibration must be versioned TRAIN-only 2/4-beat data.')
        self.state = state; self.reference = reference; self.strength = strength
        self.events = [[] for _ in state.history]
        self.release_rows = 0; self.sampled_rows = 0

    def __getattr__(self, name):
        if name == 'state': raise AttributeError(name)
        return getattr(self.state, name)

    def __deepcopy__(self, memo):
        shadow = type(self.state).__new__(type(self.state))
        memo[id(self)] = shadow; memo[id(self.state)] = shadow
        shadow.__dict__.update({k:copy.deepcopy(v, memo) for k,v in self.state.__dict__.items() if k != '_natural_release_proxy'})
        return shadow

    def choose(self, logits, beats, fixed_grid, plan=None):
        scores = logits.clone()
        if self.reference is not None and plan is not None:
            for i in range(len(logits)):
                if not bool(plan[i].sum()) or torch.isfinite(logits[i,0]): continue
                style = LABELS[int(plan[i].argmax())]
                scores[i], used = release_scores(logits[i], self.events[i], float(beats[i]), style, self.reference, self.strength)
                self.release_rows += int(used > 0)
        # Keep ordinary cut behavior; do not apply the legacy jack cycle score.
        had_override = 'jack_scores' in self.state.__dict__
        original = self.state.jack_scores
        self.state.jack_scores = types.MethodType(lambda self,x,*a:x, self.state)
        try:
            # The same ordinary sampler and RNG as v8: A changes only the
            # jack period scores; R additionally changes supported local scores.
            result = self.state.choose(scores, beats, fixed_grid, plan=plan)
        finally:
            # Remove the temporary instance override, retaining class dispatch.
            if had_override: self.state.jack_scores = original
            else: del self.state.jack_scores
        self.sampled_rows = self.state.jack_choices
        return result

    def advance(self, masks, beats, fixed_grid, plan=None):
        self.state.advance(masks, beats, fixed_grid, plan=plan)
        for i,m in enumerate(masks.tolist()):
            t = float(beats[i]); history = [(b,v) for b,v in self.events[i] if b >= t-4-1e-6]
            if m: history.append((t,m))
            self.events[i] = history

    def summary(self):
        return {**self.state.summary(), 'natural_release_schema':SCHEMA, 'release_scored_rows':self.release_rows,
                'natural_sampled_rows':self.sampled_rows, 'jack_policy':'Same ordinary sampler/RNG; no jack cycle/AAB penalty; optional full-bar local release; Long and cut choice unchanged; existing gameplay caps retained.',
                'pressure_windows_beats':[2,4], 'release_strength':self.strength}


def enable_natural_release(model, reference=None, strength=.35):
    if getattr(model, 'natural_release_enabled', False): return
    model.natural_release_enabled = True
    model.natural_release_schema = SCHEMA
    original = model.generate
    def generate(this, *args, **kwargs):
        state = kwargs.get('phrase_state')
        if state is not None and not isinstance(state, NaturalReleasePhrase):
            proxy = getattr(state, '_natural_release_proxy', None)
            if proxy is None:
                phase = kwargs.get('phase', args[2] if len(args)>2 else None)
                proxy = NaturalReleasePhrase(state, phase.device, reference, strength)
                state._natural_release_proxy = proxy
            kwargs['phrase_state'] = proxy
        return original(*args, **kwargs)
    model.generate = types.MethodType(generate, model)
