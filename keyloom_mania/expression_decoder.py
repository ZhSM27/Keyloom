"""Bounded next-event feasibility; no timing or musical count edits."""
import torch

from .segment_labels import LABELS


def stream_repeat_scores(lane,beats,plan,logits):
    """Soft excess 16th-chain cost above the best legal same-count choice."""
    selected=plan[:,[LABELS.index('Jumpstream'),LABELS.index('Handstream')]].amax(-1)>.5
    adjacent=((beats[:,None]-lane.last_hit)-.25).abs()<=1e-6
    runs=torch.where(adjacent,lane.sixteenth_runs,0).float()
    cost=runs.square()@lane.mask_columns.float().T
    available=torch.isfinite(logits)
    minimum=cost.masked_fill(~available,torch.inf).amin(-1,keepdim=True)
    minimum=torch.where(torch.isfinite(minimum),minimum,0.)
    excess=(cost-minimum).clamp(0,4)
    return torch.where(selected[:,None]&available,-excess,0.)


def next_event_viability(lane, beats, next_beats, next_counts, plan):
    """Score each current mask by available next layouts under typical run caps.

    New-chain random ceilings are unknown until a mask is committed. Their
    conservative two/four-hit values are a soft estimate, never a legality mask.
    """
    mini=plan[:,LABELS.index('Minijacks')]>.5
    columns=lane.mask_columns
    current=columns[None].expand(len(beats),-1,-1)
    continuous=beats-lane.last_row<=.5+1e-7
    row_runs=torch.where(continuous[:,None],lane.column_runs,0)
    runs=torch.where(current,row_runs[:,None]+1,0)
    last_hit=torch.where(current,beats[:,None,None],lane.last_hit[:,None])
    adjacent=(beats[:,None]-lane.last_hit-.25).abs()<=1e-6
    sixteenth=torch.where(current,torch.where(adjacent,lane.sixteenth_runs+1,1)[:,None],lane.sixteenth_runs[:,None])
    future_adjacent=(next_beats[:,None,None]-last_hit-.25).abs()<=1e-6
    future_runs=torch.maximum(runs,torch.where(future_adjacent,sixteenth,0))
    limit=torch.where(mini,2,4)
    blocked=(next_beats[:,None,None]-last_hit<.25-1e-7)|(future_runs>=limit[:,None,None])
    legal=~(blocked[:,:,None]&columns[None,None]).any(-1)
    hands=lane.mask_hands[None].expand(len(beats),-1,-1)
    hand_runs=torch.where(continuous[:,None],lane.hand_runs,0)
    future_hands=torch.where(hands,hand_runs[:,None]+1,0)
    blocked_hands=(future_hands>=5)&mini[:,None,None]
    legal &= ~(blocked_hands[:,:,None]&lane.mask_hands[None,None]).any(-1)
    nearing=(future_hands>=4).all(-1)&mini[:,None]
    legal &= ~(nearing[:,:,None]&lane.mask_hands.all(-1)[None,None])
    matching=legal&(lane.mask_counts[None,None]==next_counts[:,None,None])
    alternatives=matching.sum(-1).float()
    scores=.6*torch.log1p(alternatives)/torch.log(torch.tensor(7.,device=beats.device))
    # The observed forced two-hit exit problem is specific to short jacks.
    # Ordinary jacks keep their existing longer-chain policy in this trial.
    active=mini&(next_counts>0)&(next_beats-beats<=.5+1e-7)
    return torch.where(active[:,None],scores,0)
