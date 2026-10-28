"""Causal hand/column features shared by reference training and generation."""
import torch

from .segment_labels import LABELS
from .tiny_data import SLOTS


MOTION_WIDTH = 34


def candidate_grid(batch, length, device):
    positions = torch.arange(length, device=device)
    slots = torch.tensor(SLOTS, device=device, dtype=torch.float64)
    values = positions // len(SLOTS) + slots[positions % len(SLOTS)] / 48
    return values[None].expand(batch, -1)


def motion_features(previous, beats, plan, seconds_per_beat, seed=None):
    """Each position sees only masks strictly before that position, never its target."""
    batch, length = previous.shape
    if beats.shape != previous.shape or plan.shape[:2] != previous.shape:
        raise ValueError("Motion features must align with the causal decoder prefix.")
    masks = torch.cat((torch.zeros_like(previous[:, :1]), previous[:, 1:]), dim=1)
    history_beats = torch.cat((beats[:, :1] - 1e-4, beats[:, :-1]), dim=1)
    prefix = 0
    if seed is not None:
        seed_masks, seed_beats = seed
        prefix = seed_masks.shape[1]
        masks = torch.cat((seed_masks, masks), dim=1)
        history_beats = torch.cat((seed_beats, history_beats), dim=1)
    columns = (masks[..., None] & (1 << torch.arange(4, device=masks.device))) != 0
    occupied = masks != 0
    hands = torch.stack(((masks & 3) != 0, (masks & 12) != 0), dim=-1)
    absent = torch.full_like(history_beats, -10000)
    last_row = torch.where(occupied, history_beats, absent).cummax(dim=1).values
    before_row = torch.cat((absent[:, :1], last_row[:, :-1]), dim=1)
    continuous = history_beats - before_row <= 0.5 + 1e-7
    event_count = occupied.long().cumsum(dim=1)

    def participation_runs(hit):
        reset = occupied[..., None] & (~hit | ~continuous[..., None])
        base = torch.where(reset, event_count[..., None] - hit.long(), 0).cummax(dim=1).values
        return event_count[..., None] - base

    row_runs = participation_runs(columns)
    hand_runs = participation_runs(hands)
    last_column = torch.where(columns, history_beats[..., None], absent[..., None]).cummax(dim=1).values
    before_column = torch.cat((absent[:, :1, None].expand(-1, 1, 4), last_column[:, :-1]), dim=1)
    hits = columns.long().cumsum(dim=1)
    reset = columns & ((history_beats[..., None] - before_column - 0.25).abs() > 1e-6)
    base = torch.where(reset, hits - 1, 0).cummax(dim=1).values
    sixteenth_runs = hits - base
    selected = slice(prefix, prefix + length)
    elapsed = beats[..., None] - last_column[:, selected]
    seen = last_column[:, selected] > -9999
    row_gap = beats - last_row[:, selected]
    active = row_gap <= 0.5 + 1e-7
    row_runs = torch.where(active[..., None], row_runs[:, selected], 0)
    hand_runs = torch.where(active[..., None], hand_runs[:, selected], 0)
    sixteenth_runs = torch.where((elapsed - 0.25).abs() <= 1e-6, sixteenth_runs[:, selected], 0)
    lower = torch.searchsorted(history_beats.contiguous(), (beats - 2).contiguous())

    def recent_counts(hit):
        cumulative = torch.cat((torch.zeros(batch, 1, hit.shape[-1], device=previous.device), hit.float().cumsum(dim=1)), dim=1)
        before = cumulative.gather(1, lower[..., None].expand(-1, -1, hit.shape[-1]))
        return cumulative[:, prefix + 1:prefix + length + 1] - before

    recent_columns = recent_counts(columns)
    recent_hands = recent_counts(hands)
    recent_notes = torch.stack((recent_columns[..., :2].sum(dim=-1), recent_columns[..., 2:].sum(dim=-1)), dim=-1)
    ids = torch.where(plan.sum(dim=-1) > 0, plan.argmax(dim=-1), -1)
    mini = ids == LABELS.index('Minijacks')
    long = ids == LABELS.index('Longjacks')
    limit = torch.where(mini, 3, 8).float()[..., None]
    remaining_columns = ((limit - torch.maximum(row_runs, sixteenth_runs)) / limit).clamp(0, 1)
    remaining_columns = torch.where(long[..., None], 1, remaining_columns)
    remaining_hands = torch.where(mini[..., None], (1 - hand_runs.float() / 5).clamp(0, 1), 1)
    beat_duration = seconds_per_beat[:, :length, None]
    return torch.cat((torch.log1p(elapsed.clamp(0, 8)) / 2.197224577,
                      row_runs.float().clamp_max(32) / 8, sixteenth_runs.float().clamp_max(32) / 8,
                      hand_runs.float().clamp_max(32) / 5, recent_notes / 32, recent_hands / 8,
                      row_gap[..., None].clamp(0, 2) / 2, remaining_columns, remaining_hands,
                      seen.float(), (last_row[:, selected] > -9999).float()[..., None],
                      (elapsed * beat_duration).clamp(0, 2) / 2), dim=-1).float()


class MotionHistory:
    """Past candidates only; overlap is checked and never replayed as new notes."""
    def __init__(self):
        self.masks = None
        self.beats = None

    def before(self, beats, prefix):
        if self.beats is None:
            return None
        if self.beats.shape[0] != beats.shape[0] or self.beats.device != beats.device:
            raise ValueError("Motion history batch/device mismatch.")
        if (beats[:, 0] <= self.beats[:, -1]).any():
            count = prefix.shape[1] if prefix is not None else 0
            if count == 0 or not torch.allclose(beats[:, count - 1], self.beats[:, -1], atol=1e-7, rtol=0):
                raise ValueError("Motion history needs an aligned generated overlap prefix.")
            matches = (self.beats[:, :, None] - beats[:, None, :count]).abs() <= 1e-7
            if not matches.any(dim=1).all():
                raise ValueError("Motion prefix lies outside persisted history.")
            stored = self.masks.gather(1, matches.long().argmax(dim=1))
            if not torch.equal(stored, prefix):
                raise ValueError("Motion continuation cannot replace generated prefix notes.")
        keep = self.beats < beats[:, :1] - 1e-7
        if not torch.equal(keep, keep[:1].expand_as(keep)):
            raise ValueError("Persisted motion batches need matching overlap lengths.")
        return self.masks[:, keep[0]], self.beats[:, keep[0]]

    def update(self, masks, beats, seed):
        if seed is not None:
            masks = torch.cat((seed[0], masks), dim=1)
            beats = torch.cat((seed[1], beats), dim=1)
        self.masks = masks[:, -768:].detach().clone()
        self.beats = beats[:, -768:].detach().clone()
