"""Bounded hierarchical density residuals on frozen arrangement features."""
import numpy as np
import torch
from torch import nn


DENSITY_SCHEMA = "section-density-v2"


def density_context(memory, segment_ids):
    memory = np.asarray(memory, dtype=np.float32)
    section = memory.copy()
    for identifier in np.unique(segment_ids):
        if identifier >= 0:
            selected = segment_ids == identifier
            section[selected] = memory[selected].mean(axis=0)
    return section, memory.mean(axis=0)


class SectionDensityAdapter(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.width = width
        def head(inputs, outputs):
            result = nn.Sequential(nn.Linear(inputs, 64), nn.GELU(), nn.Linear(64, outputs))
            nn.init.zeros_(result[-1].weight)
            nn.init.zeros_(result[-1].bias)
            return result
        self.global_head = head(width, 2)
        self.section_head = head(width, 2)
        self.local_head = head(width, 2)
        self.beat_head = head(width + 26, 4)

    def forward(self, memory, section_memory, global_memory, base_density, beat_activity, quiet):
        delta = (0.75 * self.global_head(global_memory).tanh()
                 + 0.75 * self.section_head(section_memory).tanh()
                 + 0.4 * self.local_head(memory).tanh())
        raw = torch.expm1((torch.log1p(base_density) + delta).clamp(0, 4.55))
        rows = raw[..., 1].clamp(0, 12)
        notes = torch.minimum(torch.maximum(raw[..., 0], rows), 4 * rows).clamp_max(24)
        rows = torch.minimum(rows, notes)
        bar = torch.stack((notes, rows), dim=-1).masked_fill(quiet[..., None], 0)
        features = torch.cat((memory, torch.log1p(base_density), beat_activity.flatten(-2)), dim=-1)
        weights = self.beat_head(features).softmax(dim=-1)
        # Shared beat activity keeps row/note targets feasible and their mean
        # exactly equal to the bar target; it does not prescribe actual chords.
        beats = bar[..., None, :] * (4 * weights[..., :, None])
        beats = beats * (24 / beats[..., 1].clamp_min(1e-6)).clamp_max(1)[..., None]
        return {"bar_density": beats.mean(dim=-2), "beat_density": beats, "log_delta": delta}


def section_density_loss(output, batch, structured=True):
    valid = ~batch["padding_mask"]
    truth = batch["beat_targets"]
    target_bar = truth.mean(dim=-2)
    predicted = output["bar_density"]
    bar = nn.functional.smooth_l1_loss(torch.log1p(predicted[valid]), torch.log1p(target_bar[valid]))
    beat = nn.functional.smooth_l1_loss(torch.log1p(output["beat_density"][valid]), torch.log1p(truth[valid]))
    pairs = valid[:, 1:] & valid[:, :-1] & (batch["segment_ids"][:, 1:] >= 0)
    delta = nn.functional.smooth_l1_loss(torch.diff(torch.log1p(predicted), dim=1), torch.diff(torch.log1p(target_bar), dim=1), reduction="none").mean(dim=-1)
    adjacent = delta[pairs].mean() if pairs.any() else bar * 0
    section, gap, rank = [], [], []
    for ordinal in range(len(predicted)):
        ids = batch["segment_ids"][ordinal]
        ids = ids[valid[ordinal] & (ids >= 0)].unique()
        if not len(ids):
            continue
        membership = (batch["segment_ids"][ordinal, :, None] == ids[None]) & valid[ordinal, :, None]
        counts = membership.sum(dim=0).clamp_min(1)
        p = (predicted[ordinal, :, None, :] * membership[..., None]).sum(dim=0) / counts[:, None]
        t = (target_bar[ordinal, :, None, :] * membership[..., None]).sum(dim=0) / counts[:, None]
        p, t = torch.log1p(p), torch.log1p(t)
        section.append(nn.functional.smooth_l1_loss(p, t))
        if len(ids) > 1:
            difference_p, difference_t = p[:, None] - p[None], t[:, None] - t[None]
            selected = torch.triu(torch.ones(len(ids), len(ids), dtype=torch.bool, device=p.device), diagonal=1)
            gap.append(nn.functional.smooth_l1_loss(difference_p[selected], difference_t[selected]))
            strong = selected[..., None] & (difference_t.abs() > 0.15)
            if strong.any():
                rank.append(torch.relu(0.15 - difference_t.sign() * difference_p)[strong].mean())
    def average(values):
        return torch.stack(values).mean() if values else bar * 0
    section, gap, rank = average(section), average(gap), average(rank)
    base = bar + beat + 0.5 * adjacent
    loss = base + (0.75 * section + 0.75 * gap + 0.2 * rank if structured else 0)
    return {"loss": loss, "bar_loss": bar, "beat_loss": beat, "adjacent_loss": adjacent,
            "section_loss": section, "section_gap_loss": gap, "section_rank_loss": rank}
