from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from .segment_labels import LABELS


@dataclass(frozen=True)
class LaneControlConfig:
    minimum_gap_beats: float = 0.25
    continuity_gap_beats: float = 0.5
    soft_column_run: int = 6
    maximum_column_run: int = 12
    maximum_mask_run: int = 4
    column_run_penalty: float = 2.0
    mask_run_penalty: float = 2.0
    jack_overlap_reward: float = 2.0
    allow_longjacks: bool = False
    longjack_column_run: int = 32
    policy: str = "legacy"
    style_run_limits: tuple = ()

    def __post_init__(self):
        if self.policy not in ("legacy", "gap-only", "style-aware"):
            raise ValueError("Unknown lane constraint policy.")
        for label, soft, hard, masks in self.style_run_limits:
            if label not in LABELS or not 1 <= soft <= hard or masks < 1:
                raise ValueError("Invalid style-specific run limits.")
        values = (self.minimum_gap_beats, self.continuity_gap_beats, self.column_run_penalty, self.mask_run_penalty, self.jack_overlap_reward)
        if not all(math.isfinite(value) for value in values) or self.minimum_gap_beats <= 0 or self.continuity_gap_beats < self.minimum_gap_beats:
            raise ValueError("Lane timing limits must be finite and positive.")
        lengths = (self.soft_column_run, self.maximum_column_run, self.maximum_mask_run, self.longjack_column_run)
        if any(not isinstance(value, int) or value < 1 for value in lengths):
            raise ValueError("Lane run limits must be positive integers.")
        if self.soft_column_run > self.maximum_column_run or self.longjack_column_run < self.maximum_column_run:
            raise ValueError("Lane run limits must preserve soft, regular and longjack ordering.")
        if self.column_run_penalty < 0 or self.mask_run_penalty < 0 or self.jack_overlap_reward < 0:
            raise ValueError("Lane penalties cannot be negative.")


def absolute_beats(clock, times_ms):
    times = np.asarray(times_ms, dtype=np.float64)
    if not np.isfinite(times).all():
        raise ValueError("Candidate times must be finite.")
    offsets = np.asarray(clock.offsets, dtype=np.float64)
    lengths = np.asarray([section.beat_length_ms for section in clock.sections], dtype=np.float64)
    section_beats = np.concatenate(([0.0], np.cumsum(np.diff(offsets) / lengths[:-1])))
    sections = np.maximum(np.searchsorted(offsets, times, side="right") - 1, 0)
    return section_beats[sections] + (times - offsets[sections]) / lengths[sections]


class LaneControlState:
    def __init__(self, batch_size, device, config=LaneControlConfig()):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Lane state needs a positive batch size.")
        self.config = config
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.mask_columns = (torch.arange(16, device=device)[:, None] & (1 << torch.arange(4, device=device))[None]) != 0
        self.mask_counts = self.mask_columns.sum(dim=1)
        self.last_hit = torch.full((batch_size, 4), -torch.inf, dtype=torch.float64, device=device)
        self.device = self.last_hit.device
        self.last_row = torch.full((batch_size,), -torch.inf, dtype=torch.float64, device=device)
        self.last_candidate = self.last_row.clone()
        self.column_runs = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
        self.last_mask = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.mask_runs = torch.zeros_like(self.last_mask)
        self.maximum_column_runs = torch.zeros_like(self.column_runs)
        self.maximum_mask_runs = torch.zeros_like(self.last_mask)
        self.reduced_rows = torch.zeros_like(self.last_mask)
        self.dropped_rows = torch.zeros_like(self.last_mask)
        self.unplaced_notes = torch.zeros_like(self.last_mask)

    def validate_grid(self, candidate_beats, shape):
        if candidate_beats is None or candidate_beats.shape != shape or candidate_beats.device != self.device or not candidate_beats.is_floating_point():
            raise ValueError("Lane control needs real beat positions aligned with the complete grid and device.")
        if shape[0] != self.batch_size or not torch.isfinite(candidate_beats).all() or (torch.diff(candidate_beats, dim=1) <= 0).any():
            raise ValueError("Lane candidate beats must increase strictly within each batch item.")

    def constraints(self, beats, plan=None):
        if beats.shape != (self.batch_size,) or beats.device != self.device or not torch.isfinite(beats).all():
            raise ValueError("Current beat positions must align with the lane state.")
        if (beats <= self.last_candidate).any():
            raise ValueError("Lane state cannot replay generated candidates or move backwards.")
        continuous = beats - self.last_row <= self.config.continuity_gap_beats + 1e-7
        runs = torch.where(continuous[:, None], self.column_runs, torch.zeros_like(self.column_runs))
        longjack = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        if self.config.allow_longjacks and plan is not None:
            if plan.shape != (self.batch_size, len(LABELS)):
                raise ValueError("Current pattern plans must align with the lane state.")
            longjack = plan[:, LABELS.index("Longjacks")] >= 0.5
        limit = torch.full_like(self.last_mask, self.config.maximum_column_run)
        soft = torch.full_like(limit, self.config.soft_column_run)
        mask_limit = torch.full_like(limit, self.config.maximum_mask_run)
        jack_weight = torch.zeros(self.batch_size, device=self.device)
        stream_weight = torch.zeros_like(jack_weight)
        if plan is not None:
            if plan.shape != (self.batch_size, len(LABELS)):
                raise ValueError("Current pattern plans must align with the lane state.")
            if self.config.policy == "style-aware":
                active = plan.argmax(dim=-1)
                known = plan.sum(dim=-1) > 0
                for label, style_soft, style_hard, style_masks in self.config.style_run_limits:
                    selected = known & (active == LABELS.index(label))
                    soft = torch.where(selected, style_soft, soft)
                    limit = torch.where(selected, style_hard, limit)
                    mask_limit = torch.where(selected, style_masks, mask_limit)
                jack_labels = ("Jacks", "Chordjacks", "Minijacks", "Gluts")
                jack_weight = plan[:, [LABELS.index(label) for label in jack_labels]].max(dim=1).values.clamp(0, 1)
                stream_weight = plan[:, [LABELS.index("Jumpstream"), LABELS.index("Handstream")]].max(dim=1).values.clamp(0, 1)
        limit = torch.where(longjack, self.config.longjack_column_run, limit)
        cooling = beats[:, None] - self.last_hit < self.config.minimum_gap_beats - 1e-7
        blocked = cooling if self.config.policy == "gap-only" else cooling | (runs >= limit[:, None])
        legal = ~(blocked[:, None] & self.mask_columns[None]).any(dim=-1)
        repeated = continuous[:, None] & (torch.arange(16, device=self.device)[None] == self.last_mask[:, None]) & (self.last_mask[:, None] != 0)
        if self.config.policy != "gap-only":
            legal &= ~(repeated & (self.mask_runs[:, None] >= mask_limit[:, None]))
        excess = (runs - soft[:, None] + 1).clamp_min(0).to(torch.float32)
        excess = torch.where(longjack[:, None], torch.zeros_like(excess), excess)
        penalty = self.config.column_run_penalty * (excess.square() @ self.mask_columns.to(torch.float32).T)
        penalty += self.config.mask_run_penalty * repeated * (self.mask_runs[:, None] - 1).clamp_min(0)
        if self.config.policy == "gap-only":
            return legal, torch.zeros_like(penalty)
        if plan is not None:
            if plan.shape != (self.batch_size, len(LABELS)):
                raise ValueError("Current pattern plans must align with the lane state.")
            if self.config.policy == "legacy":
                jack_weight = plan[:, [LABELS.index("Jacks"), LABELS.index("Chordjacks")]].max(dim=1).values.clamp(0, 1)
            overlap = (self.mask_columns[self.last_mask][:, None] & self.mask_columns[None]).sum(dim=-1)
            if self.config.policy == "style-aware":
                penalty *= 1 - 0.75 * jack_weight[:, None]
                grid_adjacent = ((beats - self.last_row) - 0.25).abs() < 1e-6
                penalty += 1.5 * stream_weight[:, None] * grid_adjacent[:, None] * overlap
            penalty -= self.config.jack_overlap_reward * jack_weight[:, None] * continuous[:, None] * overlap
        return legal, penalty

    def prepare_counts(self, counts, beats, plan=None):
        if counts.shape != (self.batch_size,) or counts.dtype != torch.long or ((counts < 0) | (counts > 4)).any():
            raise ValueError("Lane control requires aligned counts between zero and four.")
        legal, _ = self.constraints(beats, plan)
        possible = legal & (self.mask_counts[None] <= counts[:, None])
        adjusted = self.mask_counts[None].expand(self.batch_size, -1).masked_fill(~possible, -1).max(dim=1).values
        self.reduced_rows += (adjusted < counts).to(torch.long)
        self.dropped_rows += ((counts > 0) & (adjusted == 0)).to(torch.long)
        self.unplaced_notes += counts - adjusted
        return adjusted

    def apply_logits(self, logits, beats, plan=None):
        legal, penalty = self.constraints(beats, plan)
        return (logits - penalty.to(logits.dtype)).masked_fill(~legal, -torch.inf)

    def advance(self, masks, beats):
        if masks.shape != (self.batch_size,) or masks.dtype != torch.long or ((masks < 0) | (masks > 15)).any():
            raise ValueError("Lane state updates require valid integer masks.")
        occupied = masks != 0
        continuous = beats - self.last_row <= self.config.continuity_gap_beats + 1e-7
        columns = self.mask_columns[masks]
        runs = torch.where(continuous[:, None], self.column_runs, torch.zeros_like(self.column_runs))
        updated_runs = torch.where(columns, runs + 1, torch.zeros_like(runs))
        self.column_runs = torch.where(occupied[:, None], updated_runs, self.column_runs)
        same = continuous & (masks == self.last_mask)
        self.mask_runs = torch.where(occupied, torch.where(same, self.mask_runs + 1, torch.ones_like(self.mask_runs)), self.mask_runs)
        self.last_mask = torch.where(occupied, masks, self.last_mask)
        self.last_hit = torch.where(columns, beats[:, None], self.last_hit)
        self.last_row = torch.where(occupied, beats, self.last_row)
        self.last_candidate = beats.clone()
        self.maximum_column_runs = torch.maximum(self.maximum_column_runs, self.column_runs)
        self.maximum_mask_runs = torch.maximum(self.maximum_mask_runs, self.mask_runs)

    def summary(self):
        return {
            "configuration": asdict(self.config),
            "maximum_column_runs": self.maximum_column_runs.cpu().tolist(),
            "maximum_mask_runs": self.maximum_mask_runs.cpu().tolist(),
            "count_reduced_rows": self.reduced_rows.cpu().tolist(),
            "dropped_rows": self.dropped_rows.cpu().tolist(),
            "unplaced_notes": self.unplaced_notes.cpu().tolist(),
        }
