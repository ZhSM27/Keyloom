from __future__ import annotations

import math

import torch
from torch import nn

from .factorized_model import FactorizedRiceGenerator
from .segment_labels import LABELS
from .tiny_model import TinyModelConfig


SPACING_FEATURES = 8
SPACING_CAP_SLOTS = 96
JACK_LABELS = ("Jacks", "Longjacks", "Chordjacks", "Minijacks")
JACK_INDICES = tuple(LABELS.index(label) for label in JACK_LABELS)

SPACING_OBJECTIVE = {
    "schema_version": "prefix-continuation-v2",
    "prefix_slots": [0, 24, 48, 72],
    "evaluation_prefix_slots": 48,
    "loss": "onset_count_layout_only_after_reference_prefix",
    "generation": "BOS_plus_prefix_then_autoregressive_target_with_spacing_repeat_control",
    "model_architecture": "spacing-aware-v1",
    "spacing_features": "per_column_log_elapsed_and_seen_8",
    "repeat_control": {
        "regular_short_gap_slots": 2,
        "regular_logit_penalty": 8.0,
        "jack_short_gap_slots": 1,
        "jack_logit_penalty": 3.0,
        "jack_labels": list(JACK_LABELS),
    },
}

PRODUCTION_REPEAT_CONTROL = {
    "regular_short_gap_slots": 2,
    "regular_logit_penalty": 32.0,
    "jack_short_gap_slots": 2,
    "jack_logit_penalty": 32.0,
    "reduce_forced_counts": True,
    "jack_labels": list(JACK_LABELS),
}


def spacing_state(previous: torch.Tensor, cap_slots: int = SPACING_CAP_SLOTS) -> tuple[torch.Tensor, torch.Tensor]:
    if previous.ndim != 2 or previous.dtype != torch.int64 or cap_slots < 1:
        raise ValueError("Spacing state requires an integer mask grid and a positive cap.")
    if ((previous < 0) | (previous > 16)).any():
        raise ValueError("Spacing history masks must stay in the 0..16 range.")
    batch, length = previous.shape
    if length == 0:
        raise ValueError("Spacing history must contain at least a BOS position.")
    columns = 1 << torch.arange(4, device=previous.device, dtype=torch.long)
    positions = torch.arange(length, device=previous.device)[None, :, None].expand(batch, length, 4)
    hit = (previous[:, :, None] & columns[None, None, :]) != 0
    last = torch.where(hit, positions, torch.full_like(positions, -1)).cummax(dim=1).values
    seen = last >= 0
    elapsed = torch.where(seen, positions - last + 1, torch.full_like(last, cap_slots)).clamp_max(cap_slots)
    return elapsed, seen


def spacing_features(previous: torch.Tensor, cap_slots: int = SPACING_CAP_SLOTS) -> torch.Tensor:
    elapsed, seen = spacing_state(previous, cap_slots)
    scaled = torch.log1p(elapsed.to(torch.float32)) / math.log1p(cap_slots)
    return torch.cat((scaled, seen.to(torch.float32)), dim=-1)


class SpacingAwareRiceGenerator(FactorizedRiceGenerator):
    def __init__(self, config: TinyModelConfig = TinyModelConfig()):
        super().__init__(config)
        self.spacing_projection = nn.Linear(SPACING_FEATURES, config.width, bias=False)
        nn.init.xavier_uniform_(self.spacing_projection.weight)

    def decode_layout(self, memory, phase, condition, previous, plan, current_counts):
        length = previous.shape[1]
        if current_counts.shape != previous.shape or current_counts.dtype != torch.int64 or ((current_counts < 0) | (current_counts > 4)).any():
            raise ValueError("Every layout position requires an explicit integer count from zero through four.")
        positions = torch.arange(length, device=phase.device)
        hidden = (
            self.note_embedding(previous)
            + self.current_count_embedding(current_counts)
            + self.spacing_projection(spacing_features(previous))
            + self.position_embedding(positions)
            + self.phase_projection(phase[:, :length])
            + self.condition_projection(condition).unsqueeze(1)
            + self.plan_projection(plan[:, :length])
        )
        causal = torch.ones(length, length, device=phase.device, dtype=torch.bool).triu(1)
        raw = self.output(self.decoder(hidden, memory, tgt_mask=causal))
        legal = self.mask_counts[None, None, :] == current_counts[:, :, None]
        return raw.masked_fill(~legal, -torch.inf), raw

    def configuration(self):
        return {
            **super().configuration(),
            "architecture": "spacing-aware-v1",
            "spacing_features": SPACING_FEATURES,
            "spacing_cap_slots": SPACING_CAP_SLOTS,
        }


def repeat_penalty_matrix(previous, plan, position, control=None):
    if previous.ndim != 2 or previous.shape[1] == 0:
        raise ValueError("Repeat control requires a nonempty autoregressive history.")
    control = control or SPACING_OBJECTIVE["repeat_control"]
    if plan is None:
        jack = torch.zeros(previous.shape[0], dtype=torch.bool, device=previous.device)
    else:
        if plan.ndim != 3 or plan.shape[0] != previous.shape[0] or plan.shape[2] != len(LABELS) or not 0 <= position < plan.shape[1]:
            raise ValueError("Repeat control plans must align with the current candidate grid.")
        jack = plan[:, position, list(JACK_INDICES)].to(torch.bool).any(dim=-1)
    columns = 1 << torch.arange(4, device=previous.device, dtype=torch.long)
    regular_gap = int(control["regular_short_gap_slots"])
    jack_gap = int(control["jack_short_gap_slots"])
    regular_strength = float(control["regular_logit_penalty"])
    jack_strength = float(control["jack_logit_penalty"])
    if regular_gap < 1 or jack_gap < 1 or jack_gap > regular_gap:
        raise ValueError("Repeat-control gaps must be positive and the jack gap cannot exceed the regular gap.")
    recent = previous[:, -regular_gap:]
    if recent.shape[1] < regular_gap:
        recent = torch.cat((torch.zeros((previous.shape[0], regular_gap - recent.shape[1]), dtype=previous.dtype, device=previous.device), recent), dim=1)
    hit = (recent[:, :, None] & columns[None, None, :]) != 0
    gaps = torch.full((previous.shape[0], 4), regular_gap + 1, device=previous.device, dtype=torch.long)
    for offset in range(regular_gap):
        gaps = torch.where(hit[:, regular_gap - 1 - offset] & (gaps > offset + 1), torch.full_like(gaps, offset + 1), gaps)
    threshold = torch.where(jack[:, None], torch.full_like(gaps, jack_gap), torch.full_like(gaps, regular_gap))
    strength = torch.where(jack, torch.full((len(jack),), jack_strength, device=previous.device), torch.full((len(jack),), regular_strength, device=previous.device))
    column_penalty = (gaps <= threshold) * strength[:, None]
    masks = torch.arange(16, device=previous.device, dtype=torch.long)
    mask_columns = (masks[:, None] & columns[None, :]) != 0
    return column_penalty @ mask_columns.to(column_penalty.dtype).T


def reduce_forced_counts(previous, plan, counts, mask_counts, control=None):
    if counts.ndim != 1 or counts.dtype != torch.int64 or mask_counts.ndim != 1:
        raise ValueError("Forced-repeat count control requires aligned integer counts.")
    control = control or SPACING_OBJECTIVE["repeat_control"]
    adjusted = counts.clone()
    for _ in range(3):
        penalty = repeat_penalty_matrix(previous, plan, previous.shape[1] - 1, control)
        legal = mask_counts[None, :] == adjusted[:, None]
        has_clear_choice = (legal & (penalty == 0)).any(dim=1)
        reducible = (~has_clear_choice) & (adjusted > 1)
        if not reducible.any():
            break
        adjusted = torch.where(reducible, adjusted - 1, adjusted)
    return adjusted
