from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ArrangementPlannerConfig:
    width: int = 256
    heads: int = 4
    layers: int = 6
    feedforward: int = 512
    dropout: float = 0.1
    max_bars: int = 1024
    condition_width: int = 17
    phase_width: int = 7


class ArrangementPlanner(nn.Module):
    def __init__(self, config: ArrangementPlannerConfig = ArrangementPlannerConfig()):
        super().__init__()
        self.config = config
        self.music_projection = nn.Linear(1024, config.width)
        self.acoustic_projection = nn.Linear(128, config.width)
        self.rhythm_projection = nn.Linear(40, config.width)
        self.phase_projection = nn.Linear(config.phase_width, config.width)
        self.condition_projection = nn.Linear(config.condition_width, config.width, bias=False)
        self.input_norm = nn.LayerNorm(config.width)
        layer = nn.TransformerEncoderLayer(
            config.width,
            config.heads,
            config.feedforward,
            config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers, norm=nn.LayerNorm(config.width), enable_nested_tensor=False)
        self.density_head = nn.Linear(config.width, 2)
        self.chord_head = nn.Linear(config.width, 4)
        self.pattern_head = nn.Linear(config.width, 15)
        for module in (
            self.music_projection,
            self.acoustic_projection,
            self.rhythm_projection,
            self.phase_projection,
            self.condition_projection,
            self.density_head,
            self.chord_head,
            self.pattern_head,
        ):
            if module.weight.dim() > 1:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        position = torch.arange(config.max_bars, dtype=torch.float32)[:, None]
        divisor = torch.exp(torch.arange(0, config.width, 2, dtype=torch.float32) * (-math.log(10000.0) / config.width))
        encoding = torch.zeros(config.max_bars, config.width)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("position_encoding", encoding, persistent=False)

    def forward(self, music, acoustic, rhythm, phase, condition, padding_mask, return_memory=False):
        length = music.shape[1]
        if length > self.config.max_bars or any(values.shape[:2] != (music.shape[0], length) for values in (acoustic, rhythm, phase, condition, padding_mask)):
            raise ValueError("Arrangement inputs must share a supported bar sequence.")
        positions = torch.arange(length, device=music.device)
        hidden = (
            self.music_projection(music)
            + self.acoustic_projection(acoustic)
            + self.rhythm_projection(rhythm)
            + self.phase_projection(phase)
            + self.condition_projection(condition)
            + self.position_encoding[:length][None]
        )
        hidden = self.input_norm(hidden)
        memory = self.encoder(hidden, src_key_padding_mask=padding_mask)
        result = {
            "density_log": self.density_head(memory),
            "chord_logits": self.chord_head(memory),
            "pattern_logits": self.pattern_head(memory),
        }
        if return_memory:
            result["memory"] = memory
        return result

    def configuration(self):
        return asdict(self.config)


def planner_loss(output, batch, weights):
    mask = ~batch["padding_mask"]
    density_target = torch.log1p(batch["notes_per_beat"])
    rows_target = torch.log1p(batch["rows_per_beat"])
    density = nn.functional.smooth_l1_loss(output["density_log"][..., 0][mask], density_target[mask])
    rows = nn.functional.smooth_l1_loss(output["density_log"][..., 1][mask], rows_target[mask])
    has_rows = mask & (batch["rows"] > 0)
    if has_rows.any():
        chord = -(batch["chord"][has_rows] * torch.log_softmax(output["chord_logits"][has_rows], dim=-1)).sum(-1).mean()
    else:
        chord = output["chord_logits"].sum() * 0
    available = mask & batch["pattern_available"]
    positive = available[..., None] & (batch["pattern"] > 0)
    if positive.any():
        pattern_positive = nn.functional.binary_cross_entropy_with_logits(output["pattern_logits"][positive], batch["pattern"][positive])
    else:
        pattern_positive = output["pattern_logits"].sum() * 0
    if available.any():
        chart_counts = available.sum(dim=1).clamp_min(1)
        predicted_mean = (output["pattern_logits"].sigmoid() * available[..., None]).sum(dim=1) / chart_counts[:, None]
        target_mean = (batch["pattern"] * available[..., None]).sum(dim=1) / chart_counts[:, None]
        pattern_mean = nn.functional.l1_loss(predicted_mean, target_mean)
    else:
        pattern_mean = output["pattern_logits"].sum() * 0
    adjacent_density, adjacent_pattern = structure_adjacent_loss(output, batch, mask)
    repeat_density, repeat_pattern = repeated_section_loss(output, batch, mask)
    quiet_density, quiet_rows = quiet_energy_loss(output, batch, mask, density_target, rows_target)
    total = (
        density
        + rows
        + chord
        + weights["pattern_positive"] * pattern_positive
        + weights["pattern_mean"] * pattern_mean
        + weights.get("structure_adjacent", 0.0) * (adjacent_density + adjacent_pattern)
        + weights.get("structure_repeat", 0.0) * (repeat_density + repeat_pattern)
        + weights.get("quiet_energy", 0.0) * (quiet_density + quiet_rows)
    )
    return {
        "loss": total,
        "density_loss": density,
        "rows_loss": rows,
        "chord_loss": chord,
        "pattern_positive_loss": pattern_positive,
        "pattern_mean_loss": pattern_mean,
        "structure_adjacent_density_loss": adjacent_density,
        "structure_adjacent_pattern_loss": adjacent_pattern,
        "structure_repeat_density_loss": repeat_density,
        "structure_repeat_pattern_loss": repeat_pattern,
        "quiet_density_loss": quiet_density,
        "quiet_rows_loss": quiet_rows,
    }


def _time_features(batch, mask):
    values = torch.cat((batch["acoustic"], batch["rhythm"]), dim=-1)
    mask_values = mask.to(values.dtype).unsqueeze(-1)
    count = mask_values.sum(dim=1, keepdim=True).clamp_min(1)
    mean = (values * mask_values).sum(dim=1, keepdim=True) / count
    values = (values - mean) * mask_values
    return nn.functional.normalize(values, dim=-1)


def structure_adjacent_loss(output, batch, mask):
    if output["density_log"].shape[1] < 2:
        zero = output["density_log"].sum() * 0
        return zero, zero
    features = _time_features(batch, mask)
    similarity = (features[:, 1:] * features[:, :-1]).sum(dim=-1)
    section_start = batch.get("is_section_start")
    if section_start is None:
        section_start = torch.zeros_like(mask)
    valid = mask[:, 1:] & mask[:, :-1] & ~section_start[:, 1:]
    stable = valid & (similarity > 0.85)
    density_delta = (output["density_log"][:, 1:] - output["density_log"][:, :-1]).abs().mean(dim=-1)
    pattern_probability = output["pattern_logits"].sigmoid()
    pattern_delta = (pattern_probability[:, 1:] - pattern_probability[:, :-1]).abs().mean(dim=-1)
    pattern_valid = stable & batch["pattern_available"][:, 1:] & batch["pattern_available"][:, :-1]

    def weighted(values, selected):
        return (values * selected.to(values.dtype)).sum() / selected.sum().clamp_min(1)

    return weighted(density_delta, stable), weighted(pattern_delta, pattern_valid)


def repeated_section_loss(output, batch, mask, stride=8, minimum_distance=4):
    length = output["density_log"].shape[1]
    zero = output["density_log"].sum() * 0
    if length < stride + minimum_distance:
        return zero, zero
    features = _time_features(batch, mask)
    anchors = torch.arange(0, length, stride, device=mask.device)
    anchors = anchors[mask[:, anchors].any(dim=0)]
    if not len(anchors):
        return zero, zero
    similarity = torch.matmul(features[:, anchors], features.transpose(1, 2))
    positions = torch.arange(length, device=mask.device)
    distance_ok = (positions[None, :] - anchors[:, None]).abs() >= minimum_distance
    pair_valid = mask[:, anchors].unsqueeze(-1) & mask.unsqueeze(1) & distance_ok.unsqueeze(0)
    pair_weight = ((similarity - 0.9) / 0.1).clamp(0, 1) * pair_valid.to(similarity.dtype)
    density_delta = (output["density_log"][:, anchors].unsqueeze(2) - output["density_log"].unsqueeze(1)).abs().mean(dim=-1)
    pattern_probability = output["pattern_logits"].sigmoid()
    pattern_delta = (pattern_probability[:, anchors].unsqueeze(2) - pattern_probability.unsqueeze(1)).abs().mean(dim=-1)
    pattern_weight = pair_weight * batch["pattern_available"][:, anchors].unsqueeze(-1) * batch["pattern_available"].unsqueeze(1)

    def weighted(values, selected):
        return (values * selected).sum() / selected.sum().clamp_min(1)

    return weighted(density_delta, pair_weight), weighted(pattern_delta, pattern_weight)


def quiet_energy_loss(output, batch, mask, density_target, rows_target):
    energy = batch["acoustic"].mean(dim=-1)
    activity = batch.get("event_signal", batch["rhythm"][..., 32:].mean(dim=-1))
    valid_energy = energy.masked_fill(~mask, 0)
    valid_activity = activity.masked_fill(~mask, 0)
    counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
    energy_mean = valid_energy.sum(dim=1, keepdim=True) / counts
    activity_mean = valid_activity.sum(dim=1, keepdim=True) / counts
    energy_centered = (energy - energy_mean) * mask.to(energy.dtype)
    activity_centered = (activity - activity_mean) * mask.to(activity.dtype)
    energy_std = torch.sqrt((energy_centered.square().sum(dim=1, keepdim=True) / counts).clamp_min(1e-4))
    activity_std = torch.sqrt((activity_centered.square().sum(dim=1, keepdim=True) / counts).clamp_min(1e-4))
    low_energy = energy < energy_mean - 0.75 * energy_std
    eventful = activity > activity_mean + 0.25 * activity_std
    quiet = mask & low_energy & ~eventful
    if not quiet.any():
        zero = output["density_log"].sum() * 0
        return zero, zero
    density_error = torch.relu(output["density_log"][..., 0] - density_target)
    rows_error = torch.relu(output["density_log"][..., 1] - rows_target)

    def weighted(values):
        return (values * quiet.to(values.dtype)).sum() / quiet.sum().clamp_min(1)

    return weighted(density_error), weighted(rows_error)
