import numpy as np
import torch
from torch import nn

from .main_style import STYLE_NAMES
from .spacing_control import SpacingAwareRiceGenerator
from .tiny_model import TinyModelConfig
from .tiny_data import SLOTS, TICKS_PER_BEAT


COUNT_ALLOWED = np.ones((len(STYLE_NAMES), 4), dtype=bool)
for style in ("Stream", "Rolls", "Trills", "Minitrills"):
    COUNT_ALLOWED[STYLE_NAMES.index(style), 1:] = False
for style in ("Jumpstream", "Jumptrill", "Split Trill"):
    COUNT_ALLOWED[STYLE_NAMES.index(style), 2:] = False
COUNT_ALLOWED[STYLE_NAMES.index("Handstream"), 3] = False
SIXTEENTH_POSITIONS = tuple(beat * len(SLOTS) + position for beat in range(8) for position, tick in enumerate(SLOTS) if tick % (TICKS_PER_BEAT // 4) == 0)


class MainStyleGenerator(SpacingAwareRiceGenerator):
    def __init__(self, config=TinyModelConfig()):
        super().__init__(config)
        self.style_encoder = nn.Sequential(nn.Linear(15, config.width), nn.GELU(), nn.Linear(config.width, config.width))
        self.onset_style_projection = nn.Linear(config.width, config.width, bias=False)
        self.style_onset = nn.Linear(config.width, 1)
        self.count_style_projection = nn.Linear(config.width, 4, bias=False)
        self.density_projection = nn.Linear(2, config.width, bias=False)
        self.register_buffer("style_count_allowed", torch.from_numpy(COUNT_ALLOWED), persistent=False)
        for module in (self.style_encoder, self.onset_style_projection, self.style_onset, self.count_style_projection, self.density_projection):
            for parameter in module.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                else:
                    nn.init.zeros_(parameter)
        self.spacing_projection = nn.Linear(16, config.width, bias=False)
        nn.init.xavier_uniform_(self.spacing_projection.weight)

    def condition_memory(self, memory, density=None):
        if density is None or density.shape != (*memory.shape[:2], 2) or not torch.isfinite(density).all() or (density < 0).any():
            raise ValueError("Main-style generation needs aligned nonnegative row/note density conditions.")
        return memory + self.density_projection(torch.log1p(density))

    def generation_onset_logits(self, memory, condition, plan):
        style = self.style_encoder(plan)
        hidden = memory + self.condition_projection(condition).unsqueeze(1) + self.onset_style_projection(style)
        return self.style_onset(nn.functional.gelu(hidden)).squeeze(-1)

    def decode_counts(self, memory, phase, condition, previous_counts, plan, count_plan_enabled):
        logits = super().decode_counts(memory, phase, condition, previous_counts, plan, count_plan_enabled)
        if count_plan_enabled:
            logits = logits + self.count_style_projection(self.style_encoder(plan[:, :previous_counts.shape[1]]))
        if not self.training and count_plan_enabled:
            selected = plan[:, :previous_counts.shape[1]]
            ids = torch.where(selected.sum(dim=-1) > 0, selected.argmax(dim=-1) + 1, 0)
            logits = logits.masked_fill(~self.style_count_allowed[ids], -torch.inf)
        return logits

    def timing_state(self, previous):
        batch, length = previous.shape
        positions = torch.arange(length, device=previous.device)
        grid = torch.as_tensor(SLOTS, dtype=torch.float32, device=previous.device)
        beats = (positions // len(SLOTS)).float() + grid[positions % len(SLOTS)] / TICKS_PER_BEAT
        history = torch.cat((torch.zeros_like(previous[:, :1]), previous[:, 1:]), dim=1)
        hit = (history[:, :, None] & (1 << torch.arange(4, device=previous.device))[None, None]) != 0
        history_beats = torch.cat((beats[:1] - 1, beats[:-1]))
        last = torch.where(hit, history_beats[None, :, None], torch.full((batch, length, 4), -1000.0, device=previous.device)).cummax(dim=1).values
        seen = last > -999
        elapsed = (beats[None, :, None] - last).clamp(0, 8)
        occupied = history != 0
        row_counts = occupied.cumsum(dim=1)[:, :, None]
        last_row = torch.where(occupied, history_beats[None], -1000.0).cummax(dim=1).values
        before_row = torch.cat((torch.full_like(last_row[:, :1], -1000), last_row[:, :-1]), dim=1)
        continuous = history_beats[None] - before_row <= 0.5 + 1e-5
        reset = occupied[:, :, None] & (~hit | ~continuous[:, :, None])
        bases = torch.where(reset, row_counts - hit.to(torch.long), torch.zeros_like(row_counts.expand(-1, -1, 4))).cummax(dim=1).values
        runs = (row_counts - bases).float().clamp(0, 32) / 32
        lower = torch.searchsorted(history_beats, history_beats - 2)
        cumulative = torch.cat((torch.zeros((batch, 1, 4), device=previous.device), hit.float().cumsum(dim=1)), dim=1)
        recent = (cumulative[:, 1:] - cumulative[:, lower]).clamp_max(16) / 16
        return torch.cat((torch.log1p(elapsed) / np.log(9), seen.float(), runs, recent), dim=-1)

    def decode_layout(self, memory, phase, condition, previous, plan, current_counts):
        length = previous.shape[1]
        style = self.style_encoder(plan[:, :length])
        hidden = self.note_embedding(previous) + self.current_count_embedding(current_counts)
        hidden = hidden + self.spacing_projection(self.timing_state(previous)) + self.position_embedding(torch.arange(length, device=phase.device))
        hidden = hidden + self.phase_projection(phase[:, :length]) + self.condition_projection(condition).unsqueeze(1) + self.plan_projection(plan[:, :length]) + style
        causal = torch.ones(length, length, device=phase.device, dtype=torch.bool).triu(1)
        raw = self.output(self.decoder(hidden, memory, tgt_mask=causal))
        legal = self.mask_counts[None, None] == current_counts[:, :, None]
        return raw.masked_fill(~legal, -torch.inf), raw

    def forward(self, acoustic, music, phase, condition, previous, current_counts, local=None, plan=None, count_plan_enabled=True, density=None):
        memory = self.condition_memory(self.encode_audio(acoustic, music, phase, local), density)
        quantity = self.decode_counts(memory, phase, condition, self.history_counts[previous], plan, count_plan_enabled)
        layout, raw = self.decode_layout(memory, phase, condition, previous, plan, current_counts)
        return {"onset_logits": self.generation_onset_logits(memory, condition, plan), "conditional_count_logits": quantity, "layout_logits": layout, "raw_layout_logits": raw, "current_counts": current_counts}

    def configuration(self):
        return {**super().configuration(), "architecture": "main-style-generator-v2-grid16", "spacing_features": 16, "style_vocabulary": list(STYLE_NAMES)}


def style_generator_loss(output, targets, plan, valid, onset_weight):
    counts = torch.as_tensor([mask.bit_count() for mask in range(16)], device=targets.device)[targets]
    valid = valid.bool()
    occupied = (counts > 0) & valid
    zero = output["raw_layout_logits"].sum() * 0
    onset_elements = nn.functional.binary_cross_entropy_with_logits(output["onset_logits"], (counts > 0).float(), pos_weight=onset_weight, reduction="none")
    onset = (onset_elements * valid).sum() / valid.sum().clamp_min(1)
    onset = onset / (2 * onset_weight / (1 + onset_weight))
    quantity = nn.functional.cross_entropy(output["conditional_count_logits"][occupied], counts[occupied] - 1) if occupied.any() else zero
    layout = nn.functional.cross_entropy(output["layout_logits"][occupied], targets[occupied]) if occupied.any() else zero
    probabilities = output["conditional_count_logits"].softmax(dim=-1)
    layout_prob = output["layout_logits"].softmax(dim=-1)
    columns = ((torch.arange(16, device=targets.device)[:, None] & (1 << torch.arange(4, device=targets.device))[None]) != 0).float()
    pressed_probability = layout_prob @ columns
    grid_positions = torch.as_tensor(SIXTEENTH_POSITIONS, device=targets.device)
    geometry_losses, histogram_losses = [], []
    for batch in range(len(targets)):
        for lower in (0, 96):
            upper = lower + 96
            style = int(plan[batch, lower].argmax()) + 1 if bool(plan[batch, lower].sum() > 0) else 0
            fixed_grid = style in (STYLE_NAMES.index("Jumpstream"), STYLE_NAMES.index("Handstream"))
            if fixed_grid:
                selected = grid_positions[(grid_positions >= lower) & (grid_positions < upper)]
                selected = selected[valid[batch, selected]]
                if not len(selected):
                    continue
                onset_probability = output["onset_logits"][batch, selected].sigmoid()
                joint = torch.cat(((1 - onset_probability)[:, None], onset_probability[:, None] * probabilities[batch, selected]), dim=-1)
                target_hist = nn.functional.one_hot(counts[batch, selected], 5).float().mean(dim=0)
                predicted_hist = joint.mean(dim=0)
                histogram_losses.append((target_hist * (target_hist.clamp_min(1e-6).log() - predicted_hist.clamp_min(1e-6).log())).sum())
                before, after = selected[:-1], selected[1:]
                adjacent = after - before == len(SLOTS) // 4
                before, after = before[adjacent], after[adjacent]
                if lower and bool(valid[batch, lower - len(SLOTS) // 4]) and bool(valid[batch, lower]):
                    before = torch.cat((torch.tensor([lower - len(SLOTS) // 4], device=targets.device), before))
                    after = torch.cat((torch.tensor([lower], device=targets.device), after))
                if not len(before):
                    continue
                expected_overlap = (pressed_probability[batch, before] * pressed_probability[batch, after]).sum(dim=-1) / 4
                expected_overlap = expected_overlap * output["onset_logits"][batch, before].sigmoid() * output["onset_logits"][batch, after].sigmoid()
                reference_overlap = (columns[targets[batch, before]] * columns[targets[batch, after]]).sum(dim=-1) / 4
                geometry_losses.append(nn.functional.smooth_l1_loss(expected_overlap, reference_overlap, beta=0.1))
                continue
            selected = torch.nonzero(occupied[batch, lower:upper], as_tuple=False).flatten() + lower
            if len(selected) < 2:
                continue
            target_hist = nn.functional.one_hot(counts[batch, selected] - 1, 4).float().mean(dim=0)
            predicted_hist = probabilities[batch, selected].mean(dim=0)
            histogram_losses.append((target_hist * (target_hist.clamp_min(1e-6).log() - predicted_hist.clamp_min(1e-6).log())).sum())
            before, after = selected[:-1], selected[1:]
            expected_overlap = (pressed_probability[batch, before] * pressed_probability[batch, after]).sum(dim=-1) / 4
            reference_overlap = (columns[targets[batch, before]] * columns[targets[batch, after]]).sum(dim=-1) / 4
            geometry_losses.append(nn.functional.smooth_l1_loss(expected_overlap, reference_overlap, beta=0.1))
    geometry = torch.stack(geometry_losses).mean() if geometry_losses else zero
    histogram = torch.stack(histogram_losses).mean() if histogram_losses else zero
    total = onset + quantity + layout + 0.2 * geometry + 0.2 * histogram
    return {"loss": total, "onset_loss": onset, "count_loss": quantity, "layout_loss": layout, "bar_overlap_loss": geometry, "bar_count_histogram_loss": histogram}
