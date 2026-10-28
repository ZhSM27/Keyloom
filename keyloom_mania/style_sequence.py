import numpy as np
import torch
from torch import nn

from .main_style import STYLE_NAMES
from .style_generator import MainStyleGenerator
from .style_rhythm import RHYTHM_GRIDS, joint_mask_probability
from .tiny_data import SLOTS, TICKS_PER_BEAT
from .tiny_model import TinyModelConfig


class StyleSequenceGenerator(MainStyleGenerator):
    def __init__(self, config=TinyModelConfig(), rhythm_priors=None):
        super().__init__(config)
        self.rhythm_head = nn.Sequential(nn.Linear(config.width, config.width), nn.GELU(), nn.Linear(config.width, 4))
        self.rhythm_history = nn.Linear(5, config.width, bias=False)
        priors = np.full((len(STYLE_NAMES), 4), 0.25, dtype=np.float32) if rhythm_priors is None else np.asarray(rhythm_priors, dtype=np.float32)
        if priors.shape != (len(STYLE_NAMES), 4) or not np.isfinite(priors).all() or (priors <= 0).any() or not np.allclose(priors.sum(axis=-1), 1):
            raise ValueError("Rhythm priors must be positive style-conditioned probabilities.")
        self.register_buffer("rhythm_log_priors", torch.from_numpy(np.log(priors)))
        self.register_buffer("rhythm_grids", torch.from_numpy(RHYTHM_GRIDS.astype(np.float32)), persistent=False)
        for module in (self.rhythm_head, self.rhythm_history):
            for parameter in module.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                else:
                    nn.init.zeros_(parameter)

    def rhythm_logits(self, memory, condition, plan):
        batch, length, width = memory.shape
        if length % len(SLOTS):
            raise ValueError("Rhythm decoding requires complete beat grids.")
        style = self.style_encoder(plan)
        hidden = memory + style + self.condition_projection(condition)[:, None]
        hidden = hidden.reshape(batch, -1, len(SLOTS), width).mean(dim=2)
        beat_plan = plan[:, ::len(SLOTS)]
        ids = torch.where(beat_plan.sum(dim=-1) > 0, beat_plan.argmax(dim=-1) + 1, 0)
        return self.rhythm_head(hidden) + 0.35 * self.rhythm_log_priors[ids]

    def density_condition(self, density, plan):
        if density is None:
            return density
        regular = torch.zeros(plan.shape[:2], dtype=torch.bool, device=plan.device)
        for label in ("Jacks", "Chordjacks", "Minijacks", "Gluts", "Jumpstream", "Handstream"):
            regular |= plan[..., STYLE_NAMES.index(label) - 1] > 0.5
        rows = torch.where(regular, density[..., 1].clamp_max(4), density[..., 1])
        maximum_count = torch.full_like(rows, 4)
        maximum_count = torch.where(plan[..., STYLE_NAMES.index("Jumpstream") - 1] > 0.5, 2, maximum_count)
        maximum_count = torch.where(plan[..., STYLE_NAMES.index("Handstream") - 1] > 0.5, torch.where(rows >= 3.5, 2, 3), maximum_count)
        notes = torch.minimum(density[..., 0], rows * maximum_count)
        feasible = torch.stack((notes, rows), dim=-1)
        return feasible + 0.1 * (density - feasible)

    @torch.inference_mode()
    def generate(self, *args, density=None, plan=None, **kwargs):
        return super().generate(*args, density=self.density_condition(density, plan), plan=plan, **kwargs)

    def generation_onset_logits(self, memory, condition, plan):
        rhythm = self.rhythm_logits(memory, condition, plan)
        allowed = rhythm.softmax(dim=-1) @ self.rhythm_grids
        penalty = allowed.clamp_min(0.02).log().reshape(memory.shape[:2])
        return super().generation_onset_logits(memory, condition, plan) + penalty

    def decode_layout(self, memory, phase, condition, previous, plan, current_counts):
        length = previous.shape[1]
        style = self.style_encoder(plan[:, :length])
        hidden = self.note_embedding(previous) + self.spacing_projection(self.timing_state(previous))
        hidden = hidden + self.position_embedding(torch.arange(length, device=phase.device))
        hidden = hidden + self.phase_projection(phase[:, :length]) + self.condition_projection(condition)[:, None]
        hidden = hidden + self.plan_projection(plan[:, :length]) + style
        causal = torch.ones(length, length, device=phase.device, dtype=torch.bool).triu(1)
        raw = self.output(self.decoder(hidden, memory, tgt_mask=causal))
        legal = self.mask_counts[None, None] == current_counts[:, :, None]
        return raw.masked_fill(~legal, -torch.inf), raw

    def decode_counts(self, memory, phase, condition, previous_counts, plan, count_plan_enabled):
        length = previous_counts.shape[1]
        occupied = (previous_counts > 0) & (previous_counts < 5)
        positions = torch.arange(length, device=phase.device)
        last = torch.where(occupied, positions[None], -192).cummax(dim=1).values
        distance = (positions[None] - last).clamp(0, 192).float() / 192
        features = torch.cat((nn.functional.one_hot(previous_counts.clamp_max(4), 5)[..., 1:].float(), distance[..., None]), dim=-1)
        history = self.rhythm_history(features)
        selected_plan = plan[:, :length] if count_plan_enabled else torch.zeros_like(plan[:, :length])
        hidden = self.count_history_embedding(previous_counts) + self.position_embedding(positions)
        hidden = hidden + self.phase_projection(phase[:, :length]) + self.condition_projection(condition)[:, None]
        hidden = hidden + self.count_plan_projection(selected_plan) + history
        causal = torch.ones(length, length, device=phase.device, dtype=torch.bool).triu(1)
        logits = self.conditional_count_head(self.count_decoder(hidden, memory, tgt_mask=causal))
        if count_plan_enabled:
            logits = logits + self.count_style_projection(self.style_encoder(selected_plan))
        if not self.training and count_plan_enabled:
            ids = torch.where(selected_plan.sum(dim=-1) > 0, selected_plan.argmax(dim=-1) + 1, 0)
            logits = logits.masked_fill(~self.style_count_allowed[ids], -torch.inf)
        return logits

    def forward(self, acoustic, music, phase, condition, previous, current_counts, local=None, plan=None, count_plan_enabled=True, density=None):
        memory = self.condition_memory(self.encode_audio(acoustic, music, phase, local), self.density_condition(density, plan))
        quantity = self.decode_counts(memory, phase, condition, self.history_counts[previous], plan, count_plan_enabled)
        layout, raw = self.decode_layout(memory, phase, condition, previous, plan, current_counts)
        return {"onset_logits": self.generation_onset_logits(memory, condition, plan), "rhythm_logits": self.rhythm_logits(memory, condition, plan),
                "conditional_count_logits": quantity, "layout_logits": layout, "raw_layout_logits": raw, "current_counts": current_counts}

    def configuration(self):
        return {**super().configuration(), "architecture": "style-sequence-v3", "rhythm_families": 4, "layout_current_count_leakage": False}


def sequence_loss(output, targets, plan, valid, onset_weight, structural_valid=None, rhythm_labels=None):
    valid = valid.bool()
    structural_valid = valid if structural_valid is None else structural_valid.bool() & valid
    counts_table = torch.as_tensor([mask.bit_count() for mask in range(16)], device=targets.device)
    counts = counts_table[targets]
    occupied = (counts > 0) & valid
    zero = output["raw_layout_logits"].sum() * 0
    onset_elements = nn.functional.binary_cross_entropy_with_logits(output["onset_logits"], (counts > 0).float(), pos_weight=onset_weight, reduction="none")
    onset = (onset_elements * valid).sum() / valid.sum().clamp_min(1) / (2 * onset_weight / (1 + onset_weight))
    weight = torch.where(counts[occupied] == 3, 1.5, 1.0)
    quantity = (nn.functional.cross_entropy(output["conditional_count_logits"][occupied], counts[occupied] - 1, reduction="none") * weight).sum() / weight.sum().clamp_min(1) if occupied.any() else zero
    layout = nn.functional.cross_entropy(output["layout_logits"][occupied], targets[occupied]) if occupied.any() else zero
    beat_valid = structural_valid.reshape(len(targets), -1, len(SLOTS)).all(dim=-1) & (targets != 0).reshape(len(targets), -1, len(SLOTS)).any(dim=-1)
    if rhythm_labels is None:
        grid = torch.as_tensor(RHYTHM_GRIDS, device=targets.device)
        hit = (targets != 0).reshape(len(targets), -1, len(SLOTS))
        rhythm_labels = (~hit[:, :, None] | grid).all(dim=-1).long().argmax(dim=-1)
    rhythm = nn.functional.cross_entropy(output["rhythm_logits"][beat_valid], rhythm_labels[beat_valid]) if beat_valid.any() else zero
    joint = joint_mask_probability(output)
    grid_positions = torch.as_tensor([beat * len(SLOTS) + offset for beat in range(8) for offset, tick in enumerate(SLOTS) if tick % (TICKS_PER_BEAT // 4) == 0], device=targets.device)
    probability = joint[:, grid_positions]
    reference = targets[:, grid_positions]
    selected_valid = structural_valid[:, grid_positions]
    styles = plan[:, grid_positions].argmax(dim=-1) + 1
    active = plan[:, grid_positions].sum(dim=-1) > 0
    stream = active & ((styles == STYLE_NAMES.index("Jumpstream")) | (styles == STYLE_NAMES.index("Handstream")))
    columns = ((torch.arange(16, device=targets.device)[:, None] & (1 << torch.arange(4, device=targets.device))[None]) != 0).float()
    expected = probability @ columns
    truth = columns[reference]
    pair_valid = selected_valid[:, :-1] & selected_valid[:, 1:]
    overlap_expected = (expected[:, :-1] * expected[:, 1:]).sum(dim=-1) / 4
    overlap_reference = (truth[:, :-1] * truth[:, 1:]).sum(dim=-1) / 4
    geometry_values = nn.functional.smooth_l1_loss(overlap_expected, overlap_reference, beta=0.1, reduction="none")
    geometry = geometry_values[pair_valid].mean() if pair_valid.any() else zero
    row_counts = counts_table[reference]
    no_overlap = (torch.arange(16, device=targets.device)[:, None] & torch.arange(16, device=targets.device)[None]) == 0
    transition = no_overlap.to(probability.dtype)
    windows = probability.unfold(1, 4, 1).permute(0, 1, 3, 2)
    ref_windows = reference.unfold(1, 4, 1)
    count_windows = row_counts.unfold(1, 4, 1)
    windows_valid = selected_valid.unfold(1, 4, 1).all(dim=-1) & stream.unfold(1, 4, 1).all(dim=-1)
    same_style = (styles.unfold(1, 4, 1) == styles[:, :-3, None]).all(dim=-1)
    matched = (count_windows > 0).all(dim=-1) & ((ref_windows[..., :-1] & ref_windows[..., 1:]) == 0).all(dim=-1)
    hs = styles[:, :-3] == STYLE_NAMES.index("Handstream")
    js = styles[:, :-3] == STYLE_NAMES.index("Jumpstream")
    matched &= (hs & (count_windows[..., 0] == 3)) | (js & (count_windows[..., 0] == 2) & (count_windows[..., 1] == 1) & (count_windows[..., 2:] <= 2).all(dim=-1))
    positive = windows_valid & same_style & matched
    first_count = torch.where(hs, 3, 2)
    state = windows[:, :, 0] * (counts_table == first_count[..., None])
    allowed = counts_table > 0
    for offset in range(1, 4):
        row_allowed = allowed[None, None].expand_as(state)
        js_allowed = counts_table == 1 if offset == 1 else (counts_table > 0) & (counts_table <= 2)
        row_allowed = row_allowed & (~js[..., None] | js_allowed)
        state = (state @ transition) * windows[:, :, offset] * row_allowed
    motif = -state.sum(dim=-1).clamp_min(1e-6).log()[positive].mean() if positive.any() else zero
    predicted_hist = probability.reshape(len(targets), 2, 16, 16).mean(dim=2)
    count_bins = nn.functional.one_hot(counts_table, 5).float()
    predicted_hist = predicted_hist @ count_bins
    target_hist = nn.functional.one_hot(row_counts.reshape(len(targets), 2, 16), 5).float().mean(dim=2)
    bars_valid = selected_valid.reshape(len(targets), 2, 16).all(dim=-1)
    histogram_values = (target_hist * (target_hist.clamp_min(1e-6).log() - predicted_hist.clamp_min(1e-6).log())).sum(dim=-1)
    histogram = histogram_values[bars_valid].mean() if bars_valid.any() else zero
    loss = onset + quantity + layout + 0.25 * rhythm + 0.2 * histogram + 0.3 * geometry + 0.15 * motif
    return {"loss": loss, "onset_loss": onset, "count_loss": quantity, "layout_loss": layout, "rhythm_loss": rhythm,
            "bar_overlap_loss": geometry, "bar_count_histogram_loss": histogram, "sequence_motif_loss": motif}
