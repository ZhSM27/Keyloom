import numpy as np
import torch
from torch import nn

from .main_style import STYLE_NAMES
from .phrase_generator import PhraseGenerator, phrase_generator_loss
from .style_generator import MainStyleGenerator
from .style_rhythm import REGULAR_STYLES, joint_mask_probability
from .tiny_data import SLOTS
from .tiny_model import TinyModelConfig


class CadenceGenerator(PhraseGenerator):
    def __init__(self, config=TinyModelConfig(), rhythm_priors=None, count_weights=None):
        super().__init__(config, rhythm_priors)
        weights = np.ones((len(STYLE_NAMES), 4), dtype=np.float32) if count_weights is None else np.asarray(count_weights, dtype=np.float32)
        if weights.shape != (len(STYLE_NAMES), 4) or not np.isfinite(weights).all() or np.any((weights < 1) | (weights > 3)):
            raise ValueError("Cadence count weights must be finite style-conditioned values in 1..3.")
        self.register_buffer("count_class_weights", torch.from_numpy(weights))

    def generation_onset_logits(self, memory, condition, plan):
        logits = MainStyleGenerator.generation_onset_logits(self, memory, condition, plan)
        rhythm = self.rhythm_logits(memory, condition, plan)
        probabilities = rhythm.softmax(dim=-1)
        support = (probabilities @ self.rhythm_grids).clamp_min(0.01).log().reshape(logits.shape)
        if self.rhythm_decoding == "soft":
            return logits + support
        if self.rhythm_decoding != "learned-grid" or self.training:
            raise ValueError("Discrete cadence decoding is an inference-only policy.")
        confidence, family = probabilities.max(dim=-1)
        regular = plan[:, ::len(SLOTS), [STYLE_NAMES.index(label) - 1 for label in REGULAR_STYLES]].max(dim=-1).values > 0.5
        threshold = torch.where(regular, 0.95, 0.80)
        family = torch.where((confidence >= threshold) & (family < 3), family, 0)
        allowed = self.rhythm_grids[family].bool().reshape(logits.shape)
        return (logits + support).masked_fill(~allowed, -torch.inf)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        output["count_class_weights"] = self.count_class_weights
        return output

    def configuration(self):
        return {**super().configuration(), "architecture": "cadence-generator-v5",
                "rhythm_policy": "All styles use one learned beat grid; fine family requires confidence 0.80, regular styles 0.95; mixed defaults to sixteenth."}


def cadence_generator_loss(output, targets, plan, valid, onset_weight, structural_valid, rhythm_labels):
    losses = phrase_generator_loss(output, targets, plan, valid, onset_weight, structural_valid, rhythm_labels)
    valid = valid.bool()
    reliable = structural_valid.bool() & valid
    counts_table = torch.as_tensor([mask.bit_count() for mask in range(16)], device=targets.device)
    counts = counts_table[targets]
    occupied = valid & (counts > 0)
    zero = output["raw_layout_logits"].sum() * 0
    styles = torch.where(plan.sum(dim=-1) > 0, plan.argmax(dim=-1) + 1, 0)
    onset_values = nn.functional.binary_cross_entropy_with_logits(output["onset_logits"], (counts > 0).float(), reduction="none")
    onset = onset_values[valid].mean() if valid.any() else zero
    if occupied.any():
        weights = output["count_class_weights"][styles[occupied], counts[occupied] - 1]
        values = nn.functional.cross_entropy(output["conditional_count_logits"][occupied], counts[occupied] - 1, reduction="none")
        quantity = (values * weights).sum() / weights.sum().clamp_min(1)
    else:
        quantity = zero
    conditional = output["conditional_count_logits"].softmax(dim=-1)
    quad_values = nn.functional.binary_cross_entropy(conditional[..., 3].clamp(1e-6, 1 - 1e-6), (counts == 4).float(), reduction="none")
    quad_weights = torch.where(counts == 4, 3.0, 1.0)
    quad = (quad_values * quad_weights)[occupied].mean() if occupied.any() else zero
    conditional_histogram = []
    for lower in (0, 96):
        selected = reliable[:, lower:lower + 96] & (counts[:, lower:lower + 96] > 0)
        denominator = selected.sum(dim=-1)
        predicted = (conditional[:, lower:lower + 96] * selected[..., None]).sum(dim=1) / denominator.clamp_min(1)[:, None]
        reference = (nn.functional.one_hot((counts[:, lower:lower + 96] - 1).clamp_min(0), 4).float() * selected[..., None]).sum(dim=1) / denominator.clamp_min(1)[:, None]
        values = ((reference - predicted).square() * torch.as_tensor([1.0, 1.0, 1.5, 3.0], device=targets.device)).sum(dim=-1)
        if (denominator > 0).any():
            conditional_histogram.append(values[denominator > 0].mean())
    histogram = torch.stack(conditional_histogram).mean() if conditional_histogram else zero
    fine = torch.as_tensor([tick % 12 != 0 for tick in SLOTS], device=targets.device).repeat(targets.shape[1] // len(SLOTS))
    fine_negative = valid & fine[None] & (targets == 0)
    fine_false_positive = nn.functional.softplus(output["onset_logits"])[fine_negative].mean() if fine_negative.any() else zero
    joint = joint_mask_probability(output)
    positions = torch.as_tensor([beat * 24 + offset for beat in range(8) for offset, tick in enumerate(SLOTS) if tick % 12 == 0], device=targets.device)
    probability = joint[:, positions]
    reference = targets[:, positions]
    masks = torch.arange(16, device=targets.device)
    overlap = counts_table[masks[:, None] & masks[None]]
    transition = (counts_table[:, None] > 2) & (counts_table[None] > 1) & (overlap >= 1)
    transition &= (counts_table[None] < counts_table[:, None]) | (overlap < counts_table[None])
    pair_positive = transition[reference[:, :-1], reference[:, 1:]]
    paired_style = styles[:, positions]
    pair_valid = reliable[:, positions[:-1]] & reliable[:, positions[1:]]
    pair_valid &= (paired_style[:, :-1] == STYLE_NAMES.index("Chordjacks")) & (paired_style[:, 1:] == STYLE_NAMES.index("Chordjacks"))
    matched = pair_valid & pair_positive
    transition_probability = ((probability[:, :-1] @ transition.to(joint.dtype)) * probability[:, 1:]).sum(dim=-1)
    chordjack = -transition_probability.clamp_min(1e-6).log()[matched].mean() if matched.any() else zero
    chordjack = (chordjack - 0.693147).clamp_min(0)
    losses["loss"] = losses["loss"] - losses["onset_loss"] - losses["count_loss"] + onset + quantity
    losses["loss"] = losses["loss"] + 0.20 * quad + 0.50 * histogram + 0.20 * fine_false_positive + 0.10 * chordjack
    losses.update(onset_loss=onset, count_loss=quantity, quad_calibration_loss=quad,
                  occupied_count_histogram_loss=histogram, fine_false_positive_loss=fine_false_positive, chordjack_transition_loss=chordjack)
    return losses
