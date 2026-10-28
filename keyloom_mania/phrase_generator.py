import torch
from torch import nn

from .rhythm_decode import RhythmDecodedGenerator
from .style_sequence import sequence_loss
from .style_rhythm import joint_mask_probability
from .tiny_model import TinyModelConfig


class PhraseGenerator(RhythmDecodedGenerator):
    def __init__(self, config=TinyModelConfig(), rhythm_priors=None):
        super().__init__(config, rhythm_priors)
        self.recent_layout_projection = nn.Linear(16, config.width, bias=False)
        self.variant_head = nn.Linear(config.width, 12)
        nn.init.xavier_uniform_(self.recent_layout_projection.weight)
        nn.init.xavier_uniform_(self.variant_head.weight)
        nn.init.zeros_(self.variant_head.bias)

    def decode_layout(self, memory, phase, condition, previous, plan, current_counts):
        length = previous.shape[1]
        history = nn.functional.one_hot(previous.clamp_max(15), 16).float()
        history[:, 0] = 0
        history[..., 0] = 0
        cumulative = history.cumsum(dim=1)
        delayed = torch.cat((torch.zeros_like(cumulative[:, :min(96, length)]), cumulative[:, :-96]), dim=1)
        recent = (cumulative - delayed) / 16
        style = self.style_encoder(plan[:, :length])
        hidden = self.note_embedding(previous) + self.spacing_projection(self.timing_state(previous))
        hidden = hidden + self.position_embedding(torch.arange(length, device=phase.device)) + self.recent_layout_projection(recent)
        hidden = hidden + self.phase_projection(phase[:, :length]) + self.condition_projection(condition)[:, None]
        hidden = hidden + self.plan_projection(plan[:, :length]) + style
        causal = torch.ones(length, length, device=phase.device, dtype=torch.bool).triu(1)
        raw = self.output(self.decoder(hidden, memory, tgt_mask=causal))
        legal = self.mask_counts[None, None] == current_counts[:, :, None]
        return raw.masked_fill(~legal, -torch.inf), raw

    def forward(self, acoustic, music, phase, condition, previous, current_counts, local=None, plan=None, count_plan_enabled=True, density=None):
        memory = self.condition_memory(self.encode_audio(acoustic, music, phase, local), self.density_condition(density, plan))
        quantity = self.decode_counts(memory, phase, condition, self.history_counts[previous], plan, count_plan_enabled)
        layout, raw = self.decode_layout(memory, phase, condition, previous, plan, current_counts)
        variants = self.variant_head(memory + self.style_encoder(plan))
        return {"onset_logits": self.generation_onset_logits(memory, condition, plan), "rhythm_logits": self.rhythm_logits(memory, condition, plan),
                "conditional_count_logits": quantity, "layout_logits": layout, "raw_layout_logits": raw,
                "variant_prediction": variants, "current_counts": current_counts}

    def configuration(self):
        return {**super().configuration(), "architecture": "phrase-generator-v4", "recent_layout_features": 16}


def phrase_generator_loss(output, targets, plan, valid, onset_weight, structural_valid, rhythm_labels):
    losses = sequence_loss(output, targets, plan, valid, onset_weight, structural_valid, rhythm_labels)
    probabilities = joint_mask_probability(output)
    columns = ((torch.arange(16, device=targets.device)[:, None] & (1 << torch.arange(4, device=targets.device))[None]) != 0).float()
    counts = columns.sum(dim=-1)
    target_columns = columns[targets]
    target_counts = counts[targets].long()
    labels = torch.cat((target_columns, nn.functional.one_hot(target_counts, 5)[..., 1:].float(),
                        torch.roll(target_columns, 6, dims=1)), dim=-1)
    labels[:, :6, -4:] = 0
    variant = nn.functional.binary_cross_entropy_with_logits(output["variant_prediction"][valid], labels[valid]) if valid.any() else output["variant_prediction"].sum() * 0
    predicted_rows = (1 - probabilities[..., 0]).reshape(len(targets), 8, 24).sum(dim=-1)
    predicted_notes = (probabilities @ counts).reshape(len(targets), 8, 24).sum(dim=-1)
    true_rows = (targets != 0).reshape(len(targets), 8, 24).sum(dim=-1).float()
    true_notes = target_counts.reshape(len(targets), 8, 24).sum(dim=-1).float()
    beat_valid = valid.reshape(len(targets), 8, 24).all(dim=-1)
    density_values = nn.functional.smooth_l1_loss(torch.log1p(torch.stack((predicted_notes, predicted_rows), dim=-1)),
                                               torch.log1p(torch.stack((true_notes, true_rows), dim=-1)), reduction="none").mean(dim=-1)
    weights = torch.where(true_rows <= 1, 2.0, 1.0)
    density = (density_values * weights)[beat_valid].mean() if beat_valid.any() else variant * 0
    moment_losses = []
    selected_styles = torch.where(plan.sum(dim=-1) > 0, plan.argmax(dim=-1) + 1, 0)
    for style in selected_styles.unique():
        chosen = structural_valid & valid & (selected_styles == style) & (targets > 0)
        if int(style) == 0 or int(chosen.sum()) < 16:
            continue
        empirical = nn.functional.one_hot(targets[chosen], 16).float().mean(dim=0)
        predicted = probabilities[chosen].mean(dim=0)
        moment_losses.append(nn.functional.smooth_l1_loss(predicted, empirical, beta=0.05))
    diversity = torch.stack(moment_losses).mean() if moment_losses else variant * 0
    motif = losses["sequence_motif_loss"]
    losses["loss"] = losses["loss"] - 0.15 * motif + 0.05 * (motif - 0.693147).clamp_min(0) + 0.15 * variant + 0.3 * density + 0.4 * diversity
    losses.update(variant_loss=variant, density_calibration_loss=density, style_layout_moment_loss=diversity)
    return losses
