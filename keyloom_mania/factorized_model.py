from __future__ import annotations

import torch
from torch import nn

from .segment_model import SegmentRiceGenerator
from .segment_supervision import PLAN_WIDTH
from .tiny_model import TinyModelConfig


class FactorizedRiceGenerator(SegmentRiceGenerator):
    def __init__(self, config: TinyModelConfig = TinyModelConfig()):
        super().__init__(config)
        del self.count_head
        self.count_history_embedding = nn.Embedding(6, config.width)
        self.current_count_embedding = nn.Embedding(5, config.width)
        self.count_plan_projection = nn.Linear(PLAN_WIDTH, config.width, bias=False)
        count_layer = nn.TransformerDecoderLayer(config.width, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True)
        self.count_decoder = nn.TransformerDecoder(count_layer, 1, norm=nn.LayerNorm(config.width))
        self.conditional_count_head = nn.Linear(config.width, 4)
        for module in (self.count_history_embedding, self.current_count_embedding, self.count_plan_projection, self.count_decoder, self.conditional_count_head):
            for parameter in module.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
        self.register_buffer("mask_counts", torch.tensor([mask.bit_count() for mask in range(16)]), persistent=False)
        self.register_buffer("history_counts", torch.tensor([mask.bit_count() for mask in range(16)] + [5]), persistent=False)
        self.register_buffer("overlap_counts", torch.tensor([[(previous & mask).bit_count() for mask in range(16)] for previous in range(17)]), persistent=False)

    def onset_logits(self, memory, condition):
        return self.onset_head(memory + self.condition_projection(condition).unsqueeze(1)).squeeze(-1)

    def generation_onset_logits(self, memory, condition, plan):
        return self.onset_logits(memory, condition)

    def condition_memory(self, memory, density=None):
        if density is not None:
            raise ValueError("This generator architecture does not accept neural density conditioning.")
        return memory

    def decode_counts(self, memory, phase, condition, previous_counts, plan, count_plan_enabled: bool):
        length = previous_counts.shape[1]
        positions = torch.arange(length, device=phase.device)
        selected_plan = plan[:, :length] if count_plan_enabled else torch.zeros_like(plan[:, :length])
        hidden = self.count_history_embedding(previous_counts) + self.position_embedding(positions) + self.phase_projection(phase[:, :length]) + self.condition_projection(condition).unsqueeze(1) + self.count_plan_projection(selected_plan)
        causal = torch.ones(length, length, device=phase.device, dtype=torch.bool).triu(1)
        return self.conditional_count_head(self.count_decoder(hidden, memory, tgt_mask=causal))

    def generation_count_logits(self, memory, phase, condition, previous_counts, previous, plan, count_plan_enabled):
        return self.decode_counts(memory, phase, condition, previous_counts, plan, count_plan_enabled)

    def generation_count_choice(self, counts, conditional, memory, phase, condition, previous, generated_counts,
                                plan, position, candidate_beats, lane_state, constrained_counts):
        return counts

    def generation_prefix_counts(self, prefix_masks):
        return self.mask_counts[prefix_masks]

    def generation_count_history(self, desired, placed):
        return placed

    def generation_layout_scores(self, logits, memory, phase, condition, previous_counts, desired,
                                 plan, position, candidate_beats, lane_state, onset_probability):
        return logits

    def decode_layout(self, memory, phase, condition, previous, plan, current_counts):
        length = previous.shape[1]
        if current_counts.shape != previous.shape or current_counts.dtype != torch.int64 or ((current_counts < 0) | (current_counts > 4)).any():
            raise ValueError("Every layout position requires an explicit integer count from zero through four.")
        positions = torch.arange(length, device=phase.device)
        hidden = self.note_embedding(previous) + self.current_count_embedding(current_counts) + self.position_embedding(positions) + self.phase_projection(phase[:, :length]) + self.condition_projection(condition).unsqueeze(1) + self.plan_projection(plan[:, :length])
        causal = torch.ones(length, length, device=phase.device, dtype=torch.bool).triu(1)
        raw = self.output(self.decoder(hidden, memory, tgt_mask=causal))
        legal = self.mask_counts[None, None, :] == current_counts[:, :, None]
        return raw.masked_fill(~legal, -torch.inf), raw

    def forward(self, acoustic, music, phase, condition, previous, current_counts, local=None, plan=None, count_plan_enabled=True):
        if plan is None or plan.shape != (*phase.shape[:2], PLAN_WIDTH) or previous.shape != phase.shape[:2]:
            raise ValueError("Factorized teacher forcing requires a complete grid and explicit local plan.")
        memory = self.encode_audio(acoustic, music, phase, local)
        quantity = self.decode_counts(memory, phase, condition, self.history_counts[previous], plan, count_plan_enabled)
        layout, raw = self.decode_layout(memory, phase, condition, previous, plan, current_counts)
        return {"onset_logits": self.onset_logits(memory, condition), "conditional_count_logits": quantity, "layout_logits": layout, "raw_layout_logits": raw, "current_counts": current_counts}

    @staticmethod
    def _beat_budgets(values, name, shape, candidates_per_beat, device):
        if values is None:
            return None
        if not torch.is_tensor(values):
            values = torch.as_tensor(values)
        if values.shape != shape or not values.is_floating_point() and values.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError(f"{name} must align with batch beats and use a numeric dtype.")
        values = values.to(device=device, dtype=torch.float32)
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} must be finite and nonnegative.")
        maximum = 4 if name == "chord_sizes" else candidates_per_beat * 4 if name == "note_budgets" else candidates_per_beat
        return values.clamp(0, maximum) if name in ("row_budgets", "note_budgets") else values.round().to(torch.long)

    @staticmethod
    def _discrete_budget_schedule(values):
        cumulative = torch.cumsum(values, dim=1)
        previous = torch.cat((torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), dim=1)
        return (torch.floor(cumulative + 0.5) - torch.floor(previous + 0.5)).to(torch.long).clamp_min(0)

    @staticmethod
    def _budget_selection(onset_probability, row_budgets, note_budgets, chord_sizes, candidates_per_beat):
        batch, length = onset_probability.shape
        selected = torch.zeros((batch, length), dtype=torch.bool, device=onset_probability.device)
        for sample in range(batch):
            for beat in range(row_budgets.shape[1]):
                start = beat * candidates_per_beat
                stop = min(length, start + candidates_per_beat)
                if stop <= start:
                    continue
                rows = int(row_budgets[sample, beat].item())
                notes = int(note_budgets[sample, beat].item())
                chord = 1 if chord_sizes is None else int(chord_sizes[sample, beat].item())
                if chord == 2:
                    maximum_rows = max(1, (2 * notes) // 3)
                elif chord == 3:
                    maximum_rows = max(1, (4 * notes) // 7)
                elif chord == 4:
                    maximum_rows = max(1, (4 * notes) // 7)
                else:
                    maximum_rows = max(1, notes)
                count = min(rows, maximum_rows, stop - start)
                if count <= 0:
                    continue
                positions = torch.topk(onset_probability[sample, start:stop], count).indices
                selected[sample, start + positions] = True
        return selected

    @staticmethod
    def _allocate_budget_counts(selected, onset_probability, row_budgets, note_budgets, chord_sizes, candidates_per_beat):
        counts = torch.zeros_like(selected, dtype=torch.long)
        if chord_sizes is None:
            return counts
        batch, length = selected.shape
        for sample in range(batch):
            phase = 0
            for beat in range(row_budgets.shape[1]):
                start = beat * candidates_per_beat
                stop = min(length, start + candidates_per_beat)
                positions = torch.nonzero(selected[sample, start:stop], as_tuple=False).flatten()
                if not len(positions):
                    continue
                budget = int(note_budgets[sample, beat].item())
                chord = int(chord_sizes[sample, beat].item())
                if chord == 2:
                    cycle = 2
                    values = torch.tensor([2 if (ordinal + phase) % cycle == 0 else 1 for ordinal in range(len(positions))], dtype=torch.long, device=selected.device)
                elif chord == 3:
                    cycle = 4
                    values = torch.tensor([(3, 1, 2, 1)[(ordinal + phase) % cycle] for ordinal in range(len(positions))], dtype=torch.long, device=selected.device)
                elif chord == 4:
                    cycle = 4
                    values = torch.tensor([(4, 1, 2, 1)[(ordinal + phase) % cycle] for ordinal in range(len(positions))], dtype=torch.long, device=selected.device)
                else:
                    cycle = 1
                    values = torch.full((len(positions),), chord, dtype=torch.long, device=selected.device)
                order = torch.argsort(onset_probability[sample, start:stop][positions], descending=True)
                values = torch.minimum(values, torch.full_like(values, 4)).clamp(1, 4)
                while int(values.sum()) > budget:
                    reduced = False
                    for ordinal in reversed(order.tolist()):
                        if values[ordinal] > 1:
                            values[ordinal] -= 1
                            reduced = True
                            if int(values.sum()) <= budget:
                                break
                    if not reduced:
                        break
                counts[sample, start + positions] = values
                phase = (phase + len(positions)) % cycle
        return counts

    @torch.inference_mode()
    def generate(self, acoustic, music, phase, condition, local=None, plan=None, count_plan_enabled=True, oracle_counts=None, row_budgets=None, note_budgets=None, chord_sizes=None, avoid_jacks=None, candidates_per_beat=24, prefix_masks=None, candidate_beats=None, lane_state=None, density=None, phrase_state=None, fast_empty=False):
        if self.training:
            raise ValueError("Generation requires evaluation mode.")
        shape = phase.shape[:2]
        if phrase_state is not None and lane_state is None:
            raise ValueError("Persistent phrase decoding requires real beat positions and lane state.")
        if plan is None or plan.shape != (*shape, PLAN_WIDTH):
            raise ValueError("Generation requires a complete explicit local plan.")
        if prefix_masks is None:
            prefix_masks = torch.empty((shape[0], 0), device=phase.device, dtype=torch.long)
        if prefix_masks.ndim != 2 or prefix_masks.shape[0] != shape[0] or prefix_masks.shape[1] >= shape[1] or prefix_masks.dtype != torch.long or prefix_masks.device != phase.device or ((prefix_masks < 0) | (prefix_masks > 15)).any():
            raise ValueError("Generation prefixes must be aligned legal masks and leave a target region.")
        prefix_slots = prefix_masks.shape[1]
        if lane_state is not None:
            lane_state.validate_grid(candidate_beats, shape)
            if oracle_counts is not None:
                raise ValueError("Oracle counts and lane constraints are separate diagnostic paths.")
            if prefix_slots:
                if torch.isneginf(lane_state.last_candidate).all():
                    for position in range(prefix_slots):
                        lane_state.advance(prefix_masks[:, position], candidate_beats[:, position])
                elif not torch.allclose(lane_state.last_candidate, candidate_beats[:, prefix_slots - 1], rtol=0, atol=1e-7):
                    raise ValueError("Continuation prefix and persisted lane history must end at the same candidate.")
        elif candidate_beats is not None:
            raise ValueError("Candidate beat positions require an explicit lane state.")
        if oracle_counts is not None and (oracle_counts.shape != shape or oracle_counts.dtype != torch.int64 or oracle_counts.device != phase.device or ((oracle_counts < 0) | (oracle_counts > 4)).any()):
            raise ValueError("Oracle counts must explicitly align with the complete candidate grid.")
        if row_budgets is None and note_budgets is None:
            beat_shape = None
        else:
            if candidates_per_beat < 1 or shape[1] % candidates_per_beat:
                raise ValueError("Candidate positions must divide into complete beats.")
            beat_shape = (shape[0], shape[1] // candidates_per_beat)
        row_budgets = self._beat_budgets(row_budgets, "row_budgets", beat_shape, candidates_per_beat, phase.device)
        note_budgets = self._beat_budgets(note_budgets, "note_budgets", beat_shape, candidates_per_beat, phase.device)
        if (row_budgets is None) != (note_budgets is None):
            raise ValueError("Row and note budgets must be supplied together.")
        if chord_sizes is not None:
            if row_budgets is None:
                raise ValueError("Chord sizes require row and note budgets.")
            chord_sizes = self._beat_budgets(chord_sizes, "chord_sizes", beat_shape, 1, phase.device).clamp(1, 4)
        if avoid_jacks is not None:
            if row_budgets is None:
                raise ValueError("Jack avoidance requires row and note budgets.")
            avoid_jacks = self._beat_budgets(avoid_jacks, "avoid_jacks", beat_shape, 1, phase.device).clamp(0, 1).to(phase.dtype)
        if oracle_counts is not None and row_budgets is not None:
            raise ValueError("Oracle counts and decoded budgets are separate diagnostic paths.")
        discrete_row_budgets = None if row_budgets is None else self._discrete_budget_schedule(row_budgets)
        discrete_note_budgets = None if note_budgets is None else self._discrete_budget_schedule(note_budgets)
        memory = self.condition_memory(self.encode_audio(acoustic, music, phase, local), density)
        onset_probability = self.generation_onset_logits(memory, condition, plan).sigmoid()
        density_alignment = None
        if getattr(self, 'density_coordination', False) and oracle_counts is None and row_budgets is None:
            from .density_coordination import coordinate_onsets
            onset_probability, density_alignment = coordinate_onsets(onset_probability,density,candidate_beats,prefix_masks,
                                                                     candidates_per_beat=candidates_per_beat)
            if hasattr(self, 'refresh_density_onsets'):
                self.refresh_density_onsets(onset_probability,candidate_beats)
        selected = None if discrete_row_budgets is None else self._budget_selection(onset_probability, discrete_row_budgets, discrete_note_budgets, chord_sizes, candidates_per_beat)
        planned_counts = None if selected is None or chord_sizes is None else self._allocate_budget_counts(selected, onset_probability, discrete_row_budgets, discrete_note_budgets, chord_sizes, candidates_per_beat)
        remaining_notes = None if discrete_note_budgets is None else discrete_note_budgets.clone()
        prefix_counts = self.mask_counts[prefix_masks]
        previous = torch.cat((torch.full((shape[0], 1), 16, device=phase.device, dtype=torch.long), prefix_masks), dim=1)
        music_prefix = self.generation_prefix_counts(prefix_masks)
        previous_counts = torch.cat((torch.full((shape[0], 1), 5, device=phase.device, dtype=torch.long), music_prefix), dim=1)
        musical_counts = list(music_prefix.unbind(dim=1))
        masks, generated_counts, probabilities = list(prefix_masks.unbind(dim=1)), list(prefix_counts.unbind(dim=1)), []
        if remaining_notes is not None and prefix_slots:
            for position in range(prefix_slots):
                beat = position // candidates_per_beat
                remaining_notes[:, beat] = (remaining_notes[:, beat] - prefix_counts[:, position]).clamp_min(0)
        for position in range(prefix_slots, shape[1]):
            if fast_empty and oracle_counts is None and selected is None and bool((onset_probability[:, position] < 0.5).all()):
                predicted = torch.zeros(shape[0], dtype=torch.long, device=phase.device)
                if phrase_state is not None:
                    from .tiny_data import SLOTS
                    fixed_grid = SLOTS[position % len(SLOTS)] % 12 == 0
                    empty_logits = torch.full((shape[0], 16), -torch.inf, device=phase.device)
                    empty_logits[:, 0] = 0
                    predicted = phrase_state.choose(empty_logits, candidate_beats[:, position], fixed_grid, plan=plan[:, position])
                    phrase_state.advance(predicted, candidate_beats[:, position], fixed_grid, plan=plan[:, position])
                if lane_state is not None:
                    lane_state.advance(predicted, candidate_beats[:, position])
                masks.append(predicted)
                generated_counts.append(predicted)
                musical_counts.append(predicted)
                probabilities.append(torch.full((shape[0], 4), 0.25, device=phase.device))
                previous = torch.cat((previous, predicted[:, None]), dim=1)
                previous_counts = torch.cat((previous_counts, predicted[:, None]), dim=1)
                continue
            conditional = self.generation_count_logits(memory, phase, condition, previous_counts, previous, plan, count_plan_enabled)[:, -1].softmax(-1)
            if oracle_counts is not None:
                counts = oracle_counts[:, position]
            elif planned_counts is not None and chord_sizes is not None:
                counts = planned_counts[:, position]
            elif selected is not None:
                beat = position // candidates_per_beat
                beat_stop = min(shape[1], (beat + 1) * candidates_per_beat)
                later = selected[:, position + 1 : beat_stop].sum(dim=1)
                cap = torch.minimum(torch.full_like(remaining_notes[:, beat], 4), remaining_notes[:, beat] - later).clamp_min(0)
                desired = (conditional.argmax(-1) + 1).clamp(1, 4)
                counts = torch.minimum(desired, cap)
                counts = counts * selected[:, position].to(torch.long)
            else:
                counts = (conditional.argmax(-1) + 1) * (onset_probability[:, position] >= 0.5)
            counts = self.generation_count_choice(counts, conditional, memory, phase, condition, previous, generated_counts,
                                                  plan, position, candidate_beats, lane_state,
                                                  oracle_counts is not None or selected is not None)
            desired = counts.clone()
            musical_counts.append(desired)
            if lane_state is not None:
                counts = lane_state.prepare_counts(counts, candidate_beats[:, position], plan[:, position])
            if remaining_notes is not None:
                beat = position // candidates_per_beat
                remaining_notes[:, beat] = (remaining_notes[:, beat] - counts).clamp_min(0)
            generated_counts.append(counts)
            count_prefix = torch.stack(generated_counts, dim=1)
            layout, _ = self.decode_layout(memory, phase, condition, previous, plan, count_prefix)
            if avoid_jacks is not None:
                beat = position // candidates_per_beat
                overlap = self.overlap_counts[previous[:, -1]].to(layout.dtype)
                layout[:, -1] = layout[:, -1] - 100.0 * avoid_jacks[:, beat, None] * overlap
            if lane_state is not None:
                layout[:, -1] = lane_state.apply_logits(layout[:, -1], candidate_beats[:, position], plan[:, position])
            layout[:, -1] = self.generation_layout_scores(layout[:, -1], memory, phase, condition, previous_counts,
                                                          desired, plan, position, candidate_beats, lane_state, onset_probability)
            if phrase_state is None:
                predicted = layout[:, -1].argmax(-1)
            else:
                from .tiny_data import SLOTS
                fixed_grid = SLOTS[position % len(SLOTS)] % 12 == 0
                predicted = phrase_state.choose(layout[:, -1], candidate_beats[:, position], fixed_grid, plan=plan[:, position])
                phrase_state.advance(predicted, candidate_beats[:, position], fixed_grid, plan=plan[:, position])
            if lane_state is not None:
                lane_state.advance(predicted, candidate_beats[:, position])
            masks.append(predicted)
            probabilities.append(conditional)
            previous = torch.cat((previous, predicted[:, None]), dim=1)
            previous_counts = torch.cat((previous_counts, self.generation_count_history(desired,counts)[:, None]), dim=1)
        conditional = torch.stack(probabilities, dim=1)
        target_onset = onset_probability[:, prefix_slots:]
        count_probability = torch.cat(((1 - target_onset).unsqueeze(-1), target_onset.unsqueeze(-1) * conditional), dim=-1)
        result = {"masks": torch.stack(masks, dim=1), "counts": torch.stack(generated_counts, dim=1), "music_counts":torch.stack(musical_counts,dim=1), "onset_probability": onset_probability, "conditional_count_probability": conditional, "count_probability": count_probability}
        if density_alignment is not None:
            result['density_alignment'] = density_alignment
        if fast_empty:
            result["empty_candidate_conditional_probability"] = "uniform placeholder on skipped empty candidates; masks and occupied-candidate probabilities preserve decoder semantics"
        if prefix_slots:
            result.update(prefix_masks=prefix_masks, target_masks=result["masks"][:, prefix_slots:], target_counts=result["counts"][:, prefix_slots:])
        if selected is not None:
            result.update(selected_rows=selected, remaining_note_budget=remaining_notes)
            if planned_counts is not None:
                result["planned_counts"] = planned_counts
        return result


def factorized_loss(output, targets, onset_positive_weight):
    counts = torch.tensor([mask.bit_count() for mask in range(16)], device=targets.device)[targets]
    if not torch.equal(output["current_counts"], counts):
        raise ValueError("Layout targets must be compatible with their teacher-forced current counts.")
    occupied = counts > 0
    onset = nn.functional.binary_cross_entropy_with_logits(output["onset_logits"], occupied.to(output["onset_logits"].dtype), pos_weight=onset_positive_weight)
    onset = onset / (2 * onset_positive_weight / (1 + onset_positive_weight))
    quantity = nn.functional.cross_entropy(output["conditional_count_logits"][occupied], counts[occupied] - 1) if occupied.any() else output["conditional_count_logits"].sum() * 0
    layout = nn.functional.cross_entropy(output["layout_logits"][occupied], targets[occupied]) if occupied.any() else output["raw_layout_logits"].sum() * 0
    return {"loss": onset + quantity + layout, "onset_loss": onset, "conditional_count_loss": quantity, "legal_layout_loss": layout}


def teacher_forced_nll(output, targets):
    counts = torch.tensor([mask.bit_count() for mask in range(16)], device=targets.device)[targets]
    occupied = counts > 0
    onset = nn.functional.binary_cross_entropy_with_logits(output["onset_logits"], occupied.to(output["onset_logits"].dtype), reduction="sum")
    quantity = nn.functional.cross_entropy(output["conditional_count_logits"][occupied], counts[occupied] - 1, reduction="sum") if occupied.any() else onset * 0
    layout = nn.functional.cross_entropy(output["layout_logits"].reshape(-1, 16), targets.reshape(-1), reduction="sum") if occupied.any() else onset * 0
    return {"onset": onset.item(), "conditional_count": quantity.item(), "layout": layout.item()}
