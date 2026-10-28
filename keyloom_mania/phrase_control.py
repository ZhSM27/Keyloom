import math

import torch


class PhraseControlState:
    def __init__(self, batch_size, device, mode="penalty", seed=42, grace_rows=32, strength=3.0):
        if batch_size < 1 or mode not in ("penalty", "sample") or grace_rows < 16 or not math.isfinite(strength) or strength < 0:
            raise ValueError("Invalid phrase control policy.")
        self.mode, self.grace_rows, self.strength = mode, grace_rows, strength
        self.periods = (2, 4, 8, 16, 32)
        self.history = [[] for _ in range(batch_size)]
        self.runs = torch.zeros(batch_size, len(self.periods), device=device, dtype=torch.long)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.changed = 0
        self.last_beats = None

    def period_scores(self, logits, beats, fixed_grid):
        if len(logits) != len(self.history):
            raise ValueError("Phrase history must align with the decoder batch.")
        if not fixed_grid:
            return logits.clone()
        if self.last_beats is not None and (beats <= self.last_beats).any():
            raise ValueError("Phrase history cannot replay prefix positions.")
        scores = logits.clone()
        masks = torch.arange(16, device=logits.device)
        for sample, history in enumerate(self.history):
            if self.last_beats is not None and abs(float(beats[sample] - self.last_beats[sample]) - 0.25) > 1e-6:
                continue
            for ordinal, period in enumerate(self.periods):
                if len(history) < period or len(set(history[-period:])) < 2 or sum(mask != 0 for mask in history[-period:]) < max(2, period // 2):
                    continue
                prospective = int(self.runs[sample, ordinal]) + period + 1
                excess = max(0, prospective - self.grace_rows)
                if excess:
                    penalty = min(30.0, self.strength * (1 + excess / 4))
                    scores[sample] -= (masks == history[-period]).to(scores.dtype) * penalty
        return scores

    def choose(self, logits, beats, fixed_grid, plan=None):
        scores = self.period_scores(logits, beats, fixed_grid)
        if self.mode == "sample" and fixed_grid:
            values, indices = scores.topk(4, dim=-1)
            values = values.masked_fill(values < values[:, :1] - 3.0, -torch.inf)
            probabilities = (values / 0.7).softmax(dim=-1)
            selected = torch.multinomial(probabilities, 1, generator=self.generator)
            predicted = indices.gather(1, selected).squeeze(1)
        else:
            predicted = scores.argmax(dim=-1)
        self.changed += int((predicted != logits.argmax(dim=-1)).sum())
        return predicted

    def advance(self, masks, beats, fixed_grid, plan=None):
        if not fixed_grid:
            return
        for sample, mask in enumerate(masks.tolist()):
            history = self.history[sample]
            contiguous = self.last_beats is None or abs(float(beats[sample] - self.last_beats[sample]) - 0.25) <= 1e-6
            for ordinal, period in enumerate(self.periods):
                matched = contiguous and len(history) >= period and history[-period] == mask
                self.runs[sample, ordinal] = self.runs[sample, ordinal] + 1 if matched else 0
            history.append(mask)
            if len(history) > 128:
                del history[:-128]
        self.last_beats = beats.clone()

    def summary(self):
        return {"mode": self.mode, "grace_rows": self.grace_rows, "strength": self.strength, "changed_layout_rows": self.changed,
                "periods": list(self.periods), "policy": "Persistent fixed-sixteenth history; penalize extending multi-mask periods after two bars; preserve uniform jack runs; optional seeded sampling among four legal layouts."}
