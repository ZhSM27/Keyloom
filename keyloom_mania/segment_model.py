from __future__ import annotations

import torch
from torch import nn

from .rhythm_model import RhythmRiceGenerator
from .segment_supervision import PLAN_WIDTH
from .tiny_model import TinyModelConfig


class SegmentRiceGenerator(RhythmRiceGenerator):
    def __init__(self, config: TinyModelConfig = TinyModelConfig()):
        super().__init__(config)
        self.plan_projection = nn.Linear(PLAN_WIDTH, config.width, bias=False)
        nn.init.xavier_uniform_(self.plan_projection.weight)

    def decode_plan(self, memory, phase, condition, previous, plan=None):
        if plan is None:
            return super().decode(memory, phase, condition, previous)
        if plan.shape != (*phase.shape[:2], PLAN_WIDTH) or previous.shape[1] > phase.shape[1]:
            raise ValueError("Local pattern plans must align with the complete candidate grid.")
        length = previous.shape[1]
        positions = torch.arange(length, device=previous.device)
        hidden = self.note_embedding(previous) + self.position_embedding(positions) + self.phase_projection(phase[:, :length]) + self.condition_projection(condition).unsqueeze(1) + self.plan_projection(plan[:, :length])
        causal = torch.ones(length, length, device=previous.device, dtype=torch.bool).triu(1)
        return self.output(self.decoder(hidden, memory, tgt_mask=causal))

    def forward(self, acoustic, music, phase, condition, previous, local=None, plan=None):
        memory = self.encode_audio(acoustic, music, phase, local)
        return {"logits": self.decode_plan(memory, phase, condition, previous, plan), **self.auxiliary(memory, condition)}

    @torch.inference_mode()
    def generate(self, acoustic, music, phase, condition, local=None, plan=None):
        memory = self.encode_audio(acoustic, music, phase, local)
        previous = torch.full((acoustic.shape[0], 1), 16, dtype=torch.long, device=acoustic.device)
        generated = []
        for index in range(acoustic.shape[1]):
            predicted = self.decode_plan(memory, phase, condition, previous, plan)[:, -1].argmax(dim=-1)
            generated.append(predicted)
            previous = torch.cat((previous, predicted.unsqueeze(1)), dim=1)
        return torch.stack(generated, dim=1)
