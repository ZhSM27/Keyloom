from __future__ import annotations

import torch
from torch import nn

from .local_rhythm import LOCAL_WIDTH
from .tiny_model import TinyModelConfig, TinyRiceGenerator


class RhythmRiceGenerator(TinyRiceGenerator):
    def __init__(self, config: TinyModelConfig = TinyModelConfig()):
        super().__init__(config)
        self.local_projection = nn.Sequential(nn.Linear(LOCAL_WIDTH, config.width, bias=False), nn.GELU(), nn.Linear(config.width, config.width, bias=False))
        self.onset_head = nn.Linear(config.width, 1)
        self.count_head = nn.Linear(config.width, 5)
        for module in (self.local_projection, self.onset_head, self.count_head):
            for parameter in module.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)

    def encode_audio(self, acoustic, music, phase, local=None):
        length = acoustic.shape[1]
        if length > self.config.max_positions or music.shape[:2] != acoustic.shape[:2] or phase.shape[:2] != acoustic.shape[:2]:
            raise ValueError("Audio and grid positions must have matching supported sequence lengths.")
        positions = torch.arange(length, device=acoustic.device)
        hidden = self.acoustic_projection(acoustic) + self.music_projection(music) + self.phase_projection(phase) + self.position_embedding(positions)
        if local is not None:
            if local.shape != (*acoustic.shape[:2], LOCAL_WIDTH):
                raise ValueError("Local audio must align with every candidate grid position.")
            hidden = hidden + self.local_projection(local)
        return self.audio_encoder(hidden)

    def auxiliary(self, memory, condition):
        hidden = memory + self.condition_projection(condition).unsqueeze(1)
        return {"onset_logits": self.onset_head(hidden).squeeze(-1), "count_logits": self.count_head(hidden)}

    def forward(self, acoustic, music, phase, condition, previous, local=None):
        memory = self.encode_audio(acoustic, music, phase, local)
        return {"logits": self.decode(memory, phase, condition, previous), **self.auxiliary(memory, condition)}

    @torch.inference_mode()
    def generate(self, acoustic, music, phase, condition, local=None):
        memory = self.encode_audio(acoustic, music, phase, local)
        previous = torch.full((acoustic.shape[0], 1), 16, dtype=torch.long, device=acoustic.device)
        generated = []
        for index in range(acoustic.shape[1]):
            predicted = self.decode(memory, phase, condition, previous)[:, -1].argmax(dim=-1)
            generated.append(predicted)
            previous = torch.cat((previous, predicted.unsqueeze(1)), dim=1)
        return torch.stack(generated, dim=1)


def auxiliary_loss(output, targets, onset_positive_weight, count_weights):
    counts = torch.tensor([mask.bit_count() for mask in range(16)], device=targets.device)[targets]
    onset = nn.functional.binary_cross_entropy_with_logits(output["onset_logits"], (counts > 0).to(output["onset_logits"].dtype), pos_weight=onset_positive_weight)
    onset = onset / (2 * onset_positive_weight / (1 + onset_positive_weight))
    quantity = nn.functional.cross_entropy(output["count_logits"].reshape(-1, 5), counts.reshape(-1), weight=count_weights)
    main = nn.functional.cross_entropy(output["logits"].reshape(-1, 16), targets.reshape(-1))
    return {"loss": main + 0.25 * onset + 0.25 * quantity, "cross_entropy": main, "onset_loss": onset, "count_loss": quantity}
