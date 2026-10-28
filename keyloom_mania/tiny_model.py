from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TinyModelConfig:
    width: int = 192
    heads: int = 6
    audio_layers: int = 2
    decoder_layers: int = 3
    feedforward: int = 768
    dropout: float = 0.0
    max_positions: int = 256


class TinyRiceGenerator(nn.Module):
    def __init__(self, config: TinyModelConfig = TinyModelConfig()):
        super().__init__()
        self.config = config
        self.acoustic_projection = nn.Linear(128, config.width)
        self.music_projection = nn.Linear(1024, config.width)
        self.phase_projection = nn.Linear(5, config.width)
        self.condition_projection = nn.Linear(5, config.width)
        self.position_embedding = nn.Embedding(config.max_positions, config.width)
        self.note_embedding = nn.Embedding(17, config.width)
        audio_layer = nn.TransformerEncoderLayer(config.width, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True)
        self.audio_encoder = nn.TransformerEncoder(audio_layer, config.audio_layers, norm=nn.LayerNorm(config.width), enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(config.width, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers, norm=nn.LayerNorm(config.width))
        self.output = nn.Linear(config.width, 16)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def encode_audio(self, acoustic, music, phase):
        length = acoustic.shape[1]
        if length > self.config.max_positions or music.shape[:2] != acoustic.shape[:2] or phase.shape[:2] != acoustic.shape[:2]:
            raise ValueError("Audio and grid positions must have matching supported sequence lengths.")
        positions = torch.arange(length, device=acoustic.device)
        memory = self.acoustic_projection(acoustic) + self.music_projection(music) + self.phase_projection(phase) + self.position_embedding(positions)
        return self.audio_encoder(memory)

    def decode(self, memory, phase, condition, previous):
        length = previous.shape[1]
        positions = torch.arange(length, device=previous.device)
        hidden = self.note_embedding(previous) + self.position_embedding(positions) + self.phase_projection(phase[:, :length]) + self.condition_projection(condition).unsqueeze(1)
        causal = torch.ones(length, length, device=previous.device, dtype=torch.bool).triu(1)
        return self.output(self.decoder(hidden, memory, tgt_mask=causal))

    def forward(self, acoustic, music, phase, condition, previous):
        memory = self.encode_audio(acoustic, music, phase)
        return self.decode(memory, phase, condition, previous)

    @torch.inference_mode()
    def generate(self, acoustic, music, phase, condition):
        memory = self.encode_audio(acoustic, music, phase)
        previous = torch.full((acoustic.shape[0], 1), 16, dtype=torch.long, device=acoustic.device)
        generated = []
        for index in range(acoustic.shape[1]):
            predicted = self.decode(memory, phase, condition, previous)[:, -1].argmax(dim=-1)
            generated.append(predicted)
            previous = torch.cat((previous, predicted.unsqueeze(1)), dim=1)
        return torch.stack(generated, dim=1)

    def configuration(self):
        return asdict(self.config)
