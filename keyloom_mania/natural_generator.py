"""Reference-layout generator with causal hand state, preserving the old heads."""
import torch
from torch import nn

from .cadence_generator import CadenceGenerator
from .motion_state import MOTION_WIDTH, MotionHistory, candidate_grid, motion_features
from .tiny_model import TinyModelConfig


class NaturalLayoutGenerator(CadenceGenerator):
    def __init__(self, config=TinyModelConfig(), **kwargs):
        super().__init__(config, **kwargs)
        self.motion_projection = nn.Linear(MOTION_WIDTH, config.width, bias=False)
        nn.init.zeros_(self.motion_projection.weight)
        self._motion_seed = None
        self._motion_beats = None

    def decode_layout(self, memory, phase, condition, previous, plan, current_counts):
        length = previous.shape[1]
        history = nn.functional.one_hot(previous.clamp_max(15), 16).float()
        history[:, 0] = 0
        history[..., 0] = 0
        cumulative = history.cumsum(dim=1)
        delayed = torch.cat((torch.zeros_like(cumulative[:, :min(96, length)]), cumulative[:, :-96]), dim=1)
        recent = (cumulative - delayed) / 16
        beats = self._motion_beats[:, :length] if self._motion_beats is not None else candidate_grid(len(previous), length, previous.device)
        motion = motion_features(previous, beats, plan[:, :length], phase[:, :length, 4], self._motion_seed)
        style = self.style_encoder(plan[:, :length])
        hidden = self.note_embedding(previous) + self.spacing_projection(self.timing_state(previous))
        hidden = hidden + self.position_embedding(torch.arange(length, device=phase.device)) + self.recent_layout_projection(recent)
        hidden = hidden + self.phase_projection(phase[:, :length]) + self.condition_projection(condition)[:, None]
        hidden = hidden + self.plan_projection(plan[:, :length]) + style + self.motion_projection(motion)
        causal = torch.ones(length, length, device=phase.device, dtype=torch.bool).triu(1)
        raw = self.output(self.decoder(hidden, memory, tgt_mask=causal))
        legal = self.mask_counts[None, None] == current_counts[:, :, None]
        return raw.masked_fill(~legal, -torch.inf), raw

    @torch.inference_mode()
    def generate(self, *args, motion_state=None, **kwargs):
        phase = kwargs.get('phase', args[2] if len(args) > 2 else None)
        if phase is None:
            raise ValueError('Generation requires phase inputs.')
        beats = kwargs.get('candidate_beats')
        if beats is None:
            beats = candidate_grid(phase.shape[0], phase.shape[1], phase.device)
        state = motion_state if motion_state is not None else MotionHistory()
        seed = state.before(beats, kwargs.get('prefix_masks'))
        self._motion_seed, self._motion_beats = seed, beats
        try:
            result = super().generate(*args, **kwargs)
            state.update(result['masks'], beats, seed)
            return result
        finally:
            self._motion_seed, self._motion_beats = None, None

    def configuration(self):
        return {**super().configuration(), 'architecture': 'natural-layout-v2', 'motion_features': MOTION_WIDTH}
