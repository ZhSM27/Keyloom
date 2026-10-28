"""Learn finite hand dynamics inside the layout decoder; protect musical heads."""
import copy
import torch
from torch import nn
from .expression_generator import ExpressionGenerator, event_context
from .motion_state import candidate_grid, motion_features
from .short_hand_dynamics import BINS, CHANNELS, SCALARS, SCHEMA, WINDOWS, ordered_hand_context
from .tiny_model import TinyModelConfig
from .segment_labels import LABELS


class HandDynamicsGenerator(ExpressionGenerator):
    def __init__(self, config=TinyModelConfig(), **kwargs):
        super().__init__(config, **kwargs)
        self.short_encoder = nn.Sequential(nn.Conv1d(CHANNELS, 24, 3, padding=1), nn.GELU(),
            nn.Conv1d(24, 24, 3, padding=1), nn.GELU())
        self.short_projection = nn.Linear(24 * BINS + SCALARS, config.width)
        nn.init.zeros_(self.short_projection.weight); nn.init.zeros_(self.short_projection.bias)
        self.hand_allocation_head = nn.Linear(config.width, 9)
        self.protected_layout_decoder = copy.deepcopy(self.decoder)
        for parameter in self.protected_layout_decoder.parameters():
            parameter.requires_grad_(False)
        self.hand_load_strength = 0.
        self._allocation_logits = None

    def decode_layout(self, memory, phase, condition, previous, plan, current_counts):
        if self._reference_pass:
            return super().decode_layout(memory, phase, condition, previous, plan, current_counts)
        length = previous.shape[1]
        jack = plan[:, :length, [LABELS.index(n) for n in ('Minijacks', 'Jacks', 'Chordjacks', 'Gluts', 'Longjacks')]].amax(-1) > .5
        protected = None
        if not bool(jack.all()):
            original = self.decoder
            try:
                self.decoder = self.protected_layout_decoder
                _, protected = super().decode_layout(memory, phase, condition, previous, plan, current_counts)
            finally:
                self.decoder = original
            if not bool(jack.any()):
                self._allocation_logits = self.hand_allocation_head(protected.new_zeros(*protected.shape[:2], self.config.width))
                return protected.masked_fill(self.mask_counts[None, None] != current_counts[..., None], -torch.inf), protected
        beats = self._motion_beats[:, :length] if self._motion_beats is not None else candidate_grid(len(previous), length, previous.device)
        bins, scalars = ordered_hand_context(previous, beats, phase[:, :length, 4], self._motion_seed)
        # These are tiny temporal convolutions. Native kernels avoid cuDNN's
        # large Windows workspace/search allocations for the flattened rows.
        with torch.backends.cudnn.flags(enabled=False):
            encoded = self.short_encoder(bins.reshape(-1, BINS, CHANNELS).transpose(1, 2)).flatten(1)
        short = self.short_projection(torch.cat((encoded, scalars.reshape(-1, SCALARS)), -1)).reshape(len(previous), length, -1)
        history = nn.functional.one_hot(previous.clamp_max(15), 16).float()
        history[:, 0] = 0; history[..., 0] = 0
        cumulative = history.cumsum(1)
        delayed = torch.cat((torch.zeros_like(cumulative[:, :min(96, length)]), cumulative[:, :-96]), 1)
        motion = motion_features(previous, beats, plan[:, :length], phase[:, :length, 4], self._motion_seed)
        hidden = self.note_embedding(previous) + self.spacing_projection(self.timing_state(previous))
        hidden = hidden + self.position_embedding(torch.arange(length, device=phase.device)) + self.recent_layout_projection((cumulative - delayed) / 16)
        hidden = hidden + self.phase_projection(phase[:, :length]) + self.condition_projection(condition)[:, None]
        hidden = hidden + self.plan_projection(plan[:, :length]) + self.style_encoder(plan[:, :length]) + self.motion_projection(motion) + short
        causal = torch.ones(length, length, device=phase.device, dtype=torch.bool).triu(1)
        decoded = self.decoder(hidden, memory, tgt_mask=causal)
        raw = self.output(decoded)
        context = torch.cat((memory[:, :length].detach(), motion, event_context(previous, self._motion_seed),
            nn.functional.one_hot(current_counts, 5).float()), -1)
        raw = raw + 1.5 * self.phrase_adapter(context).tanh()
        if protected is not None:
            raw = torch.where(jack[..., None], raw, protected)
        self._allocation_logits = self.hand_allocation_head(decoded)
        # The auxiliary head is training-only; generation always chooses complete
        # legal masks with the music-selected count, never an independent hand quota.
        return raw.masked_fill(self.mask_counts[None, None] != current_counts[..., None], -torch.inf), raw

    def configuration(self):
        return {**super().configuration(), 'architecture': 'hand-dynamics-v7', 'short_pressure_schema': SCHEMA,
            'pressure_windows_beats': list(WINDOWS), 'ordered_history_bin_beats': .125, 'ordered_history_bins': BINS,
            'pressure_reranking': False, 'naturalness_priority': True, 'hand_allocation_training_only': True,
            'protected_cut_layout_bank': True,
            'layout_adaptation': 'last decoder layer plus finite short-history projection; musical source bank frozen'}
