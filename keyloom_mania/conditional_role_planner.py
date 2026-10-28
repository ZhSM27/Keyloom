"""Audio-conditioned owner intent and lower-style evidence, trained separately."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from .segment_labels import LABELS
from .style_hierarchy import compatible_styles


ROLE_MODEL_SCHEMA = "conditional-role-planner-v3"


@dataclass(frozen=True)
class ConditionalRoleConfig:
    width: int = 192
    layers: int = 3
    heads: int = 6
    feedforward: int = 384
    max_bars: int = 1024
    dropout: float = .1


def pool_sections(values, segment_ids, padding_mask):
    batch, length, width = values.shape
    groups = max(1, int(segment_ids.max().item()) + 1)
    index = segment_ids.clamp_min(0) + torch.arange(batch, device=values.device)[:, None] * groups
    valid = (segment_ids >= 0) & ~padding_mask
    sums = values.new_zeros(batch * groups, width)
    sums.index_add_(0, index.flatten(), (values * valid[..., None]).reshape(-1, width))
    counts = values.new_zeros(batch * groups)
    counts.index_add_(0, index.flatten(), valid.flatten().to(values.dtype))
    means = (sums / counts.clamp_min(1)[:, None])[index]
    return torch.where(valid[..., None], means, values)


class ConditionalRolePlanner(nn.Module):
    def __init__(self, config=ConditionalRoleConfig()):
        super().__init__()
        self.config = config
        width = config.width
        self.music = nn.Linear(1024, width)
        self.acoustic = nn.Linear(128, width)
        self.rhythm = nn.Linear(40, width)
        self.phase = nn.Linear(7, width)
        self.star = nn.Linear(1, width)
        self.primary_embedding = nn.Embedding(len(LABELS), width)
        self.secondary_embedding = nn.Embedding(len(LABELS), width)
        self.secondary_count = nn.Embedding(4, width)
        self.norm = nn.LayerNorm(width)
        layer = nn.TransformerEncoderLayer(width, config.heads, config.feedforward, config.dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.section_projection = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.owner_head = nn.Linear(width, len(LABELS))
        self.style_head = nn.Linear(width, len(LABELS))
        relation = torch.zeros(len(LABELS), len(LABELS), dtype=torch.bool)
        for i, label in enumerate(LABELS):
            relation[i, [LABELS.index(name) for name in compatible_styles(label)]] = True
        self.register_buffer("lower_relation", relation, persistent=False)
        positions = torch.arange(config.max_bars, dtype=torch.float32)[:, None]
        divisor = torch.exp(torch.arange(0, width, 2) * (-math.log(10000) / width))
        encoding = torch.zeros(config.max_bars, width)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(positions * divisor), torch.cos(positions * divisor)
        self.register_buffer("positions", encoding, persistent=False)

    def forward(self, music, acoustic, rhythm, phase, star, primary, secondary, segment_ids, padding_mask):
        if music.shape[1] > self.config.max_bars or secondary.shape != (len(music), len(LABELS)):
            raise ValueError("Role Planner requires aligned bars and a fifteen-label secondary set.")
        if ((secondary != 0) & (secondary != 1)).any() or (secondary.sum(-1) > 3).any() or secondary.gather(1, primary[:, None]).any():
            raise ValueError("Secondary set must contain up to three distinct non-primary types.")
        count = secondary.sum(dim=-1).long()
        condition = (self.primary_embedding(primary) + secondary @ self.secondary_embedding.weight / count.clamp_min(1).sqrt()[:, None]
                     + self.secondary_count(count) + self.star(star[:, None]))
        hidden = self.music(music) + self.acoustic(acoustic) + self.rhythm(rhythm) + self.phase(phase)
        hidden = self.norm(hidden + condition[:, None] + self.positions[:music.shape[1]][None])
        memory = self.encoder(hidden, src_key_padding_mask=padding_mask)
        section = pool_sections(memory, segment_ids, padding_mask)
        hidden = memory + self.section_projection(section)
        requested = secondary.bool().clone()
        requested.scatter_(1, primary[:, None], True)
        allowed = (requested[..., None] & self.lower_relation[None]).any(dim=1)
        owner = self.owner_head(hidden).masked_fill(~requested[:, None], -1e4)
        style = self.style_head(hidden).masked_fill(~allowed[:, None], -1e4)
        return {"owner_logits": owner, "style_logits": style}

    def configuration(self):
        return {"schema_version": ROLE_MODEL_SCHEMA, **asdict(self.config)}


def conditional_role_loss(output, batch, class_weights=None, balanced=False):
    known = (batch["owner_target"] >= 0) & ~batch["padding_mask"]
    zero = output["owner_logits"].sum() * 0
    if not known.any():
        return {"loss": zero, "owner_loss": zero, "style_loss": zero, "segment_loss": zero}
    def owner_loss(logits):
        main = known & (batch["owner_target"] == batch["primary"][:, None])
        secondary = known & ~main
        if balanced and main.any() and secondary.any():
            return .5 * (nn.functional.cross_entropy(logits[main], batch["owner_target"][main], weight=class_weights)
                         + nn.functional.cross_entropy(logits[secondary], batch["owner_target"][secondary], weight=class_weights))
        return nn.functional.cross_entropy(logits[known], batch["owner_target"][known], weight=class_weights)
    owner = owner_loss(output["owner_logits"])
    style = nn.functional.cross_entropy(output["style_logits"][known], batch["style_target"][known], weight=class_weights)
    # Keep unknown positions outside both the segment target and pooled scores.
    section_logits = pool_sections(output["owner_logits"], batch["segment_ids"], ~known)
    segment = owner_loss(section_logits)
    return {"loss": owner + .5 * style + .4 * segment, "owner_loss": owner, "style_loss": style, "segment_loss": segment}
