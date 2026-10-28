from .style_sequence import StyleSequenceGenerator
from .style_rhythm import REGULAR_STYLES
from .main_style import STYLE_NAMES


class RhythmDecodedGenerator(StyleSequenceGenerator):
    rhythm_decoding = "soft"

    def generation_onset_logits(self, memory, condition, plan):
        logits = super().generation_onset_logits(memory, condition, plan)
        if self.rhythm_decoding == "soft":
            return logits
        if self.rhythm_decoding != "learned-grid" or self.training:
            raise ValueError("Discrete learned rhythm decoding is an inference-only policy.")
        family = self.rhythm_logits(memory, condition, plan).argmax(dim=-1)
        allowed = self.rhythm_grids[family].bool().reshape(logits.shape)
        regular = plan[..., [STYLE_NAMES.index(label) - 1 for label in REGULAR_STYLES]].max(dim=-1).values > 0.5
        allowed = allowed | ~regular
        return logits.masked_fill(~allowed, -float("inf"))


def configure_rhythm_decoding(model, policy):
    if policy not in ("soft", "learned-grid"):
        raise ValueError("Unknown rhythm decoding policy.")
    if policy != "soft" and not isinstance(model, RhythmDecodedGenerator):
        raise ValueError("Learned-grid decoding requires a trained rhythm-family head.")
    if isinstance(model, RhythmDecodedGenerator):
        model.rhythm_decoding = policy
