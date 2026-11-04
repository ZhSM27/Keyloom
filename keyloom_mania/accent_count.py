"""Accent-aware count residual and an audio-only support head over music-base.

Both modules start at zero output, so an untrained model decodes exactly like
its parent. The count residual sees the frozen audio memory, accent channels,
beat phase, plan, density and the parent's own count probabilities, for every
style. The support head sees audio memory before density conditioning plus
accents: it estimates whether a candidate deserves a row at all.

Support decoding (off unless thresholds are set) gives up planned rows the
recording does not support: a candidate the density-conditioned onset head
accepts is dropped when its audio support is below the plan's per-style
threshold, unless it continues a constant-interval stream. Chords on weakly
supported rows are demoted. Styles whose identity is a continuous chord
pattern (Chordjacks) keep every row instead; their chord sizes combine the
model's count distribution with the human distribution of chord size given the
beat position and audio support (a product of experts), which restores the human
hierarchy of large chords on supported beats and jumps between them. Every drop,
demotion and size change is counted so planned and exported rows stay comparable.

Motif reuse (off unless a strength is set) gives chord layouts a phrase memory.
Each bar looks back for an earlier bar of the same planned style with a matching
rhythm and similar audio, and layouts overlapping that bar's row at the same beat
position score higher. It only reorders layouts of the already decided count and
legal columns; timing, counts and lane limits stay as they are.
"""
import torch
from torch import nn

from .accent_features import ACCENT_LOCAL_WIDTH, ACCENT_SCHEMA
from .music_count_natural import MusicCountNaturalGenerator, count_events
from .segment_labels import LABELS
from .tiny_data import SLOTS, TICKS_PER_BEAT

SCHEMA = "accent-count-v1"
SUPPORT_SCHEMA = "accent-support-drop-v1"
MOTIF_SCHEMA = "phrase-motif-reuse-v1"
DECODING_SCHEMA = "accent-decoding-v1"
BAR_TICKS = 4 * TICKS_PER_BEAT
STREAM_GAP_TICKS = TICKS_PER_BEAT // 2   # linked rows: at most half a beat apart
STREAM_SPAN_TICKS = TICKS_PER_BEAT       # a stream lasts at least one beat
SUPPRESSED_LOGIT = -30.0
NO_SHAPING, DEMOTED, SIZE_SHAPED = range(3)
SIZE_POSITIONS = ("1/1", "1/2", "1/4", "other")


def position_classes(length, device=None):
    """Beat-position class of each candidate in a beat-aligned window: on the beat, half beat, quarter beats, other."""
    ticks = torch.tensor(SLOTS, device=device)[torch.arange(length, device=device) % len(SLOTS)]
    classes = torch.full((length,), 3, dtype=torch.long, device=device)
    classes[ticks == 0] = 0
    classes[ticks == TICKS_PER_BEAT // 2] = 1
    classes[(ticks == TICKS_PER_BEAT // 4) | (ticks == 3 * TICKS_PER_BEAT // 4)] = 2
    return classes


def candidate_ticks(length, device=None):
    index = torch.arange(length, device=device)
    return (index // len(SLOTS)) * TICKS_PER_BEAT + torch.tensor(SLOTS, device=device)[index % len(SLOTS)]


def stream_rows(decided):
    """[B, L] bool: decided candidates inside a constant-interval run that lasts at least one beat."""
    ticks = candidate_ticks(decided.shape[1]).tolist()
    result = torch.zeros_like(decided)
    for batch, row in enumerate(decided.detach().cpu().tolist()):
        positions = [index for index, value in enumerate(row) if value]
        start = 0
        while start < len(positions):
            stop = start + 1
            if stop < len(positions):
                gap = ticks[positions[stop]] - ticks[positions[start]]
                while (stop + 1 < len(positions) and gap <= STREAM_GAP_TICKS
                       and ticks[positions[stop + 1]] - ticks[positions[stop]] == gap):
                    stop += 1
                if gap <= STREAM_GAP_TICKS and ticks[positions[stop]] - ticks[positions[start]] >= STREAM_SPAN_TICKS:
                    result[batch, positions[start:stop + 1]] = True
            start = max(stop, start + 1)
    return result


class AccentCountGenerator(MusicCountNaturalGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.config.width
        self.accent_count_adapter = nn.Sequential(nn.Linear(width + ACCENT_LOCAL_WIDTH + 5 + len(LABELS) + 2 + 4 + 12, 128),
                                                  nn.GELU(), nn.Linear(128, 4))
        self.support_head = nn.Sequential(nn.Linear(width + ACCENT_LOCAL_WIDTH + 5 + len(LABELS), 128), nn.GELU(), nn.Linear(128, 1))
        for module in (self.accent_count_adapter, self.support_head):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
        self.learned_count_styles = LABELS
        self.accent_residual_scale = 1.0
        # Per-style support thresholds (len(LABELS)); None keeps support decoding off.
        self.support_drop_thresholds = None
        self.support_chord_thresholds = None
        # Styles defined by their chords (Chordjacks): a weakly supported row is dropped, stream or not, instead of demoted.
        self.support_chord_drop_styles = ()
        # Styles shaped by chord size, never by dropping or demoting rows. size_prior holds the human log P(size)
        # per beat-position class and support bin: {"schema", "support_edges": [bins - 1], "log_prior": [4, bins, 4]};
        # size_prior_weight scales it against the model's count logits.
        self.support_size_styles = ()
        self.size_prior = None
        self.size_prior_weight = 0.
        # The free-running count head rarely reaches four notes; this logit restores quads where the human prior allows them.
        self.size_quad_bonus = 0.
        # Layout distribution correction: plan style -> 16 additive logits, for two-note layouts humans use more or less.
        self.layout_mask_bias = {}
        # Layout memory: plan styles that prefer the layout of an earlier similar bar, and how strongly (logits).
        self.motif_styles = ()
        self.motif_strength = 0.
        self.motif_lookback_bars = 16
        self.motif_minimum_rhythm = .75
        self.jack_style_ceilings = ()   # handed to the lane controls by the pipeline
        self.support_trace = False   # research: keep per-candidate support and onset decisions in the result
        self._accent = None
        self._support = None
        self._count_shaping = None
        self._support_stats = None
        self._prefix_slots = 0
        self._layout_previous = None
        self._motif_owner = None
        self._motif_reset()

    def configure_decoding(self, policy):
        """Apply a checkpoint's accent_decoding policy; without one, support decoding and motif reuse stay off."""
        if policy is None:
            return
        if policy.get("schema") != DECODING_SCHEMA:
            raise ValueError("Unsupported accent decoding policy.")
        thresholds = lambda name: torch.tensor([float(policy[name][label]) for label in LABELS])
        self.accent_residual_scale = float(policy["residual_scale"])
        self.support_drop_thresholds = thresholds("drop_thresholds")
        self.support_chord_thresholds = thresholds("chord_thresholds")
        self.support_chord_drop_styles = tuple(policy.get("chord_drop_styles", ()))
        self.support_size_styles = tuple(policy["size_styles"])
        self.size_prior = policy["size_prior"]
        self.size_prior_weight = float(policy["size_prior_weight"])
        self.size_quad_bonus = float(policy["size_quad_bonus"])
        self.layout_mask_bias = dict(policy["layout_mask_bias"])
        self.motif_styles = tuple(policy["motif_styles"])
        self.motif_strength = float(policy["motif_strength"])
        self.jack_style_ceilings = tuple(tuple(item) for item in policy["jack_style_ceilings"])

    def _motif_reset(self):
        self._motif_rows = {}        # bar -> {tick within the bar: mask}, song history across windows
        self._motif_bars = {}        # bar -> (plan style, unit audio vector)
        self._motif_choice = {}      # bar -> lag in bars, or None
        self._motif_references = {}  # absolute tick -> reference mask used for this row
        self._motif_recorded = 0

    def accent_residual(self, logits, memory, phase, previous_counts, plan):
        length = previous_counts.shape[1]
        density = self._count_training_density if self._count_training_density is not None else self._music_density
        if density is None or self._accent is None or self.accent_residual_scale == 0:
            return torch.zeros_like(logits)
        features = torch.cat((memory[:, :length].detach(), self._accent[:, :length], phase[:, :length], plan[:, :length],
                              torch.log1p(density[:, :length]), logits.detach().softmax(-1), count_events(previous_counts)), -1)
        return self.accent_residual_scale * 1.5 * self.accent_count_adapter(features).tanh()

    def decode_counts(self, memory, phase, condition, previous_counts, plan, count_plan_enabled):
        base = super().decode_counts(memory, phase, condition, previous_counts, plan, count_plan_enabled)
        if self._reference_pass or not count_plan_enabled:
            return base
        return base + self.accent_residual(base, memory, phase, previous_counts, plan)

    def generation_count_logits(self, memory, phase, condition, previous_counts, previous, plan, count_plan_enabled):
        base = super().generation_count_logits(memory, phase, condition, previous_counts, previous, plan, count_plan_enabled)
        if self._reference_pass or self._reference_anchor is None or not count_plan_enabled:
            return base
        logits = base + self.accent_residual(base, memory, phase, previous_counts, plan)
        if self._count_shaping is None:
            return logits
        length = previous_counts.shape[1]
        additive, kind = self._count_shaping
        shaped = logits + additive[:, :length]
        before, after = logits[:, -1].argmax(-1), shaped[:, -1].argmax(-1)
        current = kind[:, length - 1]
        for name, changed in (("demoted", (current == DEMOTED) & (after != before)),
                              ("size_lowered", (current == SIZE_SHAPED) & (after < before)),
                              ("size_raised", (current == SIZE_SHAPED) & (after > before))):
            self._support_stats[f"{name}_rows"] += int(changed.sum())
            self._support_stats[f"{name}_notes"] += int((after - before)[changed].sum())
        return shaped

    def decode_layout(self, memory, phase, condition, previous, plan, current_counts):
        self._layout_previous = previous
        return super().decode_layout(memory, phase, condition, previous, plan, current_counts)

    def _motif_store(self, tick, mask):
        if mask:
            self._motif_rows.setdefault(tick // BAR_TICKS, {})[tick % BAR_TICKS] = mask

    def _motif_lag(self, bar, position, memory, plan, beats, onset_probability):
        """The earlier bar to echo: same style, rhythm overlap above the minimum, then closest rhythm plus audio."""
        if bar in self._motif_choice:
            return self._motif_choice[bar]
        ticks = (beats[0] * TICKS_PER_BEAT).round().long()
        inside = ticks // BAR_TICKS == bar
        vector = memory[0, inside].mean(0)
        vector = vector / vector.norm().clamp_min(1e-6)
        style = int(plan[0, position].argmax())
        self._motif_bars[bar] = (style, vector)
        planned = {int(tick) % BAR_TICKS for tick, slot in zip(ticks[inside], torch.nonzero(inside).flatten())
                   if int(slot) >= position and onset_probability[0, int(slot)] >= .5}
        planned |= set(self._motif_rows.get(bar, {}))
        covered = int(ticks[inside].max()) % BAR_TICKS if int(ticks[inside].max()) // BAR_TICKS == bar else BAR_TICKS - 1
        choice, best = None, -1.
        if len(planned) >= 4:
            for lag in range(1, self.motif_lookback_bars + 1):
                earlier = self._motif_bars.get(bar - lag)
                rows = self._motif_rows.get(bar - lag)
                if earlier is None or rows is None or earlier[0] != style:
                    continue
                other = {tick for tick in rows if tick <= covered}
                rhythm = len(planned & other) / max(1, len(planned | other))
                if rhythm < self.motif_minimum_rhythm:
                    continue
                score = rhythm + float(vector @ earlier[1])
                if score > best + 1e-6:
                    choice, best = lag, score
        self._motif_choice[bar] = choice
        return choice

    def generation_layout_scores(self, logits, memory, phase, condition, previous_counts, desired, plan, position, candidate_beats,
                                 lane_state, onset_probability):
        logits = super().generation_layout_scores(logits, memory, phase, condition, previous_counts, desired, plan, position,
                                                  candidate_beats, lane_state, onset_probability)
        if self._reference_pass or self._support_stats is None or len(logits) != 1 or not bool((desired > 0).all()):
            return logits
        if self.layout_mask_bias and float(plan[0, position].sum()) > 0:
            bias = self.layout_mask_bias.get(LABELS[int(plan[0, position].argmax())])
            if bias is not None:
                logits = logits + torch.as_tensor(bias, device=logits.device, dtype=logits.dtype)[None]
        if not self.motif_strength or not self.motif_styles or self._layout_previous is None:
            return logits
        ticks = (candidate_beats[0] * TICKS_PER_BEAT).round().long()
        previous = self._layout_previous[0]
        for slot in range(self._motif_recorded, min(position, previous.shape[0] - 1)):
            self._motif_store(int(ticks[slot]), int(previous[slot + 1]))
        self._motif_recorded = max(self._motif_recorded, position)
        if not bool(self._styles(plan[:, position:position + 1], self.motif_styles).all()):
            return logits
        tick = int(ticks[position])
        bar = tick // BAR_TICKS
        lag = self._motif_lag(bar, position, memory, plan, candidate_beats, onset_probability)
        if lag is None:
            return logits
        reference = self._motif_rows.get(bar - lag, {}).get(tick % BAR_TICKS, 0)
        if not reference:
            return logits
        shared = torch.tensor([(mask & reference).bit_count() for mask in range(16)], device=logits.device, dtype=logits.dtype)
        sizes = torch.tensor([max(mask.bit_count(), reference.bit_count()) for mask in range(16)], device=logits.device, dtype=logits.dtype)
        self._motif_references[tick] = reference
        self._support_stats["motif_scored_rows"] += 1
        return logits + self.motif_strength * (shared / sizes.clamp_min(1))[None]

    def support_logits(self, audio_memory, phase, plan):
        """Row support from audio alone (memory before density conditioning)."""
        features = torch.cat((audio_memory.detach(), self._accent, phase, plan), -1)
        return self.support_head(features).squeeze(-1)

    def _plan_threshold(self, thresholds, plan):
        weights = plan.clamp_min(0)
        total = weights.sum(-1)
        value = (weights * thresholds.to(plan.device, plan.dtype)).sum(-1) / total.clamp_min(1e-6)
        return torch.where(total > 0, value, torch.zeros_like(value))

    def generation_onset_logits(self, memory, condition, plan):
        logits = super().generation_onset_logits(memory, condition, plan)
        if self._support is None:
            return logits
        decided = logits.sigmoid() >= .5
        stream = stream_rows(decided)
        sized = self._styles(plan, self.support_size_styles)
        drop = decided & (self._support < self._plan_threshold(self.support_drop_thresholds, plan)) & ~stream & ~sized
        weak = None
        if self.support_chord_thresholds is not None:
            weak = decided & (self._support < self._plan_threshold(self.support_chord_thresholds, plan)) & ~sized
            drop |= weak & self._styles(plan, self.support_chord_drop_styles)
        drop[:, :self._prefix_slots] = False
        if not self._reference_pass:
            stats = self._support_stats
            stats["decided_rows"] += int(decided[:, self._prefix_slots:].sum())
            stats["stream_protected_rows"] += int((stream & decided)[:, self._prefix_slots:].sum())
            stats["dropped_slots"] = torch.nonzero(drop[0]).flatten().tolist()
            stats["dropped_rows"] += len(stats["dropped_slots"])
            if self.support_trace:
                stats["trace"] = {"support": self._support[0].tolist(), "decided": decided[0].tolist(),
                                  "stream": stream[0].tolist(), "dropped": drop[0].tolist()}
            self._count_shaping = self._shaping(decided & ~drop, weak, sized)
        return logits.masked_fill(drop, SUPPRESSED_LOGIT)

    @staticmethod
    def _styles(plan, styles):
        if not styles:
            return torch.zeros(plan.shape[:2], dtype=torch.bool, device=plan.device)
        return plan[..., [LABELS.index(style) for style in styles]].amax(-1) > .5

    def _shaping(self, kept, weak, sized):
        """Additive count logits [B, L, 4] and the kind of change at each candidate, or None when nothing changes."""
        additive = torch.zeros((*kept.shape, 4), device=kept.device)
        kind = torch.full(kept.shape, NO_SHAPING, dtype=torch.long, device=kept.device)
        if weak is not None:
            demote = kept & weak
            additive[..., 1:] += SUPPRESSED_LOGIT * demote[..., None]
            kind[demote] = DEMOTED
        if self.support_size_styles and self.size_prior is not None and self.size_prior_weight:
            edges = torch.as_tensor(self.size_prior["support_edges"], dtype=self._support.dtype, device=kept.device)
            log_prior = torch.as_tensor(self.size_prior["log_prior"], dtype=additive.dtype, device=kept.device)
            bins = torch.searchsorted(edges, self._support.contiguous())
            positions = position_classes(kept.shape[1], kept.device)[None].expand_as(bins)
            shaped = kept & sized
            additive += self.size_prior_weight * log_prior[positions, bins] * shaped[..., None]
            additive[..., 3] += self.size_quad_bonus * shaped
            kind[shaped] = SIZE_SHAPED
        if bool((kind == NO_SHAPING).all()):
            return None
        return additive, kind

    @torch.inference_mode()
    def generate(self, *args, accent=None, **kwargs):
        self._accent = accent
        enabled = self.support_drop_thresholds is not None and accent is not None
        if enabled:
            if args or len(accent) != 1:
                raise ValueError("Support decoding needs keyword inputs and one song per batch.")
            audio = self.encode_audio(kwargs["acoustic"], kwargs["music"], kwargs["phase"], kwargs.get("local"))
            self._support = self.support_logits(audio, kwargs["phase"], kwargs["plan"]).sigmoid()
            prefix = kwargs.get("prefix_masks")
            self._prefix_slots = 0 if prefix is None else prefix.shape[1]
            self._support_stats = {"decided_rows": 0, "stream_protected_rows": 0, "dropped_rows": 0, "dropped_slots": [],
                                   "motif_scored_rows": 0, "motif_matched_rows": 0,
                                   **{f"{name}_{unit}": 0 for name in ("demoted", "size_lowered", "size_raised")
                                      for unit in ("rows", "notes")}}
            # Song history lives as long as the song's lane state; a new lane state is a new song.
            if self._motif_owner is not kwargs.get("lane_state"):
                self._motif_owner = kwargs.get("lane_state")
                self._motif_reset()
            self._motif_recorded = 0
        try:
            result = super().generate(*args, **kwargs)
            if enabled:
                if self.motif_strength:
                    ticks = (kwargs["candidate_beats"][0] * TICKS_PER_BEAT).round().long().tolist()
                    for tick, mask in zip(ticks, result["masks"][0].tolist()):
                        self._motif_store(tick, mask)
                        if self._motif_references.pop(tick, None) == mask and mask:
                            self._support_stats["motif_matched_rows"] += 1
                stats = dict(self._support_stats)
                if "trace" in stats:
                    # Window-aligned count probabilities (zeros over the continuation prefix) and the musical counts.
                    for name, key in (("count_probability", "conditional_count_probability"),
                                      ("source_count_probability", "source_conditional_count_probability")):
                        if key in result:
                            values = result[key][0]
                            pad = values.new_zeros((self._prefix_slots, values.shape[-1]))
                            stats["trace"] = {**stats["trace"], name: torch.cat((pad, values)).tolist()}
                    stats["trace"]["music_counts"] = result["music_counts"][0].tolist()
                result["accent_support"] = stats
            return result
        finally:
            self._accent = None
            self._support = None
            self._count_shaping = None
            self._support_stats = None
            self._layout_previous = None

    def configuration(self):
        return {**super().configuration(), "architecture": SCHEMA, "accent_schema": ACCENT_SCHEMA,
                "accent_residual_bound": 1.5, "accent_residual_scale": self.accent_residual_scale,
                "learned_count_styles": list(LABELS),
                "support_decoding": None if self.support_drop_thresholds is None else {
                    "schema": SUPPORT_SCHEMA,
                    "drop_thresholds": dict(zip(LABELS, map(float, self.support_drop_thresholds))),
                    "chord_thresholds": None if self.support_chord_thresholds is None
                    else dict(zip(LABELS, map(float, self.support_chord_thresholds))),
                    "chord_drop_styles": list(self.support_chord_drop_styles),
                    "size_styles": list(self.support_size_styles),
                    "size_prior": None if not self.support_size_styles or self.size_prior is None else {
                        "schema": self.size_prior.get("schema"), "weight": self.size_prior_weight,
                        "positions": list(SIZE_POSITIONS), "support_edges": list(self.size_prior["support_edges"]),
                        "quad_bonus": self.size_quad_bonus},
                    "layout_mask_bias": self.layout_mask_bias or None,
                    "motif": None if not self.motif_strength else {
                        "schema": MOTIF_SCHEMA, "styles": list(self.motif_styles), "strength": self.motif_strength,
                        "lookback_bars": self.motif_lookback_bars, "minimum_rhythm": self.motif_minimum_rhythm},
                    "stream_exemption": "constant-interval runs, gap <= 1/2 beat, span >= 1 beat; not for chord-drop styles"}}
