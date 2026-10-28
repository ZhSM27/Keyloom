"""Studio decoding controls from play feedback; trained weights stay unchanged."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch

from .lane_control import LaneControlConfig, LaneControlState
from .phrase_control import PhraseControlState
from .segment_labels import LABELS


JACK_STYLES = ("Jacks", "Chordjacks", "Minijacks", "Gluts")
CUT_STYLES = ("Chordstream", "Jumpstream", "Handstream", "Quadstream", "Split Trill", "Jumptrill")
DOUBLE_MASKS = (3, 5, 9, 6, 10, 12)  # 12, 13, 14, 23, 24, 34
DECODER_VERSION = "playstyle-v3-jack-context"
# Natural double-row counts in deduplicated, admitted TRAIN bars. These are
# conditional reference distributions, not equal-frequency placement quotas.
# Calibrated on training-only style diagnostics.
DOUBLE_REFERENCE_COUNTS = {
    "Chordstream": (15648, 6655, 10645, 6529, 6639, 15823),
    "Jumpstream": (103713, 46215, 85238, 48484, 46915, 105461),
    "Handstream": (10706, 4660, 8608, 4946, 4909, 11098),
    "Quadstream": (2448, 1182, 1701, 1530, 1209, 2665),
    "Split Trill": (1177, 1775, 3547, 3007, 1799, 1235),
    "Jumptrill": (18076, 664, 1280, 1021, 701, 18052),
}


@dataclass(frozen=True)
class PlaystyleControlConfig:
    seed: int = 42
    minijack_triple_chance: float = 0.08
    jack_extended_chance: float = 0.08
    minijack_hand_limit: int = 5
    double_history_beats: float = 8.0
    double_minimum_history: int = 8
    double_pseudocount: float = 8.0
    double_bias_strength: float = 2.0
    double_maximum_bias: float = 3.0
    double_temperature: float = 0.9
    double_logit_window: float = 5.0
    jack_variety: bool = True
    jack_repeat_reward: float = 0.5
    jack_temperature: float = 0.8
    jack_logit_window: float = 4.0
    jack_period_strength: float = 2.5
    cut_phrase_context: bool = False
    cut_context_strength: float = .6

    def __post_init__(self):
        for chance in (self.minijack_triple_chance, self.jack_extended_chance):
            if not math.isfinite(chance) or not 0 <= chance <= 1:
                raise ValueError("Jack extension chances must be probabilities.")
        positive = (self.double_history_beats, self.double_pseudocount, self.double_temperature, self.double_logit_window,
                    self.jack_temperature,self.jack_logit_window)
        nonnegative = (self.double_bias_strength, self.double_maximum_bias,self.jack_repeat_reward,self.jack_period_strength,self.cut_context_strength)
        if any(not math.isfinite(value) or value <= 0 for value in positive) or any(not math.isfinite(value) or value < 0 for value in nonnegative):
            raise ValueError("Invalid double-layout control parameters.")
        if not isinstance(self.double_minimum_history, int) or self.double_minimum_history < 1:
            raise ValueError("Double-layout history must have a positive minimum size.")
        if not isinstance(self.minijack_hand_limit, int) or self.minijack_hand_limit < 1:
            raise ValueError("Minijack hand runs need a positive limit.")


def style_ids(plan, batch_size, device):
    if plan is None or plan.shape != (batch_size, len(LABELS)) or plan.device != device or not torch.isfinite(plan).all():
        raise ValueError("Playstyle controls need the actual local style plan.")
    return torch.where(plan.sum(dim=-1) > 0, plan.argmax(dim=-1), -1)


class PlaystyleLaneControlState(LaneControlState):
    def __init__(self, batch_size, device, config=LaneControlConfig(policy="style-aware"), playstyle=PlaystyleControlConfig()):
        if config.policy != "style-aware":
            raise ValueError("Playstyle run controls require style-aware lane control.")
        super().__init__(batch_size, device, config)
        self.playstyle = playstyle
        self.run_generator = torch.Generator(device=self.device).manual_seed(playstyle.seed)
        self.run_draw = torch.ones(batch_size, 4, device=self.device)
        self.sixteenth_runs = torch.zeros_like(self.column_runs)
        self.maximum_sixteenth_runs = torch.zeros_like(self.column_runs)
        self.mask_hands = torch.stack(((torch.arange(16, device=self.device) & 3) != 0,
                                       (torch.arange(16, device=self.device) & 12) != 0), dim=-1)
        self.hand_runs = torch.zeros(batch_size, 2, device=self.device, dtype=torch.long)
        self.maximum_hand_runs = torch.zeros_like(self.hand_runs)

    def constraints(self, beats, plan=None):
        legal, penalty = super().constraints(beats, plan)
        styles = style_ids(plan, self.batch_size, self.device)
        mini = styles == LABELS.index("Minijacks")
        ordinary = torch.zeros_like(mini)
        for label in JACK_STYLES:
            ordinary |= styles == LABELS.index(label)
        longjack = (styles == LABELS.index("Longjacks")) & self.config.allow_longjacks
        controlled = ordinary | longjack
        if not controlled.any():
            return legal, penalty

        continuous = beats - self.last_row <= self.config.continuity_gap_beats + 1e-7
        row_runs = torch.where(continuous[:, None], self.column_runs, 0)
        sixteenth_adjacent = ((beats[:, None] - self.last_hit) - 0.25).abs() <= 1e-6
        timed_runs = torch.where(sixteenth_adjacent, self.sixteenth_runs, 0)
        runs = torch.maximum(row_runs, timed_runs)
        mini_limit = torch.where(self.run_draw < self.playstyle.minijack_triple_chance, 3, 2)
        # 92% of new ordinary chains stop by four. The remaining ceilings are
        # 5/6/7/8 with conditional probabilities 1/2, 1/4, 1/8, 1/8.
        chance = self.playstyle.jack_extended_chance
        extra = (self.run_draw < chance).long() + (self.run_draw < chance / 2).long()
        extra += (self.run_draw < chance / 4).long() + (self.run_draw < chance / 8).long()
        limits = torch.where(mini[:, None], mini_limit, 4 + extra)
        cooling = beats[:, None] - self.last_hit < self.config.minimum_gap_beats - 1e-7
        blocked = cooling | ((runs >= limits) & ~longjack[:, None])
        jack_legal = ~(blocked[:, None] & self.mask_columns[None]).any(dim=-1)
        hand_runs = torch.where(continuous[:, None], self.hand_runs, 0)
        blocked_hands = (hand_runs >= self.playstyle.minijack_hand_limit) & mini[:, None]
        jack_legal &= ~(blocked_hands[:, None] & self.mask_hands[None]).any(dim=-1)
        if self.playstyle.jack_variety:
            # If both hands reach the ceiling together, the next occupied row
            # has no legal hand. Release one hand on this row instead of waiting
            # until the decoder must cancel the next row entirely.
            nearing=(hand_runs >= self.playstyle.minijack_hand_limit-1).all(dim=-1)&mini
            jack_legal &= ~(nearing[:,None]&self.mask_hands.all(dim=-1)[None])

        # Shared-column reward keeps normal jacks available, while progressive
        # penalties make third mini hits and fifth ordinary hits less attractive.
        soft = torch.where(mini[:, None], 2, 4)
        excess = (runs + 1 - soft).clamp_min(0).float()
        run_penalty = 6.0 * (excess.square() @ self.mask_columns.float().T)
        overlap = (self.mask_columns[self.last_mask][:, None] & self.mask_columns[None]).sum(dim=-1)
        reward=self.playstyle.jack_repeat_reward if self.playstyle.jack_variety else self.config.jack_overlap_reward
        run_penalty -= reward * continuous[:, None] * overlap
        run_penalty = torch.where(longjack[:, None], torch.zeros_like(run_penalty), run_penalty)
        return torch.where(controlled[:, None], jack_legal, legal), torch.where(controlled[:, None], run_penalty, penalty)

    def advance(self, masks, beats):
        columns = self.mask_columns[masks]
        adjacent = ((beats[:, None] - self.last_hit) - 0.25).abs() <= 1e-6
        continuous = beats - self.last_row <= self.config.continuity_gap_beats + 1e-7
        row_runs = torch.where(continuous[:, None], self.column_runs, 0)
        starting = columns & ~((row_runs > 0) | adjacent)
        if starting.any():
            self.run_draw[starting] = torch.rand(int(starting.sum()), device=self.device, generator=self.run_generator)
        self.sixteenth_runs = torch.where(columns, torch.where(adjacent, self.sixteenth_runs + 1, 1), self.sixteenth_runs)
        self.maximum_sixteenth_runs = torch.maximum(self.maximum_sixteenth_runs, self.sixteenth_runs)
        hands = self.mask_hands[masks]
        hand_runs = torch.where(continuous[:, None], self.hand_runs, 0)
        updated_hands = torch.where(hands, hand_runs + 1, 0)
        self.hand_runs = torch.where((masks != 0)[:, None], updated_hands, self.hand_runs)
        self.maximum_hand_runs = torch.maximum(self.maximum_hand_runs, self.hand_runs)
        super().advance(masks, beats)

    def summary(self):
        return {**super().summary(), "decoder_version": DECODER_VERSION, "playstyle": asdict(self.playstyle),
                "maximum_sixteenth_column_runs": self.maximum_sixteenth_runs.cpu().tolist(),
                "maximum_hand_runs": self.maximum_hand_runs.cpu().tolist(),
                "run_policy": "Minijacks cap 2, occasional 3, at most five consecutive occupied rows per hand (12 counts once); ordinary jacks cap 4, rare 5..8; explicitly planned Longjacks have no run/mask/hand cap; same-column speed limit always applies."}


class PlaystylePhraseControlState(PhraseControlState):
    def __init__(self, batch_size, device, playstyle=PlaystyleControlConfig(), phrase_reference=None):
        super().__init__(batch_size, device, mode="penalty", seed=playstyle.seed)
        self.playstyle = playstyle
        self.double_history = [[] for _ in range(batch_size)]
        self.double_masks = torch.tensor(DOUBLE_MASKS, device=device)
        self.double_choices = 0
        self.double_adjustments = 0
        self.jack_history=[[] for _ in range(batch_size)]
        self.jack_choices=0
        self.jack_cycle_adjustments=0
        self.event_history=[[] for _ in range(batch_size)]
        self.context_scored_rows=0
        self.phrase_reference={}
        self.phrase_reference_sha256=None
        if playstyle.cut_phrase_context:
            if phrase_reference is None:
                from .features import hash_file
                from .catalog import verify_asset
                path=verify_asset('generator/phrase-reference.json')
                source=json.loads(path.read_text(encoding='utf-8'))
                if source.get('test_used') is not False or source.get('split')!='train':raise ValueError('Phrase reference must be train-only.')
                self.phrase_reference_sha256=hash_file(path);phrase_reference=source['contexts']
            self.phrase_reference=phrase_reference

    def jack_scores(self,logits,beats,styles):
        scores=logits.clone()
        masks=torch.arange(16,device=logits.device)
        for sample,history in enumerate(self.jack_history):
            style=int(styles[sample])
            if style<0 or LABELS[style] not in JACK_STYLES or not history:continue
            if float(beats[sample])-history[-1][0]>.5+1e-6 or history[-1][1]!=style:continue
            recent=[entry[2] for entry in history if float(beats[sample])-entry[0]<=8 and entry[1]==style]
            penalty=torch.zeros(16,device=logits.device,dtype=logits.dtype)
            for period in (2,4,8):
                if len(recent)<period or len(set(recent[-period:]))<2:continue
                matched=0
                for offset in range(1,len(recent)-period+1):
                    if recent[-offset]!=recent[-offset-period]:break
                    matched+=1
                prospective=period+matched+1
                grace=max(4,period+period//2)
                if prospective>=grace:
                    value=min(8.,self.playstyle.jack_period_strength*(1+(prospective-grace)/period))
                    penalty=torch.maximum(penalty,(masks==recent[-period]).to(logits.dtype)*value)
            scores[sample]-=penalty
            self.jack_cycle_adjustments+=int(bool(((penalty>0)&torch.isfinite(logits[sample])).any()))
        return scores

    def choose(self, logits, beats, fixed_grid, plan=None):
        styles = style_ids(plan, len(self.history), logits.device)
        scores = self.period_scores(logits, beats, fixed_grid)
        longjack = styles == LABELS.index("Longjacks")
        scores = torch.where(longjack[:, None], logits, scores)
        predicted = scores.argmax(dim=-1)
        double_set = torch.zeros(16, dtype=torch.bool, device=logits.device)
        double_set[self.double_masks] = True
        for sample, history in enumerate(self.double_history):
            style = int(styles[sample])
            if style < 0 or LABELS[style] not in CUT_STYLES:
                continue
            finite = torch.isfinite(logits[sample])
            if bool((finite & ~double_set).any()) or int(finite.sum()) < 2:
                continue
            values = scores[sample, self.double_masks].clone()
            reference_counts = DOUBLE_REFERENCE_COUNTS[LABELS[style]]
            reference = torch.tensor(reference_counts, dtype=values.dtype, device=values.device)
            reference /= reference.sum()
            context_scale=1.
            if self.playstyle.cut_phrase_context and len(self.event_history[sample])>=2:
                a,b=self.event_history[sample][-2:]
                gap=float(beats[sample])-b[0]
                if a[1]==b[1]==style and gap<=.5+1e-6 and b[0]-a[0]<=.5+1e-6:
                    key=f'{LABELS[style]}:{a[2]}:{b[2]}:2:{int(gap>.25+1e-6)}'
                    entry=self.phrase_reference.get(key)
                    if entry is not None:
                        counts=torch.tensor(entry['counts'],device=values.device,dtype=values.dtype)[self.double_masks]
                        conditional=(counts+8*reference)/(counts.sum()+8)
                        bias=(conditional/reference).log().clamp(-1.,1.)
                        values+=self.playstyle.cut_context_strength*bias
                        context_scale=.35
                        self.context_scored_rows+=1
            recent = [mask for beat, label, mask in history if float(beats[sample]) - beat <= self.playstyle.double_history_beats and label == style]
            if len(recent) >= self.playstyle.double_minimum_history:
                counts = torch.tensor([recent.count(mask) for mask in DOUBLE_MASKS], device=values.device, dtype=values.dtype)
                observed = (counts + self.playstyle.double_pseudocount * reference) / (len(recent) + self.playstyle.double_pseudocount)
                bias = self.playstyle.double_bias_strength * (reference / observed).log()
                values += context_scale*bias.clamp(-self.playstyle.double_maximum_bias, self.playstyle.double_maximum_bias)
                self.double_adjustments += 1
            # All entries still carry the upstream legality/count mask. No note
            # count, timing decision, forbidden column, or -inf is changed.
            values = values.masked_fill(values < values.max() - self.playstyle.double_logit_window, -torch.inf)
            probabilities = (values / self.playstyle.double_temperature).softmax(dim=-1)
            choice = torch.multinomial(probabilities, 1, generator=self.generator)
            predicted[sample] = self.double_masks[choice].squeeze(0)
            self.double_choices += 1
        if self.playstyle.jack_variety:
            # Event history includes true beat gaps and crosses window seams.
            # Work only inside the already decided count and legal-column mask.
            jack_scores=self.jack_scores(logits,beats,styles)
            for sample,style in enumerate(styles.tolist()):
                if style<0 or LABELS[style] not in JACK_STYLES:continue
                finite=torch.isfinite(logits[sample])
                if finite[0] or int(finite.sum())<2:continue
                values=jack_scores[sample]
                values=values.masked_fill(values<values.max()-self.playstyle.jack_logit_window,-torch.inf)
                choice=torch.multinomial((values/self.playstyle.jack_temperature).softmax(-1),1,generator=self.generator)
                predicted[sample]=choice[0]
                self.jack_choices+=1
        self.changed += int((predicted != logits.argmax(dim=-1)).sum())
        return predicted

    def advance(self, masks, beats, fixed_grid, plan=None):
        super().advance(masks, beats, fixed_grid, plan=plan)
        styles = style_ids(plan, len(self.history), masks.device)
        for sample, (mask, style) in enumerate(zip(masks.tolist(), styles.tolist())):
            if mask:
                history=self.event_history[sample]
                current=float(beats[sample])
                if history and (current-history[-1][0]>.5+1e-6 or history[-1][1]!=style):history=[]
                history.append((current,style,mask));self.event_history[sample]=history[-8:]
                current=float(beats[sample]);history=self.jack_history[sample]
                if history and (current-history[-1][0]>.5+1e-6 or history[-1][1]!=style):history=[]
                if style>=0 and LABELS[style] in JACK_STYLES:
                    history.append((current,style,mask))
                    history=[entry for entry in history if current-entry[0]<=8][-128:]
                else:history=[]
                self.jack_history[sample]=history
            if style >= 0 and LABELS[style] in CUT_STYLES and mask in DOUBLE_MASKS:
                history = self.double_history[sample]
                current = float(beats[sample])
                history.append((current, style, mask))
                self.double_history[sample] = [entry for entry in history if current - entry[0] <= self.playstyle.double_history_beats][-128:]

    def summary(self):
        return {**super().summary(), "decoder_version": DECODER_VERSION, "playstyle": asdict(self.playstyle),
                "jack_sampled_rows":self.jack_choices,"jack_cycle_scored_rows":self.jack_cycle_adjustments,
                "jack_policy":"Moderate shared-column reward; seeded bounded legal-layout sampling; penalize extending actual occupied-row periods 2/4/8 before a forced run-cap switch. Preserve short repetitions and explicit Longjacks.",
                "double_sampled_rows": self.double_choices, "double_reweighted_rows": self.double_adjustments,
                "concrete_context_scored_rows":self.context_scored_rows,"phrase_reference_sha256":self.phrase_reference_sha256,
                "double_policy": "Cut-style doubles only; seeded legal-candidate sampling, bounded recent/reference correction; train-derived nonuniform references, no placement quotas; Longjacks bypass period penalties."}


def studio_playstyle_controls(device, selected_styles, style_run_limits=(), playstyle=PlaystyleControlConfig()):
    config = LaneControlConfig(policy="style-aware", minimum_gap_beats=0.25, maximum_column_run=12,
                               maximum_mask_run=4, style_run_limits=style_run_limits,
                               allow_longjacks="Longjacks" in selected_styles)
    return PlaystyleLaneControlState(1, device, config, playstyle), PlaystylePhraseControlState(1, device, playstyle)
