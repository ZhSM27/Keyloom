import collections
import math

import numpy as np
import torch

from .main_style import STYLE_NAMES
from .tiny_data import SLOTS, TICKS_PER_BEAT


RHYTHM_NAMES = ("sixteenth", "triplet", "thirty_second", "mixed")
RHYTHM_GRIDS = np.asarray([[tick % divisor == 0 for tick in SLOTS] for divisor in (12, 8, 6)] + [[True] * len(SLOTS)], dtype=bool)
REGULAR_STYLES = ("Jacks", "Chordjacks", "Minijacks", "Gluts", "Jumpstream", "Handstream")


def rhythm_targets(targets):
    occupied = np.asarray(targets).reshape(*targets.shape[:-1], -1, len(SLOTS)) != 0
    compatible = (~occupied[..., None, :] | RHYTHM_GRIDS).all(axis=-1)
    return compatible.argmax(axis=-1).astype(np.int64)


def joint_mask_probability(output):
    if output.get('raw_layout_conditioned_count',False):
        raise ValueError('Count-conditioned layout heads require a separate forward evaluation for each count before constructing a joint distribution.')
    raw = output["raw_layout_logits"]
    counts = torch.as_tensor([mask.bit_count() for mask in range(16)], device=raw.device)
    pieces = []
    quantity = output["conditional_count_logits"].softmax(dim=-1)
    onset = output["onset_logits"].sigmoid()
    for count in range(1, 5):
        selected = counts == count
        pieces.append(raw.masked_fill(~selected, -torch.inf).softmax(dim=-1) * quantity[..., count - 1, None])
    joint = torch.stack(pieces).sum(dim=0) * onset[..., None]
    joint[..., 0] = 1 - onset
    return joint


def rhythm_bar_metrics(clock, notes, starts, ends):
    rows = collections.defaultdict(int)
    for note in notes:
        if note.end_ms is None:
            rows[note.time_ms] |= 1 << note.column
    times = np.asarray(sorted(rows), dtype=np.float64)
    masks = np.asarray([rows[time] for time in times], dtype=np.int64)
    results = []
    for start, end in zip(starts, ends):
        selected = np.flatnonzero((times >= start) & (times < end))
        denominators, gaps = collections.Counter(), collections.Counter()
        shared, adjacent, sixteenth_shared = 0, 0, 0
        run, mask_run, maximum_run, maximum_mask = np.zeros(4, dtype=int), 0, 0, 0
        for offset, position in enumerate(selected):
            section = clock.sections[clock.section_at(float(times[position]))]
            beat = (times[position] - section.offset_ms) / section.beat_length_ms
            tolerance = 1.01 / section.beat_length_ms
            denominator = next((value for value in (1, 2, 3, 4, 6, 8, 12, 16, 24, 48) if abs(beat * value - round(beat * value)) / value <= tolerance), 0)
            denominators[denominator] += 1
            mask = int(masks[position])
            previous = int(masks[selected[offset - 1]]) if offset else 0
            gap = (times[position] - times[selected[offset - 1]]) / section.beat_length_ms if offset else math.inf
            continuous = gap <= 0.5 + tolerance
            columns = np.asarray([bool(mask & (1 << column)) for column in range(4)])
            run = np.where(columns, np.where(continuous, run + 1, 1), 0)
            mask_run = mask_run + 1 if continuous and previous == mask else 1
            maximum_run = max(maximum_run, int(run.max()))
            maximum_mask = max(maximum_mask, mask_run)
            if offset:
                adjacent += 1
                shared += bool(mask & previous)
                sixteenth_shared += bool(mask & previous) and abs(gap - 0.25) <= tolerance
                for name, distance in (("sixteenth", 0.25), ("twenty_fourth", 1 / 6), ("thirty_second", 0.125)):
                    gaps[name] += abs(gap - distance) <= tolerance
        off = sum(value for key, value in denominators.items() if key not in (1, 2, 4))
        results.append({"onset_rows": len(selected), "off_sixteenth_rows": off, "off_sixteenth_fraction": off / max(1, len(selected)),
                        "position_denominators": dict(denominators), "row_gaps": {name: int(value) for name, value in gaps.items()}, "adjacent_shared_pairs": int(shared),
                        "adjacent_row_pairs": adjacent, "sixteenth_shared_pairs": int(sixteenth_shared),
                        "maximum_column_run_in_bar": maximum_run, "maximum_mask_run_in_bar": maximum_mask})
    return results
