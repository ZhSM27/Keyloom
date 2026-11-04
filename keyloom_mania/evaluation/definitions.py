"""Timing-aware 4K pattern rules written from community definitions.

Inside the evaluator these rules replace the YAVSRG motif detector. They read
row timing as well as row content, recognise single-column jacks, and decide
for every bar whether each of the fifteen planner labels holds.

Paraphrased sources: the osu! wiki stream, trill and chord pages and the
osu!mania ranking-criteria glossary (stream, jumpstream, handstream,
quadstream, chordstream, jack, minijack, roll, trill, jumptrill, split
jumptrill, anchor, grace); Etterna terminology for chordjacks, gluts and the
length of longjacks. "Minitrills" has no community definition; here it means
repeated short trill fragments inside a stream.

"Tech" (technical) also has no agreed definition. Following Etterna's
"awkward patterning that can stem from any category" with polys (16ths mixed
with 12ths/24ths), MinaCalc's chaos measure (non-integer ratios between
consecutive gaps) and sudden bursts, a dense bar is Tech when its rhythm
mixes binary and ternary divisions, changes interval awkwardly, or bursts.
Tech is an evaluator label only: the released planner and generators know the
fifteen ``LABELS``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..segment_labels import LABELS
from .charts import absolute_beats

DEFINITION_SCHEMA = "community-pattern-rules-v2-tech"
EVALUATION_LABELS = (*LABELS, "Tech")
LINK_BEATS = .5           # rows further apart do not form one stream, jack or trill
TOLERANCE_MS = 3.
MINIMUM_BAR_ROWS = 4
STREAM_SHARE = .6         # bar rows inside constant-interval runs lasting at least one beat
# Most specific first. Chords define chordjacks even when one column anchors through them;
# minijacks inside a stream outrank incidental trill fragments.
PRIORITY = ("Tech", "Jumptrill", "Split Trill", "Gluts", "Chordjacks", "Longjacks", "Quadstream", "Handstream",
            "Jumpstream", "Rolls", "Trills", "Jacks", "Minijacks", "Minitrills", "Chordstream", "Stream")
ONE_HAND_JUMPS = (0b0011, 0b1100)
SPLIT_JUMPS = (0b0101, 0b1010, 0b1001, 0b0110)
RULES = {
    "Stream": "constant-interval single notes for at least one beat, no minijacks",
    "Chordstream": "stream with two or more chords per bar, jacks only beside quads",
    "Jumpstream": "chordstream whose chords are jumps (two or more per bar)",
    "Handstream": "chordstream with two or more hands per bar, jumps allowed",
    "Quadstream": "stream with two or more quads per bar; quads may jack their neighbours",
    "Rolls": "stream where single-note sweeps across adjacent columns cover half the bar",
    "Trills": "single notes alternating between two columns cover half the bar (runs of four or more)",
    "Minitrills": "stream with three or more short trill fragments (three or four notes)",
    "Jumptrill": "jumps alternating between hands (12/34) cover half the bar",
    "Split Trill": "jumps that each need both hands (13/24, 14/23) alternate over half the bar",
    "Jacks": "half the rows repeat a column from the previous row and a run reaches three",
    "Minijacks": "two or more same-column runs of two or three notes, none longer",
    "Longjacks": "a same-column run of five or more notes covers a quarter of the bar",
    "Chordjacks": "mostly chords, half of the rows chord-jacking the previous chord",
    "Gluts": "mostly jumps whose minijacks shift columns, no run beyond three",
    "Tech": "dense bar (linked rows) mixing 1/4-family and 1/3-family positions, changing interval by awkward ratios, or bursting",
}
TECH_MINIMUM_ROWS = 6
TECH_LINKED_SHARE = .75
TECH_CHAOS_SHARE = .25


@dataclass
class RowContext:
    beats: np.ndarray
    masks: np.ndarray
    sizes: np.ndarray
    gaps: np.ndarray
    gap_ms: np.ndarray
    beat_ms: np.ndarray
    overlaps: np.ndarray
    stream: np.ndarray
    fraction: np.ndarray
    runs: list
    trills: list
    rolls: list


def _segments(flags):
    """Maximal runs of consecutive True positions as (first, last)."""
    result, first = [], None
    for index, flag in enumerate(flags):
        if flag and first is None:
            first = index
        elif not flag and first is not None:
            result.append((first, index - 1))
            first = None
    if first is not None:
        result.append((first, len(flags) - 1))
    return result


def row_context(beats, masks, beat_ms, fraction):
    beats = np.asarray(beats, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.int64)
    beat_ms = np.asarray(beat_ms, dtype=np.float64)
    count = len(beats)
    values = masks.tolist()
    sizes = np.asarray([mask.bit_count() for mask in values], dtype=np.int64)
    gaps = np.diff(beats, prepend=-np.inf)
    tolerance = TOLERANCE_MS / beat_ms
    linked = gaps <= LINK_BEATS + tolerance
    previous = [0, *values[:-1]]
    overlaps = np.where(linked, [(a & b).bit_count() for a, b in zip(values, previous)], 0)
    regular = np.zeros(count, dtype=bool)
    regular[2:] = linked[2:] & linked[1:-1] & (np.abs(gaps[2:] - gaps[1:-1]) <= tolerance[2:])
    # Stream rows: constant-interval runs lasting at least one beat.
    stream = np.zeros(count, dtype=bool)
    start = 0
    while start < count - 1:
        if not linked[start + 1]:
            start += 1
            continue
        end = start + 1
        while end + 1 < count and linked[end + 1] and abs(gaps[end + 1] - gaps[start + 1]) <= tolerance[end + 1]:
            end += 1
        if beats[end] - beats[start] >= 1 - tolerance[start]:
            stream[start:end + 1] = True
        start = end
    runs = []
    for column in range(4):
        hit = (masks >> column) & 1 > 0
        continued = np.zeros(count, dtype=bool)
        continued[1:] = hit[1:] & hit[:-1] & linked[1:]
        runs += [(column, first - 1, last) for first, last in _segments(continued)]
    alternating = [bool(regular[i]) and values[i] == values[i - 2] and values[i] != values[i - 1] and not values[i] & values[i - 1]
                   if i >= 2 else False for i in range(count)]
    trills = []
    for first, last in _segments(alternating):
        pair = (values[first - 2], values[first - 1])
        if sizes[first - 2] == sizes[first - 1] == 1:
            kind = "single"
        elif all(mask in ONE_HAND_JUMPS for mask in pair):
            kind = "jumptrill"
        elif all(mask in SPLIT_JUMPS for mask in pair):
            kind = "split"
        else:
            kind = "chord"
        trills.append((first - 2, last, kind))
    columns = np.asarray([mask.bit_length() - 1 for mask in values])
    sweeping = np.zeros(count, dtype=bool)
    sweeping[2:] = (regular[2:] & (sizes[2:] == 1) & (sizes[1:-1] == 1) & (sizes[:-2] == 1)
                    & (np.abs(np.diff(columns)[1:]) == 1) & (np.diff(columns)[1:] == np.diff(columns)[:-1]))
    rolls = [(first - 2, last) for first, last in _segments(sweeping) if last - first + 3 >= 4]
    return RowContext(beats, masks, sizes, gaps, gaps * beat_ms, beat_ms, overlaps, stream,
                      np.asarray(fraction, dtype=np.float64), runs, trills, rolls)


def chart_context(clock, times, masks):
    times = np.asarray(times, dtype=np.float64)
    offsets = np.asarray(clock.offsets, dtype=np.float64)
    lengths = np.asarray(clock.lengths, dtype=np.float64)
    section = np.maximum(np.searchsorted(offsets, times, side="right") - 1, 0)
    local = (times - offsets[section]) / lengths[section]
    return row_context(absolute_beats(clock, times), masks, lengths[section], local - np.floor(local))


def _covered(rows, spans, minimum):
    if not len(rows):
        return 0.
    mask = np.zeros(len(rows), dtype=bool)
    for first, last in spans:
        if last - first + 1 >= minimum:
            mask |= (rows >= first) & (rows <= last)
    return float(mask.mean())


def bar_matches(context, rows):
    """Whether each label's definition holds for the bar made of these row indices."""
    rows = np.asarray(rows, dtype=np.int64)
    result = dict.fromkeys(EVALUATION_LABELS, False)
    count = len(rows)
    if count < MINIMUM_BAR_ROWS:
        return result
    sizes = context.sizes[rows]
    previous_sizes = np.where(rows > 0, context.sizes[np.maximum(rows - 1, 0)], 0)
    singles, jumps, hands, quads = (int(np.sum(sizes == size)) for size in (1, 2, 3, 4))
    chords = count - singles
    overlaps = context.overlaps[rows]
    jack_rows = overlaps > 0
    loose_jacks = int(np.sum(jack_rows & (sizes != 4) & (previous_sizes != 4)))
    stream_base = float(context.stream[rows].mean()) >= STREAM_SHARE
    first, last = int(rows[0]), int(rows[-1])
    runs = [(start, end) for _, start, end in context.runs if start <= last and end >= first]
    lengths = [end - start + 1 for start, end in runs]
    longest = max(lengths, default=1)
    trills = [(start, end, kind) for start, end, kind in context.trills if start <= last and end >= first]
    fragments = [(start, end) for start, end, kind in trills if kind == "single" and end - start + 1 <= 4 and first <= start and end <= last]
    chordstream = stream_base and loose_jacks <= 1 and chords >= 2
    result.update({
        "Stream": stream_base and loose_jacks <= 1 and chords <= count // 10,
        "Chordstream": chordstream,
        "Jumpstream": chordstream and jumps >= 2 and hands == 0 and quads == 0,
        "Handstream": chordstream and hands >= 2 and quads == 0,
        "Quadstream": stream_base and quads >= 2 and loose_jacks <= 1,
        "Rolls": stream_base and _covered(rows, context.rolls, 4) >= .5,
        "Trills": _covered(rows, [(a, b) for a, b, kind in trills if kind == "single"], 4) >= .5,
        "Jumptrill": _covered(rows, [(a, b) for a, b, kind in trills if kind == "jumptrill"], 4) >= .5,
        "Split Trill": _covered(rows, [(a, b) for a, b, kind in trills if kind == "split"], 4) >= .5,
        "Jacks": float(jack_rows.mean()) >= .5 and longest >= 3,
        "Minijacks": sum(2 <= length <= 3 for length in lengths) >= 2 and longest <= 3,
        "Longjacks": longest >= 5 and _covered(rows, runs, 5) >= .25,
        "Chordjacks": chords >= .6 * count and float(np.mean(jack_rows & (sizes >= 2) & (previous_sizes >= 2))) >= .5,
        "Gluts": (chords >= .6 * count and jumps >= hands + quads and float(jack_rows.mean()) >= .5 and longest <= 3
                  and float(np.mean(overlaps[jack_rows] == 1)) >= .5),
    })
    result["Tech"] = _tech(context, rows)
    result["Minitrills"] = stream_base and not result["Trills"] and len(fragments) >= 3 and _covered(rows, fragments, 3) >= .4
    return result


def _tech(context, rows):
    """Polys, chaotic interval ratios or bursts in a dense bar."""
    if len(rows) < TECH_MINIMUM_ROWS:
        return False
    gaps = context.gaps[rows]
    linked = gaps <= LINK_BEATS + 1e-6
    if linked.mean() < TECH_LINKED_SHARE:
        return False
    tolerance = TOLERANCE_MS / context.beat_ms[rows]
    fraction = context.fraction[rows]

    def on(division):
        scaled = fraction * division
        return np.abs(scaled - np.rint(scaled)) <= tolerance * division

    # Polys: 16th positions (k/4, not k/2) together with 12th/24th positions (k/3 or k/6, not k/2).
    quarters = on(4) & ~on(2)
    ternary = (on(3) | on(6)) & ~on(2)
    if quarters.sum() >= 2 and ternary.sum() >= 2:
        return True
    pairs = linked[1:] & linked[:-1]
    if pairs.sum() >= 4:
        first, second = gaps[1:][pairs], gaps[:-1][pairs]
        ratio = np.maximum(first, second) / np.minimum(first, second)
        part = ratio - np.floor(ratio + 1e-6)
        chaotic = (ratio > 1 + 1e-3) & (part >= .2) & (part <= .8)
        if chaotic.mean() >= TECH_CHAOS_SHARE:
            return True
    # Bursts: three or more rows at half the bar's usual gap or faster.
    usual = np.median(gaps[linked])
    fast = linked & (gaps <= usual / 2 + 1e-6)
    return any(last - first + 1 >= 2 for first, last in _segments(fast.tolist()))


def dominant(matches):
    return next((label for label in PRIORITY if matches[label]), None)
