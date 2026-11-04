"""Block-additive chart observations for human-referenced naturalness evaluation.

Hand-pressure observations look back over the recent 2 and 4 beats only; no
state survives four beats. Nothing here prefers lower pressure or balanced
hands: every histogram is later compared with human charts of similar
difficulty, tempo and style mix. Observations are summed per 16-bar block so a
short excerpt is calibrated against equally long human segments. Bar styles
come from the community-definition rules in ``definitions``; selected
histograms are also kept per bar style so a Jumpstream bar can be compared
with human Jumpstream bars.
"""
from __future__ import annotations

import math

import numpy as np

from ..phrase_quality import MIRROR, canonical_motif
from .definitions import DEFINITION_SCHEMA, EVALUATION_LABELS as LABELS, bar_matches, chart_context, dominant

FEATURE_SCHEMA = "chart-naturalness-features-v3-2-4-beat-" + DEFINITION_SCHEMA
PRESSURE_WINDOWS_BEATS = (2.0, 4.0)
BAR_BEATS = 4
BLOCK_BARS = 16
TOLERANCE_MS = 2.0
HANDS = ((0, 1), (2, 3))
GAP_VALUES = (1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 3 / 4, 1., 2., 4.)
GAP_LABELS = ("1/16", "1/12", "1/8", "1/6", "1/4", "1/3", "1/2", "3/4", "1", "2", "4", ">4")
SNAP_DENOMINATORS = (1, 2, 4, 3, 8, 6, 16, 12, 24)
SNAP_LABELS = ("1/1", "1/2", "1/4", "1/3", "1/8", "1/6", "1/16", "1/12", "1/24", "other")
NPS_EDGES = (1., 2., 3., 4., 5., 6., 8., 10., 12., 15., 20.)
NPS_CENTERS = (.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7., 9., 11., 13.5, 17.5, 22.)
FAMILIES = {
    "stream": ("Stream", "Chordstream", "Jumpstream", "Handstream", "Quadstream"),
    "jack": ("Jacks", "Longjacks", "Chordjacks", "Minijacks", "Gluts"),
    "trill": ("Rolls", "Trills", "Minitrills", "Split Trill", "Jumptrill"),
    "tech": ("Tech",),
}


def _hands(mask):
    return tuple(hand for hand, columns in enumerate(HANDS) if any(mask >> column & 1 for column in columns))


def _transition_vocabulary():
    tokens = set()
    for before in range(1, 16):
        for after in range(1, 16):
            first, second = _hands(before), _hands(after)
            relation = ("same" if first == second else "alternate") if len(first) == len(second) == 1 else "mixed"
            tokens.add((before.bit_count(), after.bit_count(), (before & after).bit_count(), relation))
    return tuple(sorted(tokens))


TRANSITIONS = _transition_vocabulary()
TRANSITION_INDEX = {token: index for index, token in enumerate(TRANSITIONS)}
TRANSITION_TABLE = np.zeros((16, 16), dtype=np.int64)
for _before in range(1, 16):
    for _after in range(1, 16):
        _first, _second = _hands(_before), _hands(_after)
        _relation = ("same" if _first == _second else "alternate") if len(_first) == len(_second) == 1 else "mixed"
        TRANSITION_TABLE[_before, _after] = TRANSITION_INDEX[(_before.bit_count(), _after.bit_count(), (_before & _after).bit_count(), _relation)]

# Bin counts. Count observations use their value as the bin and clip the tail.
HISTOGRAMS = {
    "hand_rows_2b": 17, "hand_rows_4b": 25, "hand_notes_2b": 25, "hand_notes_4b": 41,
    "column_hits_2b": 13, "column_hits_4b": 21, "hand_bias_4b": 11, "hand_nps_4b": len(NPS_CENTERS),
    "same_column_gap": len(GAP_LABELS), "same_hand_gap": len(GAP_LABELS), "hand_run": 13,
    "chord_size": 5, "transition": len(TRANSITIONS), "snap": len(SNAP_LABELS), "row_gap": len(GAP_LABELS),
    "column": 4,
    # Definition-derived shapes: same-column run lengths (2 minijack, 3+ jack, 5+ longjack),
    # trill and roll lengths in notes, and the busiest column's share of a stream bar.
    "jack_run": 10, "trill_run": 17, "roll_run": 17, "anchor_share": 16,
}
SCALARS = ("rows", "notes", "holds", "active_bars", "bar_rows_sum", "bar_rows_square_sum", "long_jack_runs",
           "eligible_bar_pairs", "near_identical_bar_pairs", "motif_observations", "motif_unique",
           "stream_rows", "stream_chord_rows", "strong_beat_chord_rows", "grace_rows",
           "pattern_bars", "defined_bars", "family_stream_bars", "family_jack_bars", "family_trill_bars", "family_tech_bars")
# Histograms also kept per bar style for within-style comparison.
STYLE_FEATURES = ("hand_rows_2b", "hand_notes_4b", "column_hits_4b", "same_hand_gap", "chord_size", "transition", "row_gap", "snap")
GRACE_BEATS = 1 / 6
GRACE_MINIMUM_MS = 10.


def gap_bins(gaps):
    """Nearest musical gap on a log scale; gaps beyond four beats share the last bin."""
    gaps = np.asarray(gaps, dtype=np.float64)
    result = np.full(gaps.shape, len(GAP_VALUES), dtype=np.int64)
    valid = np.isfinite(gaps) & (gaps > 0) & (gaps <= 4 * 1.05)
    if valid.any():
        distance = np.abs(np.log2(gaps[valid, None]) - np.log2(np.asarray(GAP_VALUES))[None])
        result[valid] = distance.argmin(axis=1)
    return result


def snap_bins(clock, times):
    times = np.asarray(times, dtype=np.float64)
    offsets = np.asarray(clock.offsets, dtype=np.float64)
    lengths = np.asarray(clock.lengths, dtype=np.float64)
    section = np.maximum(np.searchsorted(offsets, times, side="right") - 1, 0)
    local = (times - offsets[section]) / lengths[section]
    fraction = local - np.floor(local)
    tolerance = TOLERANCE_MS / lengths[section]
    result = np.full(len(times), len(SNAP_DENOMINATORS), dtype=np.int64)
    for index, denominator in enumerate(SNAP_DENOMINATORS):
        scaled = fraction * denominator
        hit = (np.abs(scaled - np.rint(scaled)) <= tolerance * denominator) & (result == len(SNAP_DENOMINATORS))
        result[hit] = index
    return result


def _previous_gaps(beats, hits):
    """Beats since the previous hit of each lane/hand; NaN for its first hit."""
    gaps = np.full(hits.shape, np.nan)
    for lane in range(hits.shape[1]):
        rows = np.flatnonzero(hits[:, lane])
        if len(rows) > 1:
            gaps[rows[1:], lane] = np.diff(beats[rows])
    return gaps


def bar_styles(context, bars):
    """Definition matches and dominant style for every bar that has rows."""
    records = {}
    for bar in np.unique(bars):
        rows = np.flatnonzero(bars == bar)
        matches = bar_matches(context, rows)
        records[int(bar)] = {"rows": rows, "matches": matches, "style": dominant(matches)}
    return records


def phrase_consistency(records, first_bar):
    """Phrase-majority style per four bars and whether each bar meets that style's definition."""
    groups = {}
    for bar, record in records.items():
        groups.setdefault((bar - first_bar) // 4, []).append(bar)
    result = []
    for bars in groups.values():
        styles = [records[bar]["style"] for bar in bars if records[bar]["style"]]
        if len(styles) < 2:
            continue
        labels, counts = np.unique(styles, return_counts=True)
        phrase = str(labels[counts.argmax()])
        for bar in bars:
            if len(records[bar]["rows"]) >= 4:
                result.append((bar, phrase, bool(records[bar]["matches"][phrase])))
    return result


def family_shares(segment):
    totals = np.asarray([segment["scalars"][f"family_{family}_bars"] for family in FAMILIES], dtype=np.float64)
    return (totals / totals.sum()).tolist() if totals.sum() > 0 else [1 / 3] * 3


def extract_blocks(clock, times, masks, holds):
    """Return per-16-bar histograms, additive scalars, style counts and per-style histograms."""
    times = np.asarray(times, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.int64)
    if not len(times):
        raise ValueError("Evaluation needs at least one row.")
    context = chart_context(clock, times, masks)
    beats = context.beats
    offsets = np.asarray(clock.offsets, dtype=np.float64)
    lengths = np.asarray(clock.lengths, dtype=np.float64)
    seconds_per_beat = lengths[np.maximum(np.searchsorted(offsets, times, side="right") - 1, 0)] / 1000
    bars = np.floor(beats / BAR_BEATS + 1e-9).astype(np.int64)
    first_bar = int(bars.min())
    block_of_row = (bars - first_bar) // BLOCK_BARS
    count = int(block_of_row.max()) + 1
    histograms = {name: np.zeros((count, bins), dtype=np.int64) for name, bins in HISTOGRAMS.items()}
    style_histograms = {name: np.zeros((count, len(LABELS), HISTOGRAMS[name]), dtype=np.int64) for name in STYLE_FEATURES}
    scalars = {name: np.zeros(count, dtype=np.int64) for name in SCALARS}
    records = bar_styles(context, bars)
    row_style = np.asarray([LABELS.index(records[int(bar)]["style"]) if records[int(bar)]["style"] else -1 for bar in bars], dtype=np.int64)

    def add(name, values, blocks, styles=None):
        bins = np.clip(np.asarray(values, dtype=np.int64), 0, HISTOGRAMS[name] - 1)
        np.add.at(histograms[name], (blocks, bins), 1)
        if name in style_histograms and styles is not None:
            styled = styles >= 0
            np.add.at(style_histograms[name], (blocks[styled], styles[styled], bins[styled]), 1)

    columns = ((masks[:, None] >> np.arange(4)) & 1).astype(np.int64)
    hand_notes = np.stack((columns[:, :2].sum(1), columns[:, 2:].sum(1)), axis=1)
    hand_part = hand_notes > 0
    rows = np.arange(len(times))
    windows = {}
    for window in PRESSURE_WINDOWS_BEATS:
        # Half-open (now - window, now]: the current row is included, older state is not.
        lower = np.searchsorted(beats, beats - window + 1e-6, side="left")
        sums = []
        for values in (columns, hand_part.astype(np.int64), hand_notes):
            cumulative = np.concatenate((np.zeros((1, values.shape[1]), dtype=np.int64), np.cumsum(values, axis=0)))
            sums.append(cumulative[rows + 1] - cumulative[lower])
        windows[window] = sums
    hand_blocks = np.broadcast_to(block_of_row[:, None], hand_part.shape)[hand_part]
    column_blocks = np.broadcast_to(block_of_row[:, None], columns.shape)[columns > 0]
    hand_styles = np.broadcast_to(row_style[:, None], hand_part.shape)[hand_part]
    column_styles = np.broadcast_to(row_style[:, None], columns.shape)[columns > 0]
    for window, suffix in ((2.0, "2b"), (4.0, "4b")):
        column_hits, hand_rows, notes = windows[window]
        add("hand_rows_" + suffix, hand_rows[hand_part], hand_blocks, hand_styles)
        add("hand_notes_" + suffix, notes[hand_part], hand_blocks, hand_styles)
        add("column_hits_" + suffix, column_hits[columns > 0], column_blocks, column_styles)
    four_notes = windows[4.0][2]
    share = four_notes.max(1) / np.maximum(1, four_notes.sum(1))
    add("hand_bias_4b", np.rint((share - .5) / .05), block_of_row)
    nps = four_notes / (4 * seconds_per_beat[:, None])
    add("hand_nps_4b", np.searchsorted(NPS_EDGES, nps[hand_part], side="right"), hand_blocks)
    column_gaps = _previous_gaps(beats, columns > 0)
    add("same_column_gap", gap_bins(column_gaps[columns > 0]), column_blocks)
    hand_gaps = _previous_gaps(beats, hand_part)
    add("same_hand_gap", gap_bins(hand_gaps[hand_part]), hand_blocks, hand_styles)
    gaps = np.diff(beats, prepend=-np.inf)
    continuous = gaps <= .5 + 1e-6
    runs, run_values = np.zeros(2, dtype=np.int64), np.zeros(hand_part.shape, dtype=np.int64)
    for index in rows:
        runs = np.where(hand_part[index], np.where(continuous[index], runs + 1, 1), 0)
        run_values[index] = runs
    # Bound participation runs by the 4-beat horizon as well.
    run_values = np.minimum(run_values, windows[4.0][1])
    add("hand_run", run_values[hand_part], hand_blocks)
    sizes = columns.sum(1)
    add("chord_size", sizes, block_of_row, row_style)
    linked = np.flatnonzero(gaps[1:] <= 1 + 1e-6) + 1
    add("transition", TRANSITION_TABLE[masks[linked - 1], masks[linked]], block_of_row[linked], row_style[linked])
    add("snap", snap_bins(clock, times), block_of_row, row_style)
    add("row_gap", gap_bins(gaps[1:]), block_of_row[1:], row_style[1:])
    for column in range(4):
        np.add.at(histograms["column"][:, column], block_of_row[columns[:, column] > 0], 1)
    # Definition shapes: run lengths in notes, assigned to the block where a run ends.
    for _, first, last in context.runs:
        add("jack_run", [last - first + 1], block_of_row[[last]])
    for first, last, _ in context.trills:
        add("trill_run", [last - first + 1], block_of_row[[last]])
    for first, last in context.rolls:
        add("roll_run", [last - first + 1], block_of_row[[last]])
    beat_ms = seconds_per_beat * 1000
    np.add.at(scalars["stream_rows"], block_of_row, context.stream.astype(np.int64))
    stream_chords = context.stream & (sizes >= 2)
    strong = np.abs(context.fraction * 2 - np.rint(context.fraction * 2)) <= 2 * TOLERANCE_MS / beat_ms
    np.add.at(scalars["stream_chord_rows"], block_of_row, stream_chords.astype(np.int64))
    np.add.at(scalars["strong_beat_chord_rows"], block_of_row, (stream_chords & strong).astype(np.int64))
    # Grace: an isolated very short gap (1/6 beat or less) between otherwise slower rows.
    gap_ms = context.gap_ms
    following = np.concatenate((gap_ms[1:], [np.inf]))
    grace = ((gap_ms > GRACE_MINIMUM_MS) & (gaps <= GRACE_BEATS + TOLERANCE_MS / beat_ms)
             & (np.concatenate(([np.inf], gap_ms[:-1])) >= 2 * gap_ms) & (following >= 2 * gap_ms))
    np.add.at(scalars["grace_rows"], block_of_row, grace.astype(np.int64))
    style_bars = [dict() for _ in range(count)]
    for bar, record in records.items():
        block = (bar - first_bar) // BLOCK_BARS
        rows_in_bar = record["rows"]
        if len(rows_in_bar) < 4:
            continue
        scalars["pattern_bars"][block] += 1
        if float(context.stream[rows_in_bar].mean()) >= .6:
            hits = columns[rows_in_bar].sum(0)
            add("anchor_share", [round((hits.max() / hits.sum() - .25) / .05)], np.asarray([block]))
        label = record["style"]
        if label:
            scalars["defined_bars"][block] += 1
            style_bars[block][label] = style_bars[block].get(label, 0) + 1
            family = next(name for name, labels in FAMILIES.items() if label in labels)
            scalars[f"family_{family}_bars"][block] += 1

    np.add.at(scalars["rows"], block_of_row, 1)
    np.add.at(scalars["notes"], block_of_row, sizes)
    np.add.at(scalars["holds"], block_of_row, np.asarray(holds, dtype=np.int64))
    unique_bars, bar_rows = np.unique(bars, return_counts=True)
    bar_blocks = (unique_bars - first_bar) // BLOCK_BARS
    np.add.at(scalars["active_bars"], bar_blocks, 1)
    np.add.at(scalars["bar_rows_sum"], bar_blocks, bar_rows)
    np.add.at(scalars["bar_rows_square_sum"], bar_blocks, bar_rows ** 2)
    # Ordinary long jacks: five or more same-column hits at consecutive 1/4 spacing.
    for column in range(4):
        hit_rows = np.flatnonzero(columns[:, column])
        if len(hit_rows) < 5:
            continue
        sixteenth = np.abs(np.diff(beats[hit_rows]) - .25) <= TOLERANCE_MS / (seconds_per_beat[hit_rows[1:]] * 1000)
        length = 1
        for position, adjacent in enumerate(sixteenth, start=1):
            if adjacent:
                length += 1
                continue
            if length >= 5:
                scalars["long_jack_runs"][block_of_row[hit_rows[position - 1]]] += 1
            length = 1
        if length >= 5:
            scalars["long_jack_runs"][block_of_row[hit_rows[-1]]] += 1
    # Bar layouts on the sixteenth grid: adjacent near-copies and motif variety.
    position = beats - bars * BAR_BEATS
    slot = np.rint(position * 4)
    on_grid = np.abs(position * 4 - slot) <= TOLERANCE_MS / (seconds_per_beat * 1000) * 4
    grids = {}
    for index in np.flatnonzero(on_grid & (slot >= 0) & (slot < 16)):
        grids.setdefault(int(bars[index]), np.zeros(16, dtype=np.uint8))[int(slot[index])] |= masks[index]
    motifs = [set() for _ in range(count)]
    for bar, grid in grids.items():
        block = (bar - first_bar) // BLOCK_BARS
        filled = np.count_nonzero(grid)
        if filled >= 4:
            scalars["motif_observations"][block] += 1
            motifs[block].add(canonical_motif(grid))
        following = grids.get(bar + 1)
        if following is not None and min(filled, np.count_nonzero(following)) >= 8:
            scalars["eligible_bar_pairs"][block] += 1
            similarity = max(float(np.mean(grid == following)), float(np.mean(MIRROR[grid] == following)))
            scalars["near_identical_bar_pairs"][block] += similarity >= .875
    for block, values in enumerate(motifs):
        scalars["motif_unique"][block] = len(values)
    style = [dict() for _ in range(count)]
    for bar, label, passed in phrase_consistency(records, first_bar):
        record = style[(bar - first_bar) // BLOCK_BARS].setdefault(label, [0, 0])
        record[0] += 1
        record[1] += passed
    blocks = []
    for block in range(count):
        styled = {LABELS[index]: {name: values[block, index].tolist() for name, values in style_histograms.items()}
                  for index in range(len(LABELS)) if LABELS[index] in style_bars[block]}
        blocks.append({"hist": {name: values[block].tolist() for name, values in histograms.items()},
                       "scalars": {name: int(values[block]) for name, values in scalars.items()},
                       "style": style[block], "style_bars": style_bars[block], "style_hist": styled})
    return blocks


def combine(blocks):
    """Sum block observations into one segment record."""
    histograms = {name: np.zeros(bins, dtype=np.int64) for name, bins in HISTOGRAMS.items()}
    scalars = dict.fromkeys(SCALARS, 0)
    style, style_bars, style_hist = {}, {}, {}
    for block in blocks:
        for name in HISTOGRAMS:
            histograms[name] += np.asarray(block["hist"][name], dtype=np.int64)
        for name in SCALARS:
            scalars[name] += block["scalars"][name]
        for label, (bars, passed) in block["style"].items():
            record = style.setdefault(label, [0, 0])
            record[0] += bars
            record[1] += passed
        for label, bars in block["style_bars"].items():
            style_bars[label] = style_bars.get(label, 0) + bars
            target = style_hist.setdefault(label, {name: np.zeros(HISTOGRAMS[name], dtype=np.int64) for name in STYLE_FEATURES})
            for name in STYLE_FEATURES:
                target[name] += np.asarray(block["style_hist"][label][name], dtype=np.int64)
    return {"hist": histograms, "scalars": scalars, "style": style, "style_bars": style_bars, "style_hist": style_hist}


def _mean(histogram, values=None):
    histogram = np.asarray(histogram, dtype=np.float64)
    total = histogram.sum()
    if total <= 0:
        return None
    values = np.arange(len(histogram), dtype=np.float64) if values is None else np.asarray(values, dtype=np.float64)
    return float((histogram * values).sum() / total)


def _quantile(histogram, quantile):
    histogram = np.asarray(histogram, dtype=np.float64)
    if histogram.sum() <= 0:
        return None
    return float(np.searchsorted(np.cumsum(histogram) / histogram.sum(), quantile - 1e-12))


def _ratio(numerator, denominator, minimum):
    return numerator / denominator if denominator >= minimum else None


def scalar_metrics(segment):
    """Chart-level values derived only from additive observations."""
    histogram, scalar = segment["hist"], segment["scalars"]
    transitions = histogram["transition"]
    jack = sum(value for token, value in zip(TRANSITIONS, transitions) if token[2] > 0)
    column = np.asarray(histogram["column"], dtype=np.float64)
    entropy = None
    if column.sum() > 0:
        probability = column[column > 0] / column.sum()
        entropy = float(-(probability * np.log(probability)).sum() / math.log(4))
    active = scalar["active_bars"]
    density_cv = None
    if active >= 4 and scalar["bar_rows_sum"] > 0:
        mean = scalar["bar_rows_sum"] / active
        variance = max(0., scalar["bar_rows_square_sum"] / active - mean ** 2)
        density_cv = math.sqrt(variance) / mean
    style_bars = sum(bars for bars, _ in segment["style"].values())
    style_passes = sum(passed for _, passed in segment["style"].values())
    return {
        "hand_rows_2b_mean": _mean(histogram["hand_rows_2b"]),
        "hand_notes_4b_mean": _mean(histogram["hand_notes_4b"]),
        "hand_notes_4b_p90": _quantile(histogram["hand_notes_4b"], .9),
        "column_hits_4b_p90": _quantile(histogram["column_hits_4b"], .9),
        "hand_nps_4b_mean": _mean(histogram["hand_nps_4b"], NPS_CENTERS),
        "hand_bias_4b_mean": _mean(histogram["hand_bias_4b"], .5 + .05 * np.arange(11)),
        "jack_transition_fraction": _ratio(jack, int(np.sum(transitions)), 16),
        "column_entropy": entropy,
        "long_jack_runs_per_100_bars": 100 * scalar["long_jack_runs"] / active if active >= 8 else None,
        "near_identical_bar_fraction": _ratio(scalar["near_identical_bar_pairs"], scalar["eligible_bar_pairs"], 4),
        "motif_unique_fraction": _ratio(scalar["motif_unique"], scalar["motif_observations"], 8),
        "bar_density_cv": density_cv,
        "style_self_consistency": _ratio(style_passes, style_bars, 8),
        "left_hand_share": float(column[:2].sum() / column.sum()) if column.sum() > 0 else None,
        "stream_regularity": _ratio(scalar["stream_rows"], scalar["rows"], 32),
        "chord_strong_beat_fraction": _ratio(scalar["strong_beat_chord_rows"], scalar["stream_chord_rows"], 16),
        "grace_per_100_rows": 100 * scalar["grace_rows"] / scalar["rows"] if scalar["rows"] >= 64 else None,
        "defined_bar_fraction": _ratio(scalar["defined_bars"], scalar["pattern_bars"], 8),
    }
