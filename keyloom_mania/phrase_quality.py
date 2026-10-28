import collections
import math

import numpy as np


MIRROR = np.asarray([sum(((mask >> column) & 1) << (3 - column) for column in range(4)) for mask in range(16)], dtype=np.uint8)
COUNT_TABLE = np.asarray([mask.bit_count() for mask in range(16)], dtype=np.int64)


def canonical_motif(values):
    values = np.asarray(values, dtype=np.uint8)
    return min(values.tobytes(), MIRROR[values].tobytes())


def diversity_summary(counter):
    counts = np.asarray(sorted(counter.values(), reverse=True), dtype=np.float64)
    if not len(counts):
        return {"observations": 0, "unique": 0, "top1_fraction": None, "top5_fraction": None, "effective_variants": 0}
    probabilities = counts / counts.sum()
    return {"observations": int(counts.sum()), "unique": len(counts), "top1_fraction": float(probabilities[0]),
            "top5_fraction": float(probabilities[:5].sum()), "effective_variants": float(np.exp(-(probabilities * np.log(probabilities)).sum()))}


def periodic_runs(values, periods=(2, 4, 8, 16, 32), minimum_rows=16):
    values = np.asarray(values, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 15):
        raise ValueError("Periodic analysis requires a one-dimensional mask grid.")
    reports = []
    for period in periods:
        if period < 1:
            raise ValueError("Periods must be positive.")
        equal = values[period:] == values[:-period]
        padded = np.concatenate(([False], equal, [False])).astype(np.int8)
        starts, stops = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
        candidates = []
        for start, stop in zip(starts, stops):
            length = int(stop - start + period)
            motif = values[start:start + period]
            if length < max(minimum_rows, 2 * period) or np.count_nonzero(motif) < max(2, period // 2):
                continue
            if len(np.unique(motif)) < 2:
                continue
            candidates.append({"start_row": int(start), "rows": length, "bars": length / 16, "motif": motif.tolist()})
        candidates.sort(key=lambda row: row["rows"], reverse=True)
        reports.append({"period_rows": period, "longest_bars": candidates[0]["bars"] if candidates else 0.0, "runs": candidates[:5]})
    return reports


def chart_grid(clock, notes):
    grids = [collections.defaultdict(int) for _ in clock.sections]
    off_grid = 0
    for note in notes:
        if note.end_ms is not None:
            continue
        ordinal = clock.section_at(note.time_ms)
        section = clock.sections[ordinal]
        coordinate = (note.time_ms - section.offset_ms) * 4 / section.beat_length_ms
        index = round(coordinate)
        if abs(note.time_ms - section.offset_ms - index * section.beat_length_ms / 4) > 1.01:
            off_grid += 1
            continue
        grids[ordinal][index] |= 1 << note.column
    return grids, off_grid


def chart_phrase_metrics(clock, notes):
    grids, off_grid = chart_grid(clock, notes)
    motifs, count_motifs = collections.Counter(), collections.Counter()
    periods = []
    near_equal, eligible_pairs, repeated_streak, longest_streak = 0, 0, 1, 1
    for ordinal, grid in enumerate(grids):
        if not grid:
            continue
        lower, upper = min(0, min(grid)), max(grid) + 1
        values = np.asarray([grid[index] for index in range(lower, upper)], dtype=np.uint8)
        for row in periodic_runs(values):
            row["section"] = ordinal
            for run in row["runs"]:
                run["start_ms"] = float(clock.sections[ordinal].offset_ms + (lower + run["start_row"]) * clock.sections[ordinal].beat_length_ms / 4)
            periods.append(row)
        bars = [values[position:position + 16] for position in range(0, len(values) - 15, 16)]
        for bar in bars:
            if np.count_nonzero(bar) >= 4:
                motifs[canonical_motif(bar)] += 1
                count_motifs[COUNT_TABLE[bar].astype(np.uint8).tobytes()] += 1
        repeated_streak = 1
        for before, after in zip(bars[:-1], bars[1:]):
            if min(np.count_nonzero(before), np.count_nonzero(after)) < 8:
                repeated_streak = 1
                continue
            eligible_pairs += 1
            similarity = max(float(np.mean(before == after)), float(np.mean(MIRROR[before] == after)))
            near_equal += similarity >= 0.875
            repeated_streak = repeated_streak + 1 if similarity >= 0.875 else 1
            longest_streak = max(longest_streak, repeated_streak)
    return {"longest_exact_periodic_bars": max((row["longest_bars"] for row in periods), default=0), "periods": periods,
            "near_identical_adjacent_bar_fraction": near_equal / max(1, eligible_pairs), "near_identical_bar_streak": longest_streak,
            "bar_layout_diversity_mirror_canonical": diversity_summary(motifs), "bar_count_diversity": diversity_summary(count_motifs),
            "off_grid_taps": off_grid, "limitations": "Periodic metrics project onto the fixed sixteenth grid; off-grid taps are separately counted. Mirror-canonical comparison does not authorize arbitrary column permutations."}


def log_power(frames):
    values = np.asarray(frames, dtype=np.float64)
    maximum = values.max(axis=-1)
    return maximum + np.log(np.exp(values - maximum[..., None]).mean(axis=-1).clip(1e-30))


def audio_beat_summary(frames, start_ms, beat_length_ms):
    lower = max(0, int(math.floor(start_ms / 5)))
    upper = min(len(frames), int(math.ceil((start_ms + beat_length_ms) / 5)))
    if upper <= lower:
        return np.zeros(6, dtype=np.float32)
    source = np.asarray(frames[lower:upper], dtype=np.float32)
    energy = source[:, 32:36].mean(axis=-1)
    flux = source[:, 36:40].mean(axis=-1)
    return np.asarray([energy.mean(), energy.max(), energy.std(), flux.mean(), flux.max(), np.mean(energy > energy.max() - 2)], dtype=np.float32)


def density_expression_metrics(clock, notes, bars, rhythm_frames):
    times = np.asarray([note.time_ms for note in notes if note.end_ms is None], dtype=np.float64)
    unique = np.unique(times)
    details = []
    for bar in bars:
        section = clock.sections[int(bar["section"])]
        start = float(bar["start_ms"])
        beats = []
        for beat in range(4):
            begin = start + beat * section.beat_length_ms
            end = begin + section.beat_length_ms
            count = int(np.count_nonzero((times >= begin) & (times < end)))
            rows = int(np.count_nonzero((unique >= begin) & (unique < end)))
            beats.append({"notes": count, "rows": rows, "audio": audio_beat_summary(rhythm_frames, begin, section.beat_length_ms).tolist()})
        energy = float(np.mean([beat["audio"][0] for beat in beats]))
        details.append({"start_ms": start, "energy": energy, "notes": sum(beat["notes"] for beat in beats), "peak_beat_notes": max(beat["notes"] for beat in beats), "beats": beats})
    if not details:
        return {"bars": [], "quiet_mean_notes": None, "active_mean_notes": None}
    energies = np.asarray([bar["energy"] for bar in details])
    quiet_limit, active_limit = np.quantile(energies, [0.25, 0.75])
    quiet = [bar for bar in details if bar["energy"] <= quiet_limit]
    active = [bar for bar in details if bar["energy"] >= active_limit]
    quiet_notes = float(np.mean([bar["notes"] for bar in quiet]))
    active_notes = float(np.mean([bar["notes"] for bar in active]))
    return {"quiet_mean_notes": quiet_notes, "active_mean_notes": active_notes, "active_minus_quiet_notes": active_notes - quiet_notes,
            "quiet_peak_beat_notes_p95": float(np.percentile([bar["peak_beat_notes"] for bar in quiet], 95)),
            "quiet_one_beat_concentration_mean": float(np.mean([bar["peak_beat_notes"] / max(1, bar["notes"]) for bar in quiet])),
            "quiet_bars": len(quiet), "active_bars": len(active), "bars": details,
            "policy": "Within-song pooled raw log-Mel energy descriptor quartiles; no claim of calibrated waveform loudness. Counts preserve all off-grid events; no hard allocation budget."}
