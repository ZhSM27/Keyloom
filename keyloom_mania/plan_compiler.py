from __future__ import annotations

import collections

import numpy as np

from .segment_labels import LABELS


COMPILER_CONFIG = {
    "schema_version": "rice-plan-compiler-v1",
    "smoothing_bars": 3,
    "minimum_run_bars": 1,
    "specific_minimum_run_bars": 2,
    "cooldown_bars": 1,
    "maximum_active_labels": 4,
    "specific_minimum_labels": ("Handstream", "Quadstream"),
    "fallback_on_nonempty": True,
}

COMPILER_CONFIG_V2 = {
    "schema_version": "rice-plan-compiler-v2",
    "smoothing_bars": 5,
    "minimum_run_bars": 4,
    "specific_minimum_run_bars": 4,
    "cooldown_bars": 2,
    "maximum_active_labels": 3,
    "specific_minimum_labels": ("Handstream", "Quadstream", "Chordjacks", "Longjacks"),
    "quiet_rows_per_beat": 0.20,
    "quiet_notes_per_beat": 0.35,
    "switch_change_threshold": 0.12,
    "fallback_minimum_weight": 0.25,
}

JACK_LABELS = frozenset(("Jacks", "Longjacks", "Chordjacks", "Minijacks", "Gluts"))
PATTERN_FAMILIES = {
    "jack": frozenset(("Jacks", "Longjacks", "Chordjacks", "Minijacks", "Gluts")),
    "stream": frozenset(("Stream", "Chordstream", "Jumpstream", "Handstream", "Quadstream")),
    "trill": frozenset(("Rolls", "Trills", "Minitrills", "Split Trill", "Jumptrill")),
}


def smoothed(weights, kernel: int) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(LABELS):
        raise ValueError("Pattern weights must be bars by fifteen labels.")
    if kernel < 1:
        raise ValueError("Smoothing kernel must be positive.")
    if kernel == 1 or len(values) < 2:
        return values.copy()
    half = kernel // 2
    padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
    result = np.empty_like(values)
    for position in range(len(values)):
        result[position] = padded[position : position + kernel].mean(axis=0)
    return result


def _remove_short_runs(mask: np.ndarray, minimum: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if minimum <= 1:
        return result
    start = 0
    while start < len(result):
        if not result[start]:
            start += 1
            continue
        end = start + 1
        while end < len(result) and result[end]:
            end += 1
        if end - start < minimum:
            result[start:end] = False
        start = end
    return result


def _cooldown(mask: np.ndarray, cooldown: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if cooldown <= 0:
        return result
    remaining = 0
    was_active = False
    for position, active in enumerate(result):
        if active:
            was_active = True
            remaining = cooldown
        elif was_active:
            if remaining > 0:
                result[position] = False
                remaining -= 1
            else:
                was_active = False
    return result


def _bridge_short_gaps(mask: np.ndarray, gap: int) -> np.ndarray:
    """Keep a label on across off-runs shorter than the cooldown.

    The previous compiler only looked forward from the end of a run, which
    meant it could hide the first already-inactive bar but could not prevent a
    one-bar flicker.  Bridging short gaps is the behavior we actually want for
    a segment plan: a label that is active immediately before and after a short
    gap is treated as continuous.
    """
    result = np.asarray(mask, dtype=bool).copy()
    if gap <= 0 or len(result) < 3:
        return result
    start = 0
    while start < len(result):
        if result[start]:
            start += 1
            continue
        end = start + 1
        while end < len(result) and not result[end]:
            end += 1
        if start > 0 and end < len(result) and result[start - 1] and result[end] and end - start < gap:
            result[start:end] = True
        start = end
    return result


def _top_k_filter(weights: np.ndarray, mask: np.ndarray, maximum: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if maximum < 1:
        raise ValueError("Maximum active labels must be positive.")
    for position in range(len(result)):
        active = np.flatnonzero(result[position])
        if len(active) > maximum:
            keep = active[np.argsort(weights[position, active])[-maximum:]]
            result[position] = False
            result[position, keep] = True
    return result


def fit_thresholds(weights, reference, available) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    available = np.asarray(available, dtype=bool).reshape(-1)
    if weights.ndim != 2 or weights.shape[1] != len(LABELS) or reference.shape != weights.shape or available.shape != (len(weights),):
        raise ValueError("Threshold fitting needs aligned pattern weights and references.")
    result = np.full(len(LABELS), 1.1, dtype=np.float32)
    for label in range(len(LABELS)):
        known = available & (reference[:, label] >= 0)
        if not known.any():
            continue
        target_rate = float((reference[known, label] > 0).mean())
        values = np.sort(weights[known, label])
        if target_rate <= 0:
            continue
        position = int(np.clip(np.floor((1 - target_rate) * len(values)), 0, len(values) - 1))
        result[label] = float(np.clip(values[position], 0.05, 0.95))
    return result


def compile_plan(weights, notes_per_beat, rows_per_beat, thresholds, config=None) -> dict:
    config = {**COMPILER_CONFIG, **(config or {})}
    weights = np.asarray(weights, dtype=np.float32)
    notes = np.asarray(notes_per_beat, dtype=np.float32)
    rows = np.asarray(rows_per_beat, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if weights.ndim != 2 or weights.shape != (len(notes), len(LABELS)) or rows.shape != notes.shape or thresholds.shape != (len(LABELS),):
        raise ValueError("PlanCompiler inputs have mismatched bar or label dimensions.")
    smoothed_weights = smoothed(weights, int(config["smoothing_bars"]))
    active = smoothed_weights >= thresholds[None, :]
    minimum = {
        ordinal: int(config["specific_minimum_run_bars"])
        if label in config["specific_minimum_labels"]
        else int(config["minimum_run_bars"])
        for ordinal, label in enumerate(LABELS)
    }
    for label in range(len(LABELS)):
        active[:, label] = _remove_short_runs(active[:, label], minimum[label])
        active[:, label] = _cooldown(active[:, label], int(config["cooldown_bars"]))
    active = _top_k_filter(smoothed_weights, active, int(config["maximum_active_labels"]))
    if config["fallback_on_nonempty"]:
        for position in range(len(active)):
            if notes[position] > 0.5 and not active[position].any():
                active[position, int(np.argmax(smoothed_weights[position]))] = True
    return {
        "patterns": active.astype(np.uint8),
        "smoothed_weights": smoothed_weights.astype(np.float16),
        "target_notes_per_beat": notes.astype(np.float16),
        "target_rows_per_beat": rows.astype(np.float16),
        "target_row_counts": np.clip(np.rint(rows), 0, 4).astype(np.uint8),
        "thresholds": thresholds.astype(np.float16),
        "config": config,
    }


def to_eight_beat_windows(plan: np.ndarray) -> np.ndarray:
    plan = np.asarray(plan, dtype=np.uint8)
    if plan.ndim != 2 or plan.shape[1] != len(LABELS):
        raise ValueError("Only bar-by-label plans can be expanded.")
    count = math_ceil_div(len(plan), 2)
    result = np.zeros((count, 8, len(LABELS)), dtype=np.uint8)
    for window in range(count):
        first = window * 2
        second = min(len(plan) - 1, first + 1)
        result[window, 0:4] = plan[first][None, :]
        result[window, 4:8] = plan[second][None, :]
    return result


def math_ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _family_mask(candidates: np.ndarray, weights: np.ndarray, config: dict) -> np.ndarray:
    result = candidates.copy()
    active = np.flatnonzero(result)
    if not len(active):
        return result
    top = active[np.argmax(weights[active])]
    top_label = LABELS[int(top)]
    if top_label in JACK_LABELS:
        preferred = [ordinal for ordinal, label in enumerate(LABELS) if label in JACK_LABELS]
        allowed = result[preferred]
        if allowed.any():
            filtered = np.zeros_like(result)
            filtered[preferred] = allowed
            result = filtered
    else:
        preferred = [ordinal for ordinal, label in enumerate(LABELS) if label not in JACK_LABELS]
        allowed = result[preferred]
        if allowed.any():
            filtered = np.zeros_like(result)
            filtered[preferred] = allowed
            result = filtered
    return _top_k_filter(weights, result, int(config["maximum_active_labels"]))


def _label_family(label: str) -> str:
    for family, labels in PATTERN_FAMILIES.items():
        if label in labels:
            return family
    raise ValueError(f"Unknown pattern family for {label}.")


def _dominant_family(weights: np.ndarray, candidates: np.ndarray) -> str:
    active = np.flatnonzero(candidates)
    if not len(active):
        active = np.asarray([int(np.argmax(weights))], dtype=np.int64)
    scores = collections.defaultdict(float)
    for ordinal in active:
        scores[_label_family(LABELS[int(ordinal)])] += float(weights[int(ordinal)])
    return max(scores, key=scores.get)


def _coarse_family(label: str) -> str:
    return "jack" if label in JACK_LABELS else "other"


def _segment_active(segment_weights: np.ndarray, thresholds: np.ndarray, config: dict) -> np.ndarray:
    candidates = np.asarray(segment_weights, dtype=np.float32) >= thresholds
    if not candidates.any():
        if float(np.max(segment_weights)) < float(config["fallback_minimum_weight"]):
            return np.zeros(len(LABELS), dtype=np.uint8)
        candidates[int(np.argmax(segment_weights))] = True
    family = _coarse_family(LABELS[int(np.argmax(np.where(candidates, segment_weights, -np.inf)))])
    same_family = np.asarray([_coarse_family(label) == family for label in LABELS], dtype=bool)
    filtered = candidates & same_family
    if filtered.any():
        candidates = filtered
    candidates = _top_k_filter(segment_weights[None, :], candidates[None, :], int(config["maximum_active_labels"]))[0]
    return candidates.astype(np.uint8)


def _change_boundaries(smoothed_weights: np.ndarray, rows: np.ndarray, config: dict) -> list[int]:
    """Find sparse, local change points inside non-quiet stretches.

    The smooth weights already average five bars, so a true phrase transition
    produces a broad local maximum instead of a single noisy spike.  Requiring a
    local maximum and a minimum segment length prevents a one-bar detector
    fluctuation from changing the plan.
    """
    if len(smoothed_weights) < 3:
        return []
    weight_change = np.zeros(len(smoothed_weights), dtype=np.float32)
    weight_change[1:] = np.abs(np.diff(smoothed_weights, axis=0)).max(axis=1)
    row_change = np.zeros(len(smoothed_weights), dtype=np.float32)
    row_change[1:] = np.abs(np.diff(rows)) / np.maximum(1.0, rows[1:] + rows[:-1])
    threshold = float(config["switch_change_threshold"])
    candidates = []
    for position in range(1, len(smoothed_weights) - 1):
        local = weight_change[position] >= weight_change[position - 1] and weight_change[position] >= weight_change[position + 1]
        if local and weight_change[position] >= threshold:
            candidates.append(position)
        elif local and row_change[position] >= 0.35:
            candidates.append(position)
    accepted = []
    for position in candidates:
        if not accepted or position - accepted[-1] >= int(config["minimum_run_bars"]):
            accepted.append(position)
    return accepted


def compile_plan_v2(weights, notes_per_beat, rows_per_beat, thresholds, config=None) -> dict:
    """Compile a stable, silence-aware plan.

    The old compiler recomputed an independent top-k set on every bar. This
    version splits non-quiet stretches at local change points, chooses one
    family and one small label set per stretch, and clears low-density bars.
    """
    config = {**COMPILER_CONFIG_V2, **(config or {})}
    weights = np.asarray(weights, dtype=np.float32)
    notes = np.asarray(notes_per_beat, dtype=np.float32)
    rows = np.asarray(rows_per_beat, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if weights.ndim != 2 or weights.shape != (len(notes), len(LABELS)) or rows.shape != notes.shape or thresholds.shape != (len(LABELS),):
        raise ValueError("PlanCompiler v2 inputs have mismatched bar or label dimensions.")
    smooth = smoothed(weights, int(config["smoothing_bars"]))
    quiet = (rows <= float(config["quiet_rows_per_beat"])) & (notes <= float(config["quiet_notes_per_beat"]))
    active = np.zeros((len(smooth), len(LABELS)), dtype=np.uint8)
    segment_ids = np.full(len(smooth), -1, dtype=np.int32)
    segment_index = 0
    position = 0
    while position < len(smooth):
        if quiet[position]:
            position += 1
            continue
        stop = position + 1
        while stop < len(smooth) and not quiet[stop]:
            stop += 1
        stretch_weights = smooth[position:stop]
        boundaries = [0, *_change_boundaries(stretch_weights, rows[position:stop], config), stop - position]
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            if upper <= lower:
                continue
            segment = slice(position + lower, position + upper)
            segment_weights = smooth[segment].mean(axis=0)
            active[segment] = _segment_active(segment_weights, thresholds, config)
            segment_ids[segment] = segment_index
            segment_index += 1
        position = stop
    for label in range(len(LABELS)):
        active[:, label] = _bridge_short_gaps(active[:, label] > 0, int(config["cooldown_bars"]))
    active[quiet] = 0
    active = _top_k_filter(smooth, active > 0, int(config["maximum_active_labels"]))
    return {
        "patterns": active,
        "smoothed_weights": smooth.astype(np.float16),
        "target_notes_per_beat": notes.astype(np.float16),
        "target_rows_per_beat": rows.astype(np.float16),
        "target_row_counts": np.clip(np.rint(rows * 4), 0, 96).astype(np.uint8),
        "target_row_budget": np.clip(rows, 0, 24).astype(np.float16),
        "target_note_budget": np.clip(notes, 0, 96).astype(np.float16),
        "quiet": quiet.astype(np.uint8),
        "segment_ids": segment_ids,
        "thresholds": thresholds.astype(np.float16),
        "config": config,
    }
