"""Generator-native density contracts and observational execution accounting.

These are necessary feasibility bounds, never musical quality or learned capacity.
The optional budget changes only neural row conditions, not events or count logits.
"""
from __future__ import annotations

import numpy as np
from bisect import bisect_left

from .segment_labels import LABELS

BUDGET_VERSION = "generator-density-budget-v1"
DIAGNOSTIC_VERSION = "density-execution-stages-v1"
BUDGET_OPTIONS = (
    {"id": "legacy", "label": "原密度条件 + 执行诊断", "status": "baseline", "available": True},
    {"id": "feasible", "label": "生成器感知预算（未通过回归）", "status": "research", "available": False,
     "unavailable_reason": "整曲配对回归中 Stream 通过率下降 9.375 个百分点；候选仅保留离线，不在工作台启用。"},
    {"id": "count-reference", "label": "JS 音乐行数与押数协调", "status": "research", "available": True,
     "supported_generator_ids": ["music-base"],
     "supported_main_styles": ["Jumpstream"],
     "notice": "仅限 Music Base r1 的 JS 主型；保留原落点，默认仍为原密度条件，需试听试玩。"},
)


def validate_budget(value="legacy", *, for_studio=False, generator_id=None, main_style=None):
    if not isinstance(value, str) or value not in {item["id"] for item in BUDGET_OPTIONS}:
        raise ValueError("请选择有效的生成器密度预算方案。")
    if for_studio:
        option = next(item for item in BUDGET_OPTIONS if item['id'] == value)
        if not option['available']:
            raise ValueError(option['unavailable_reason'])
        if option.get('supported_generator_ids') and generator_id not in option['supported_generator_ids']:
            raise ValueError('音乐行数与押数协调目前仅验证了Music Base r1，请切回原密度条件。')
        if main_style is not None and option.get('supported_main_styles') and main_style not in option['supported_main_styles']:
            raise ValueError('该密度条件仅支持 Jumpstream 主型。')
    return value


def _density(values):
    result = np.asarray(values)
    if result.ndim != 3 or result.shape[1:] != (4, 2) or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Density must be a finite nonnegative [bars, 4, 2] array.")
    return result


def generator_capabilities(model, identity):
    """Read the selected native model; do not infer limits from a model's name."""
    allowed = getattr(model, "style_count_allowed", None)
    if allowed is not None:
        allowed = allowed.detach().cpu().numpy().astype(bool)
        if allowed.shape != (len(LABELS) + 1, 4) or not allowed.any(axis=1).all():
            raise ValueError("Generator count support does not match the style vocabulary.")
        support = {label: (np.flatnonzero(allowed[index + 1]) + 1).tolist()
                   for index, label in enumerate(LABELS)}
    else:
        support = {}
    from .tiny_data import SLOTS
    return {"version": BUDGET_VERSION, **identity,
            "architecture": model.configuration().get("architecture"),
            "accepts_density": hasattr(model, "density_projection"),
            "has_native_density_transform": callable(getattr(model, "density_condition", None)),
            "count_head_support": support,
            "candidate_rows_per_beat": len(SLOTS),
            "capacity_kind": "necessary_count_head_bounds_only",
            "attainable_density_calibration": "unavailable",
            "boundary": "Count support is not learned attainable density; learned grids, music and layout can narrow it."}


def native_density_condition(model, density, patterns):
    """Observe the same pure native transform used at the generate() boundary."""
    import torch

    values = _density(density)
    if not hasattr(model, "density_projection"):
        return None
    transform = getattr(model, "density_condition", None)
    if transform is None:
        return values.copy()
    patterns = np.asarray(patterns)
    if patterns.shape != (len(values), len(LABELS)):
        raise ValueError("Native density inspection needs aligned style patterns.")
    device = next(model.parameters()).device
    with torch.inference_mode():
        d = torch.as_tensor(values.reshape(1, -1, 2), dtype=torch.float32, device=device)
        p = torch.as_tensor(np.repeat(patterns, 4, axis=0)[None], dtype=torch.float32, device=device)
        effective = transform(d, p).cpu().numpy().reshape(values.shape)
    _density(effective)
    return effective


def prepare_density_budget(density, style_ids, quiet, capabilities, mode="legacy"):
    """Keep desired notes; bounded repair of inconsistent rows, with no hard fill.

    At most 25% (or 0.25 rows/beat for near-zero conditions) is changed per beat.
    Silence, note targets and type choices are immutable. Unresolved conflicts
    remain explicit; we never lower the musical objective to improve its metric.
    """
    validate_budget(mode)
    desired = _density(density)
    styles, quiet = np.asarray(style_ids), np.asarray(quiet, dtype=bool)
    if styles.shape != (len(desired),) or quiet.shape != styles.shape or not np.issubdtype(styles.dtype, np.integer) or ((styles < 0) | (styles > len(LABELS))).any():
        raise ValueError("Budget styles and quiet mask must align with the density timeline.")
    condition = desired.astype(np.result_type(desired.dtype, np.float32), copy=True)
    diagnostics = []
    for index, (style_id, silent) in enumerate(zip(styles, quiet)):
        style = LABELS[int(style_id) - 1] if style_id else None
        counts = capabilities["count_head_support"].get(style, [])
        original = desired[index]
        row_low = original[:, 0] / max(counts) if counts else np.zeros(4)
        slots = capabilities['candidate_rows_per_beat']
        row_high = np.minimum(original[:, 0] / min(counts), slots) if counts else np.full(4, float(slots))
        conflict = (original[:, 1] < row_low - 1e-6) | (original[:, 1] > row_high + 1e-6)
        eligible = bool(not silent and style and counts and capabilities["accepts_density"])
        if mode == "feasible" and eligible:
            wanted = np.clip(original[:, 1], row_low, row_high)
            bound = np.maximum(.25, original[:, 1] * .25)
            condition[index, :, 1] = np.clip(wanted, np.maximum(0, original[:, 1] - bound), original[:, 1] + bound)
        remaining = (condition[index, :, 1] < row_low - 1e-6) | (condition[index, :, 1] > row_high + 1e-6)
        diagnostics.append({"bar": index + 1, "style": style, "quiet": bool(silent),
            "eligible": eligible, "supported_counts": counts,
            "necessary_rows_interval": np.stack((row_low, row_high), axis=-1).tolist() if counts else None,
            "conflicting_beats": int(conflict.sum()) if counts and not silent else 0,
            "unresolved_beats": int(remaining.sum()) if counts and not silent else 0,
            "changed_beats": int(np.count_nonzero(condition[index, :, 1] != original[:, 1]))})
    return condition, {"version": BUDGET_VERSION, "mode": mode, "generator": capabilities,
        "note_target_policy": "preserve_original_music_objective",
        "onset_policy": "native_decisions_no_probability_calibration_or_hard_fill",
        "max_relative_row_change": .25, "near_zero_row_change_bound": .25,
        "changed_beats": sum(item["changed_beats"] for item in diagnostics),
        "conflicting_beats": sum(item["conflicting_beats"] for item in diagnostics),
        "unresolved_beats": sum(item["unresolved_beats"] for item in diagnostics), "bars": diagnostics}


class ExecutionLedger:
    """Keep only committed positions, never overlapping continuation prefixes."""
    def __init__(self):
        self.events = {}

    def record(self, time_ms, music_count, prepared_count, mask):
        values = (int(music_count), int(prepared_count), int(mask))
        if not (0 <= values[0] <= 4 and 0 <= values[1] <= 4 and 0 <= values[2] <= 15):
            raise ValueError("Execution ledger received invalid counts or masks.")
        stamp = int(round(time_ms))
        if stamp in self.events:
            raise ValueError("Execution positions were committed twice; check continuation ownership.")
        self.events[stamp] = values


def execution_diagnostics(desired, condition, effective, bars, beat_length_ms, ledger, exported_notes, budget):
    """Decompose signed errors and retain gross losses/additions separately.

    All stages use identical rounded half-open intervals clipped to real audio.
    A partial beat is exposed, but excluded from full-beat error summaries.
    """
    desired, condition = _density(desired), _density(condition)
    if desired.shape != condition.shape or len(bars) != len(desired) or not np.isfinite(beat_length_ms) or beat_length_ms <= 0:
        raise ValueError("Execution diagnostics require an aligned positive beat timeline.")
    if effective is not None and _density(effective).shape != desired.shape:
        raise ValueError("Effective native conditions do not match the timeline.")
    exported = {}
    for stamp, column in exported_notes:
        stamp, column = int(round(stamp)), int(column)
        if column < 0 or column >= 4 or exported.get(stamp, 0) & (1 << column):
            raise ValueError("Exported notes contain an invalid or duplicate lane event.")
        exported[stamp] = exported.get(stamp, 0) | (1 << column)
    timeline = sorted(set(ledger.events) | set(exported))
    summaries = {key: 0 for key in ("music_notes", "prepared_notes", "layout_notes", "export_notes",
        "music_rows", "prepared_rows", "layout_rows", "export_rows",
        "prepare_removed_notes", "prepare_added_notes", "layout_removed_notes", "layout_added_notes",
        "export_removed_notes", "export_added_notes", "prepare_dropped_rows", "layout_dropped_rows", "export_dropped_rows")}
    trace, full_errors, full_mask, included = [], [], [], set()
    for index, bar in enumerate(bars):
        edges = np.rint(float(bar["start_ms"]) + np.arange(5) * beat_length_ms).astype(np.int64)
        beat_records = []
        for beat in range(4):
            start, stop = max(0, int(edges[beat])), min(int(edges[beat + 1]), int(round(bar["end_ms"])))
            full = bool(start == edges[beat] and stop == edges[beat + 1] and stop > start)
            stamps = timeline[bisect_left(timeline, start):bisect_left(timeline, max(start, stop))]
            included.update(stamps)
            counts = np.array([[*ledger.events.get(t, (0, 0, 0))[:2],
                                ledger.events.get(t, (0, 0, 0))[2].bit_count(), exported.get(t, 0).bit_count()]
                               for t in stamps], dtype=np.int64).reshape(-1, 4)
            notes, rows = counts.sum(axis=0), (counts > 0).sum(axis=0)
            stages = np.stack((notes, rows), axis=-1)
            losses = {}
            for position, name in enumerate(("prepare", "layout", "export")):
                delta = counts[:, position + 1] - counts[:, position]
                losses[name + "_removed_notes"] = int(-np.minimum(delta, 0).sum())
                losses[name + "_added_notes"] = int(np.maximum(delta, 0).sum())
                losses[name + "_dropped_rows"] = int(((counts[:, position] > 0) & (counts[:, position + 1] == 0)).sum())
            for name, values in zip(("music", "prepared", "layout", "export"), stages):
                summaries[name + "_notes"] += int(values[0])
                summaries[name + "_rows"] += int(values[1])
            for key, value in losses.items():
                summaries[key] += value
            # Exact telescoping attribution, not a causal claim about neural decisions.
            deltas = {"planner_to_music": (stages[0] - desired[index, beat]).tolist(),
                      "music_to_prepared": (stages[1] - stages[0]).tolist(),
                      "prepared_to_layout": (stages[2] - stages[1]).tolist(),
                      "layout_to_export": (stages[3] - stages[2]).tolist()}
            if effective is not None:
                del deltas['planner_to_music']
                deltas.update({
                    'planner_to_condition': (condition[index, beat] - desired[index, beat]).tolist(),
                    'condition_to_native': (effective[index, beat] - condition[index, beat]).tolist(),
                    'native_to_music': (stages[0] - effective[index, beat]).tolist()})
            if full:
                full_errors.append(np.abs(stages[3] - desired[index, beat]))
            full_mask.append(full)
            beat_records.append({"beat": beat + 1, "start_ms": start, "end_ms": max(start, stop),
                "complete": full, "desired": desired[index, beat].tolist(),
                "generator_condition": condition[index, beat].tolist(),
                "native_effective_condition": effective[index, beat].tolist() if effective is not None else None,
                "stage_counts": {name: values.tolist() for name, values in zip(("music", "prepared", "layout", "export"), stages)},
                "music_chord_histogram": np.bincount(counts[:, 0], minlength=5)[1:5].tolist(),
                "export_chord_histogram": np.bincount(counts[:, 3], minlength=5)[1:5].tolist(),
                "signed_deltas": deltas, "losses": losses})
        trace.append({"bar": index + 1, "style": bar.get("style"), "quiet": bool(bar["quiet"]), "beats": beat_records})
    summaries["full_beats"] = sum(full_mask)
    summaries["notes_beat_mae"] = float(np.mean(full_errors, axis=0)[0]) if full_errors else None
    summaries["rows_beat_mae"] = float(np.mean(full_errors, axis=0)[1]) if full_errors else None
    summaries["unattributed_positions"] = len(set(timeline) - included)
    summaries['unattributed_music_notes'] = sum(ledger.events.get(t, (0, 0, 0))[0] for t in set(timeline) - included)
    summaries['unattributed_export_notes'] = sum(exported.get(t, 0).bit_count() for t in set(timeline) - included)
    return {"version": DIAGNOSTIC_VERSION, "budget": budget, "summary": summaries, "bars": trace,
            "units": "[notes, occupied rows] per beat; chord histograms index sizes 1..4",
            "edge_policy": "rounded half-open intervals clipped to audio; partial beats excluded from MAE",
            "boundary": "Observed stage differences, not causal proof or musical quality; no attainable-density calibration."}
