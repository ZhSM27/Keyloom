from dataclasses import asdict, dataclass

import numpy as np

from .patterns import _merge_intervals
from .segment_labels import LABELS


STYLE_NAMES = ("Unspecified", *LABELS)
SPECIFIC_CHILDREN = {
    "Stream": ("Rolls", "Trills", "Minitrills"),
    "Chordstream": ("Jumpstream", "Handstream", "Quadstream"),
    "Jacks": ("Chordjacks", "Longjacks", "Minijacks", "Gluts"),
}
ALLOWED_OVERLAP = {
    "Stream": ("Rolls", "Trills", "Minitrills"),
    "Chordstream": ("Jumpstream", "Handstream", "Quadstream"),
    "Jacks": ("Chordjacks", "Longjacks", "Minijacks", "Gluts"),
    "Chordjacks": ("Jacks", "Longjacks", "Gluts"),
    "Longjacks": ("Jacks", "Chordjacks", "Gluts"),
    "Minijacks": ("Jacks",),
    "Gluts": ("Jacks", "Chordjacks", "Longjacks"),
    "Jumpstream": ("Chordstream",),
    "Handstream": ("Chordstream",),
    "Quadstream": ("Chordstream",),
    "Rolls": ("Stream",),
    "Trills": ("Stream", "Minitrills"),
    "Minitrills": ("Stream", "Trills"),
}


@dataclass(frozen=True)
class MainStylePolicy:
    minimum_reference_coverage: float = 0.25
    minimum_prediction_confidence: float = 0.30
    smoothing_bars: int = 3
    specific_relative_strength: float = 0.5
    allow_longjacks: bool = False
    quiet_rows_per_beat: float = 0.05
    quiet_notes_per_beat: float = 0.10


def main_targets(evidence, available=None, policy=MainStylePolicy()):
    values = np.asarray(evidence, dtype=np.float32).copy()
    if values.ndim < 2 or values.shape[-1] != len(LABELS) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Main style evidence must end in fifteen finite overlapping coverages.")
    if not policy.allow_longjacks:
        source = values[..., LABELS.index("Longjacks")].copy()
        values[..., LABELS.index("Jacks")] = np.maximum(values[..., LABELS.index("Jacks")], source)
        values[..., LABELS.index("Longjacks")] = 0
    for parent, children in SPECIFIC_CHILDREN.items():
        children = [child for child in children if policy.allow_longjacks or child != "Longjacks"]
        strength = values[..., [LABELS.index(child) for child in children]].max(axis=-1)
        parent_values = values[..., LABELS.index(parent)]
        specific = (strength >= policy.minimum_reference_coverage) & (strength >= parent_values * policy.specific_relative_strength)
        values[..., LABELS.index(parent)] = np.where(specific, 0, parent_values)
    strongest = values.max(axis=-1)
    result = values.argmax(axis=-1).astype(np.int64) + 1
    result[strongest < policy.minimum_reference_coverage] = 0
    if available is not None:
        available = np.asarray(available, dtype=bool)
        if available.shape != result.shape:
            raise ValueError("Reference availability must align with style evidence.")
        result[~available] = 0
    return result


def one_hot_plan(style_ids):
    ids = np.asarray(style_ids)
    if not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= len(STYLE_NAMES))):
        raise ValueError("Main style IDs are outside the explicit vocabulary.")
    return np.eye(len(STYLE_NAMES), dtype=np.float32)[ids][..., 1:]


def compile_main_plan(probabilities, notes, rows, sections, policy=MainStylePolicy()):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    notes, rows = np.asarray(notes, dtype=np.float32), np.asarray(rows, dtype=np.float32)
    sections = np.asarray(sections)
    if probabilities.shape != (len(notes), len(STYLE_NAMES)) or rows.shape != notes.shape or sections.shape != notes.shape:
        raise ValueError("Main Planner outputs must align with their bars and sections.")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Main Planner outputs must be categorical probabilities.")
    if not np.isfinite(notes).all() or not np.isfinite(rows).all():
        raise ValueError("Density predictions must be finite.")
    notes, rows = notes.clip(0, 96), rows.clip(0, 24)
    quiet = (notes <= policy.quiet_notes_per_beat) & (rows <= policy.quiet_rows_per_beat)
    smooth = probabilities.copy()
    radius = policy.smoothing_bars // 2
    if policy.smoothing_bars < 1 or policy.smoothing_bars % 2 != 1:
        raise ValueError("Main style smoothing needs a positive odd bar count.")
    for position in range(len(notes)):
        lower, upper = position, position + 1
        while lower > max(0, position - radius) and sections[lower - 1] == sections[position] and not quiet[lower - 1] and not quiet[position]:
            lower -= 1
        while upper < min(len(notes), position + radius + 1) and sections[upper] == sections[position] and not quiet[upper] and not quiet[position]:
            upper += 1
        smooth[position] = probabilities[lower:upper].mean(axis=0)
    if not policy.allow_longjacks:
        smooth[:, STYLE_NAMES.index("Jacks")] += smooth[:, STYLE_NAMES.index("Longjacks")]
        smooth[:, STYLE_NAMES.index("Longjacks")] = 0
    ids = smooth.argmax(axis=1).astype(np.int64)
    confidence = smooth.max(axis=1)
    ids[quiet | (confidence < policy.minimum_prediction_confidence)] = 0
    return {
        "style_ids": ids, "patterns": one_hot_plan(ids).astype(np.uint8),
        "confidence": confidence, "target_row_budget": np.where(quiet, 0, rows).astype(np.float32),
        "target_note_budget": np.where(quiet, 0, notes).astype(np.float32), "quiet": quiet.astype(np.uint8),
        "policy": asdict(policy),
    }


def evaluate_bars(analysis, starts, ends, style_ids, quiet=None, scored=None):
    starts, ends, ids = np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64), np.asarray(style_ids)
    if starts.shape != ends.shape or ids.shape != starts.shape or starts.ndim != 1 or not np.isfinite(starts).all() or not np.isfinite(ends).all() or np.any(ends <= starts):
        raise ValueError("Bar evaluation needs aligned, finite positive spans.")
    one_hot_plan(ids)
    quiet = np.zeros(len(ids), dtype=bool) if quiet is None else np.asarray(quiet, dtype=bool)
    scored = np.ones(len(ids), dtype=bool) if scored is None else np.asarray(scored, dtype=bool)
    if quiet.shape != ids.shape or scored.shape != ids.shape:
        raise ValueError("Bar masks must align with planned bars.")
    coverages = {label: _merge_intervals([tuple(interval) for interval in value["intervals_ms"]]) for label, value in analysis["coverage"].items()}
    records = []
    for index, (start, end, style_id) in enumerate(zip(starts, ends, ids)):
        label = STYLE_NAMES[int(style_id)]
        eligible = bool(scored[index] and not quiet[index] and style_id > 0)
        fractions = {name: sum(max(0, min(end, upper) - max(start, lower)) for lower, upper in intervals) / (end - start) for name, intervals in coverages.items()}
        target = fractions.get(label, 0.0)
        allowed = set(ALLOWED_OVERLAP.get(label, ())) | {label}
        other = {name: fraction for name, fraction in fractions.items() if name not in allowed}
        maximum = max(other, key=other.get) if other else None
        other_fraction = other.get(maximum, 0.0)
        records.append({"bar": index, "start_ms": float(start), "end_ms": float(end), "planned_style": label, "eligible": eligible,
                        "target_coverage": float(target), "largest_other_label": maximum, "largest_other_coverage": float(other_fraction),
                        "pass": bool(target >= 0.5 and other_fraction < 0.5) if eligible else None,
                        "clean_pass": bool(target >= 0.5 and other_fraction < 0.35) if eligible else None})
    selected = [record for record in records if record["eligible"]]
    per_style = {}
    for label in sorted({record["planned_style"] for record in selected}):
        group = [record for record in selected if record["planned_style"] == label]
        per_style[label] = {"bars": len(group), "pass_rate": float(np.mean([record["pass"] for record in group])), "mean_target_coverage": float(np.mean([record["target_coverage"] for record in group]))}
    return {"bars": records, "eligible_bars": len(selected), "passed_bars": sum(record["pass"] for record in selected),
            "bar_pass_rate": float(np.mean([record["pass"] for record in selected])) if selected else None,
            "per_style": per_style, "policy": "complete-chart detector context; interval coverage clipped to each actually planned bar; target>=0.5 and other<0.5; unspecified and ungenerated bars excluded"}
