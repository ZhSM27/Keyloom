"""Validate the handoff between HTTP requests, planning and note generation."""
import math
import numpy as np
from ..segment_labels import LABELS

PLAN_SCHEMA = "studio-plan-v1"


def option_id(component, value):
    if not isinstance(value, str):
        raise ValueError(f"{component} model ID must be a string.")
    return value.removeprefix(component + "/")


def validate_settings(settings):
    for key, lower, upper in (("bpm", 30, 400), ("first_beat_ms", -3000, 60000), ("target_star", 2, 9)):
        value = float(settings[key])
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{key} must be finite and between {lower} and {upper}.")


def validate_plan(plan, beat_density, bar_count):
    """Reject malformed plans before decoding; never repair musical budgets here."""
    patterns = np.asarray(plan["patterns"])
    density = np.asarray(beat_density)
    if patterns.shape != (bar_count, len(LABELS)) or density.shape != (bar_count, 4, 2):
        raise ValueError("Planner output does not match the bar/style/density contract.")
    if not np.isfinite(patterns).all() or not np.isfinite(density).all() or (density < 0).any():
        raise ValueError("Planner output contains invalid values.")
    if ((patterns != 0) & (patterns != 1)).any():
        raise ValueError("The compiled style plan must contain binary activations.")
    for key in ("target_note_budget", "target_row_budget", "quiet", "segment_ids"):
        values = np.asarray(plan[key])
        if values.shape != (bar_count,) or not np.isfinite(values).all():
            raise ValueError(f"Planner output {key} does not match its bar timeline.")
    if 'hierarchy' in plan:
        from ..hierarchical_planner import validate_hierarchy
        validate_hierarchy(plan['hierarchy'], plan['quiet'])
        if any(not np.array_equal(plan[key], plan['hierarchy'][key]) for key in ('phrase_ids', 'section_ids', 'segment_ids')):
            raise ValueError('Compiled hierarchy IDs do not match their metadata.')


def named_options(component, options):
    return [{**item, "model_key": f"{component}/{item['id']}", "component": component} for item in options]
