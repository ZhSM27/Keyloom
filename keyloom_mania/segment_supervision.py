"""Binary per-beat style plan representation."""
import numpy as np
from .segment_labels import LABELS
from .tiny_data import BEATS_PER_WINDOW, SEQUENCE_LENGTH
PLAN_WIDTH = len(LABELS)
PLAN_MODES = ("reference", "absent", "shifted")


def positive_beat_plan(coverage: np.ndarray) -> np.ndarray:
    if coverage.ndim != 3 or coverage.shape[1:] != (SEQUENCE_LENGTH, PLAN_WIDTH) or not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
        raise ValueError("Beat plans require finite, complete, aligned segment coverage.")
    slots = SEQUENCE_LENGTH // BEATS_PER_WINDOW
    grouped = coverage.reshape(len(coverage), BEATS_PER_WINDOW, slots, PLAN_WIDTH)
    return np.any(grouped > 0, axis=2).astype(np.float32)

def expand_plan(plans: np.ndarray, mode: str) -> np.ndarray:
    if mode not in PLAN_MODES or plans.ndim != 3 or plans.shape[1:] != (BEATS_PER_WINDOW, PLAN_WIDTH) or not np.isin(plans, (0, 1)).all():
        raise ValueError("Invalid binary positive beat plan or plan mode.")
    if mode == "absent":
        return np.zeros((len(plans), SEQUENCE_LENGTH, PLAN_WIDTH), dtype=np.float32)
    selected = np.roll(plans, BEATS_PER_WINDOW // 2, axis=1) if mode == "shifted" else plans
    return np.repeat(selected, SEQUENCE_LENGTH // BEATS_PER_WINDOW, axis=1).astype(np.float32)
