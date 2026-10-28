"""Portable candidate-grid constants shared by native model layers."""
import math
import numpy as np
from .timing import GRID_POLICIES
TICKS_PER_BEAT = 48
SLOTS = tuple(sorted({numerator*TICKS_PER_BEAT//denominator for denominator in GRID_POLICIES["common"] for numerator in range(denominator)}))
STYLE_KEYS = ("Chordjacks", "Minijacks", "Jumpstream", "Handstream")
BEATS_PER_WINDOW = 8
SEQUENCE_LENGTH = BEATS_PER_WINDOW * len(SLOTS)


def window_grid(clock, start_beat: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    section = clock.sections[0]
    ticks = np.array([(start_beat + beat) * TICKS_PER_BEAT + slot for beat in range(BEATS_PER_WINDOW) for slot in SLOTS], dtype=np.int64)
    times = section.offset_ms + ticks * section.beat_length_ms / TICKS_PER_BEAT
    phase = (ticks % TICKS_PER_BEAT) / TICKS_PER_BEAT
    bar_phase = (ticks % (TICKS_PER_BEAT * 4)) / (TICKS_PER_BEAT * 4)
    position = np.stack((np.sin(phase * 2 * np.pi), np.cos(phase * 2 * np.pi), np.sin(bar_phase * 2 * np.pi), np.cos(bar_phase * 2 * np.pi), np.full(len(ticks), section.beat_length_ms / 1000)), axis=-1).astype(np.float32)
    return ticks, times, position


def interpolate_features(features: np.ndarray, times_ms: np.ndarray, hop_ms: float) -> np.ndarray:
    if features.ndim != 2 or not len(features) or not np.isfinite(features).all() or not math.isfinite(hop_ms) or hop_ms <= 0:
        raise ValueError("Invalid feature interpolation source.")
    if times_ms.ndim != 1 or not np.isfinite(times_ms).all() or np.any(times_ms < 0) or np.any(times_ms > (len(features) - 1) * hop_ms):
        raise ValueError("Audio conditioning positions must be inside the cached frame anchors.")
    coordinate = times_ms / hop_ms
    lower = np.floor(coordinate).astype(np.int64)
    upper = np.minimum(lower + 1, len(features) - 1)
    weight = (coordinate - lower).astype(np.float32)[:, None]
    return features[lower].astype(np.float32) * (1 - weight) + features[upper].astype(np.float32) * weight
