"""Inference grid adapter; dataset preparation is outside the runtime package."""
import numpy as np
from .tiny_data import window_grid

def window_grid_from_section(section: dict, start_beat: int) -> tuple[np.ndarray, np.ndarray]:
    from .timing import BeatClock, TimingSection

    _, times, phase = window_grid(BeatClock([TimingSection(**section)]), start_beat)
    return times, phase
