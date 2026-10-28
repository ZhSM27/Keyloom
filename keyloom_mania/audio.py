"""Audio timeline and relative asset validation."""
import math
from pathlib import Path
MAX_DURATION_SECONDS = 3600


class TimestampTracker:
    def __init__(self, codec: str):
        self.codec = codec
        self.sample_count = 0
        self.origin_seconds = None
        self.previous_end_seconds = None
        self.previous_drift_ms = 0.0
        self.max_gap_ms = 0.0
        self.max_drift_ms = 0.0
        self.vorbis_jitter_frames = 0
        self.checked_frames = 0

    def observe(self, timestamp_seconds: float | None, samples: int, sample_rate: int) -> None:
        if timestamp_seconds is not None:
            if self.origin_seconds is None:
                self.origin_seconds = timestamp_seconds - self.sample_count / sample_rate
            drift_ms = (timestamp_seconds - self.origin_seconds - self.sample_count / sample_rate) * 1000
            self.max_drift_ms = max(self.max_drift_ms, abs(drift_ms))
            if self.previous_end_seconds is not None:
                self.max_gap_ms = max(self.max_gap_ms, abs(timestamp_seconds - self.previous_end_seconds) * 1000)
            if abs(drift_ms) > 5:
                if self.codec != "vorbis" or abs(drift_ms) * sample_rate / 1000 > 512.01 or abs(self.previous_drift_ms) > 5:
                    raise ValueError("Audio presentation timestamps diverge from the decoded sample timeline.")
                self.vorbis_jitter_frames += 1
            self.previous_drift_ms = drift_ms
            self.previous_end_seconds = timestamp_seconds + samples / sample_rate
            self.checked_frames += 1
        self.sample_count += samples

    def finish(self) -> None:
        if abs(self.previous_drift_ms) > 5:
            raise ValueError("Audio ends with an unresolved timestamp discontinuity.")

def asset_path(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ":" in relative_path:
        raise ValueError("Asset path must be relative to the dataset root.")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Asset path escapes the dataset root.")
    return resolved
