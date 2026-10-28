from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .audio import asset_path
from .chart_validation import chart_statistics


SCHEMA_VERSION = "beat-events-v1"
GRID_POLICIES = {
    "common": (1, 2, 3, 4, 6, 8, 12, 16),
    "extended": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 24, 32),
    "dense_diagnostic": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 24, 32, 48, 64),
}
GRID_TOLERANCE_MS = 1.0
MICRO_TOLERANCE_MS = 2.0


@dataclass(frozen=True)
class TimingSection:
    offset_ms: float
    beat_length_ms: float
    meter: int


@dataclass(frozen=True)
class Note:
    time_ms: int
    column: int
    end_ms: int | None = None


class BeatClock:
    def __init__(self, sections: list[TimingSection]):
        if not sections:
            raise ValueError("A beat clock needs at least one uninherited timing point.")
        for section in sections:
            if not math.isfinite(section.offset_ms) or not math.isfinite(section.beat_length_ms):
                raise ValueError("Timing values must be finite.")
            if section.beat_length_ms <= 0 or section.meter < 1:
                raise ValueError("Beat length and meter must be positive.")
        by_offset = {section.offset_ms: section for section in sections}
        self.sections = tuple(by_offset[offset] for offset in sorted(by_offset))
        self.offsets = tuple(section.offset_ms for section in self.sections)
        self.lengths = np.array([section.beat_length_ms for section in self.sections])

    def section_at(self, time_ms: float) -> int:
        if not math.isfinite(time_ms):
            raise ValueError("Time must be finite.")
        return max(0, bisect.bisect_right(self.offsets, time_ms) - 1)

    def grid_time(self, section_index: int, beat_index: int, numerator: int, denominator: int) -> float:
        if any(not isinstance(value, int) for value in (section_index, beat_index, numerator, denominator)):
            raise ValueError("Grid coordinates must be integers.")
        if not 0 <= section_index < len(self.sections):
            raise ValueError("Unknown timing section.")
        if denominator <= 0 or not 0 <= numerator < denominator:
            raise ValueError("Beat fraction must be between zero inclusive and one exclusive.")
        section = self.sections[section_index]
        time_ms = section.offset_ms + (beat_index + numerator / denominator) * section.beat_length_ms
        if self.section_at(time_ms) != section_index:
            raise ValueError("Grid position crosses a timing section boundary.")
        return time_ms

    def locate(self, times_ms, denominators=GRID_POLICIES["common"]) -> dict[str, np.ndarray]:
        times = np.asarray(times_ms, dtype=np.float64)
        denominators = tuple(sorted(set(denominators)))
        if times.ndim != 1 or not np.isfinite(times).all():
            raise ValueError("Times must be a finite one-dimensional sequence.")
        if not denominators or any(not isinstance(value, int) or value < 1 for value in denominators):
            raise ValueError("Grid denominators must be positive integers.")
        sections = np.maximum(np.searchsorted(self.offsets, times, side="right") - 1, 0)
        offsets = np.asarray(self.offsets)[sections]
        lengths = self.lengths[sections]
        phases = (times - offsets) / lengths
        upper_offsets = np.append(np.asarray(self.offsets)[1:], np.inf)[sections]
        lower_offsets = np.where(sections == 0, -np.inf, offsets)
        errors = []
        steps = []
        for denominator in denominators:
            lower_step = np.floor(phases * denominator).astype(np.int64)
            candidates = np.stack([lower_step, lower_step + 1])
            candidate_times = offsets + candidates * lengths / denominator
            valid = (candidate_times >= lower_offsets) & (candidate_times < upper_offsets)
            distances = np.where(valid, np.abs(candidate_times - times), np.inf)
            chosen = distances.argmin(axis=0)
            indices = np.arange(times.size)
            steps.append(candidates[chosen, indices])
            errors.append(distances[chosen, indices])
        errors = np.stack(errors)
        steps = np.stack(steps)
        nearest = errors.argmin(axis=0)
        within_grid = errors <= GRID_TOLERANCE_MS + 1e-7
        within_micro = errors <= MICRO_TOLERANCE_MS + 1e-7
        selected = np.where(
            within_grid.any(axis=0), within_grid.argmax(axis=0),
            np.where(within_micro.any(axis=0), within_micro.argmax(axis=0), nearest),
        )
        indices = np.arange(times.size)
        selected_denominators = np.asarray(denominators)[selected]
        selected_steps = steps[selected, indices]
        grid_times = offsets + selected_steps * lengths / selected_denominators
        residuals = times - grid_times
        modes = np.where(within_grid.any(axis=0), "grid", np.where(within_micro.any(axis=0), "micro", "free"))
        return {
            "section": sections,
            "step": selected_steps,
            "denominator": selected_denominators,
            "grid_time_ms": grid_times,
            "residual_ms": residuals,
            "nearest_error_ms": errors.min(axis=0),
            "mode": modes,
        }

    def render_grid_position(self, section_index: int, beat_index: int, numerator: int, denominator: int, policy: str = "common", micro_offset_ms: int | None = None) -> int:
        grid_time = self.grid_time(section_index, beat_index, numerator, denominator)
        if Fraction(numerator, denominator).denominator not in GRID_POLICIES[policy]:
            raise ValueError("Fraction is outside the selected rhythm vocabulary.")
        offset = 0 if micro_offset_ms is None else micro_offset_ms
        if not isinstance(offset, int) or abs(offset) > MICRO_TOLERANCE_MS:
            raise ValueError("A micro offset must be an explicitly supplied integer within 2 ms.")
        rendered = math.floor(grid_time + offset + 0.5)
        if micro_offset_ms is not None and self.section_at(rendered) != section_index:
            raise ValueError("Micro offset crosses a timing section boundary.")
        return rendered

    def encode_time(self, time_ms: int, denominators=GRID_POLICIES["common"]) -> dict:
        return self.encode_times([time_ms], denominators)[0]

    def encode_times(self, times_ms: list[int], denominators=GRID_POLICIES["common"]) -> list[dict]:
        if any(not isinstance(time_ms, int) for time_ms in times_ms):
            raise ValueError("Source note times must be integer milliseconds.")
        located = self.locate(times_ms, denominators)
        positions = []
        for index, time_ms in enumerate(times_ms):
            section_index = int(located["section"][index])
            step = int(located["step"][index])
            denominator = int(located["denominator"][index])
            beat_index, numerator = divmod(step, denominator)
            fraction = Fraction(numerator, denominator)
            positions.append({
                "time_ms": time_ms,
                "mode": str(located["mode"][index]),
                "section_index": section_index,
                "beat_index": beat_index,
                "numerator": fraction.numerator,
                "denominator": fraction.denominator,
                "residual_ms": float(located["residual_ms"][index]),
            })
        return positions

    def decode_time(self, position: dict, preserve_residual: bool = True) -> int:
        if position["mode"] not in {"grid", "micro", "free"}:
            raise ValueError("Unknown time mode.")
        if position["mode"] == "free":
            if not preserve_residual:
                raise ValueError("Free events require explicit free-time handling; they cannot be silently snapped.")
            if not isinstance(position["time_ms"], int):
                raise ValueError("Free time must be an integer millisecond timestamp.")
            return position["time_ms"]
        if position["mode"] == "micro" and not preserve_residual:
            raise ValueError("Micro-offset events require an explicit offset decision.")
        grid_time = self.grid_time(position["section_index"], position["beat_index"], position["numerator"], position["denominator"])
        residual = position["residual_ms"] if preserve_residual else 0.0
        if not math.isfinite(residual):
            raise ValueError("Time residual must be finite.")
        return math.floor(grid_time + residual + 0.5)


def read_chart(path: Path) -> tuple[BeatClock, list[Note]]:
    text = path.read_text(encoding="utf-8-sig", errors="strict")
    statistics = chart_statistics(text)
    if statistics["validation_errors"]:
        raise ValueError("Invalid source chart: " + "; ".join(statistics["validation_errors"][:3]))
    section_name = ""
    metadata = {}
    sections = []
    notes = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("["):
            section_name = line
        elif section_name in {"[General]", "[Difficulty]"} and ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        elif section_name == "[TimingPoints]":
            fields = line.split(",")
            if len(fields) < 7 or fields[6].strip() == "1":
                sections.append(TimingSection(float(fields[0]), float(fields[1]), int(fields[2]) if len(fields) > 2 else 4))
        elif section_name == "[HitObjects]":
            fields = line.split(",")
            time_ms = float(fields[2])
            end_ms = float(fields[5].split(":")[0]) if int(fields[3]) & 128 else None
            if not time_ms.is_integer() or (end_ms is not None and not end_ms.is_integer()):
                raise ValueError("Fractional source milliseconds need a separate schema; no rounding is allowed here.")
            notes.append(Note(int(time_ms), min(3, int(float(fields[0]) * 4 / 512)), int(end_ms) if end_ms is not None else None))
    if metadata.get("Mode") != "3" or float(metadata.get("CircleSize", "0")) != 4:
        raise ValueError("Only native osu!mania 4K charts are supported.")
    if not notes:
        raise ValueError("A source chart must contain notes.")
    return BeatClock(sections), sorted(notes, key=lambda note: (note.time_ms, note.column))


def encode_rows(clock: BeatClock, notes: list[Note], policy: str = "common") -> list[dict]:
    rows = collections.defaultdict(list)
    for note in notes:
        rows[note.time_ms].append(note)
    sorted_rows = sorted(rows.items())
    head_positions = clock.encode_times([time_ms for time_ms, _ in sorted_rows], GRID_POLICIES[policy])
    end_positions = iter(clock.encode_times([note.end_ms for _, row_notes in sorted_rows for note in row_notes if note.end_ms is not None], GRID_POLICIES[policy]))
    result = []
    for (time_ms, row_notes), position in zip(sorted_rows, head_positions):
        if any(not 0 <= note.column <= 3 for note in row_notes):
            raise ValueError("Note column must be between 0 and 3.")
        if len({note.column for note in row_notes}) != len(row_notes):
            raise ValueError("Duplicate column at the same onset time.")
        result.append({
            "position": position,
            "tap_mask": sum(1 << note.column for note in row_notes if note.end_ms is None),
            "hold_mask": sum(1 << note.column for note in row_notes if note.end_ms is not None),
            "hold_ends": [
                {"column": note.column, "position": next(end_positions)}
                for note in row_notes if note.end_ms is not None
            ],
        })
    return result


def decode_rows(clock: BeatClock, rows: list[dict]) -> list[Note]:
    notes = []
    previous_time = -math.inf
    holding_until = [-math.inf] * 4
    for row in rows:
        time_ms = clock.decode_time(row["position"])
        if time_ms <= previous_time:
            raise ValueError("Onset rows must be strictly increasing.")
        previous_time = time_ms
        taps, holds = row["tap_mask"], row["hold_mask"]
        if not isinstance(taps, int) or not isinstance(holds, int) or not 0 <= taps <= 15 or not 0 <= holds <= 15 or taps & holds or not taps | holds:
            raise ValueError("Invalid 4K row masks.")
        ends = {item["column"]: item["position"] for item in row["hold_ends"]}
        if len(ends) != len(row["hold_ends"]) or set(ends) != {column for column in range(4) if holds & (1 << column)}:
            raise ValueError("Hold ends must match hold heads exactly.")
        for column in range(4):
            if not (taps | holds) & (1 << column):
                continue
            if holding_until[column] > time_ms:
                raise ValueError("A new head overlaps a held column.")
            end_ms = clock.decode_time(ends[column]) if holds & (1 << column) else None
            if end_ms is not None and end_ms <= time_ms:
                raise ValueError("A hold must end after its head.")
            holding_until[column] = end_ms if end_ms is not None else time_ms
            notes.append(Note(time_ms, column, end_ms))
    return notes


def make_window(clock: BeatClock, notes: list[Note], start_ms: int, end_ms: int, context_rows: int = 32) -> dict:
    if not isinstance(start_ms, int) or not isinstance(end_ms, int) or end_ms <= start_ms or context_rows < 0:
        raise ValueError("Window bounds must be increasing integer milliseconds; context size must be nonnegative.")
    encoded = encode_rows(clock, notes)
    previous = [row for row in encoded if row["position"]["time_ms"] < start_ms]
    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "history_rows": previous[-context_rows:] if context_rows else [],
        "target_rows": [row for row in encoded if start_ms <= row["position"]["time_ms"] < end_ms],
        "carry_holds": [
            {"column": note.column, "end_position": clock.encode_time(note.end_ms)}
            for note in notes if note.end_ms is not None and note.time_ms < start_ms < note.end_ms
        ],
    }
