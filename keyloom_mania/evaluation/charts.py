"""Load exported packages, single charts and human reference charts for evaluation.

Unlike the training-era ``timing.read_chart`` this reader tolerates fractional
milliseconds and per-object rounding in community charts: hit objects closer
than ``ROW_MERGE_MS`` share one row. Long-note heads count as taps; their share
is reported because the released generators only produce taps.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..timing import BeatClock, Note, TimingSection

ROW_MERGE_MS = 2.0
AUDIO_SUFFIXES = (".mp3", ".ogg", ".wav", ".flac", ".m4a")
# Charts written by this project (or its research runs) never enter a human reference.
GENERATED_MARKERS = ("Keyloom", "neo-ChartComposer", "Planner + compiled plan")
HEADER_PATTERN = re.compile(rb"osu file format v\d+")


@dataclass
class ChartInput:
    name: str
    chart_bytes: bytes
    chart_path: Path
    metadata: dict
    clock: BeatClock
    notes: list
    issues: list = field(default_factory=list)
    plan_bars: list | None = None
    execution: dict | None = None
    target_star: float | None = None
    audio_path: Path | None = None
    _temporary: tempfile.TemporaryDirectory | None = None

    @property
    def sha256(self):
        return hashlib.sha256(self.chart_bytes).hexdigest()

    def close(self):
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def parse_chart_text(text):
    """Return metadata, uninherited timing sections, notes and tolerated issues."""
    section, metadata, sections, notes, issues = "", {}, [], [], []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("﻿")
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section in ("General", "Metadata", "Difficulty") and ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        elif section == "TimingPoints":
            fields = line.split(",")
            try:
                offset, beat_length = float(fields[0]), float(fields[1])
                meter = int(fields[2]) if len(fields) > 2 and fields[2].strip() else 4
            except (ValueError, IndexError):
                issues.append("malformed_timing_point")
                continue
            if len(fields) >= 7 and fields[6].strip() != "1":
                continue
            if not math.isfinite(offset) or not math.isfinite(beat_length) or beat_length <= 0:
                issues.append("invalid_timing_point")
                continue
            sections.append(TimingSection(offset, beat_length, meter if meter >= 1 else 4))
        elif section == "HitObjects":
            fields = line.split(",")
            try:
                position, time_ms, kind = float(fields[0]), float(fields[2]), int(fields[3])
                end_ms = float(fields[5].split(":")[0]) if kind & 128 else None
            except (ValueError, IndexError):
                issues.append("malformed_hit_object")
                continue
            if not kind & (1 | 128) or not math.isfinite(time_ms) or not 0 <= position <= 512:
                issues.append("unsupported_hit_object")
                continue
            column = min(3, max(0, int(position * 4 / 512)))
            notes.append(Note(int(round(time_ms)), column, int(round(end_ms)) if end_ms is not None and math.isfinite(end_ms) else None))
    notes.sort(key=lambda note: (note.time_ms, note.column))
    unique = []
    for note in notes:
        if unique and unique[-1].time_ms == note.time_ms and unique[-1].column == note.column:
            issues.append("duplicate_note")
            continue
        unique.append(note)
    return metadata, sections, unique, issues


def is_mania_4k(metadata):
    try:
        return metadata.get("Mode", "0").strip() == "3" and float(metadata.get("CircleSize", "0")) == 4
    except ValueError:
        return False


def is_generated(metadata):
    text = " ".join(metadata.get(key, "") for key in ("Creator", "Version", "Tags", "Source"))
    return any(marker in text for marker in GENERATED_MARKERS)


def chart_rows(notes):
    """Merge per-object rounding into rows: times (ms), masks and hold-head counts."""
    times, masks, holds = [], [], []
    for note in notes:
        if times and note.time_ms - times[-1] <= ROW_MERGE_MS and not masks[-1] & (1 << note.column):
            masks[-1] |= 1 << note.column
            holds[-1] += note.end_ms is not None
            continue
        times.append(float(note.time_ms))
        masks.append(1 << note.column)
        holds.append(int(note.end_ms is not None))
    return np.asarray(times, dtype=np.float64), np.asarray(masks, dtype=np.int64), np.asarray(holds, dtype=np.int64)


def absolute_beats(clock, times_ms):
    """Continuous beat coordinate across timing sections (same convention as lane_control)."""
    times = np.asarray(times_ms, dtype=np.float64)
    offsets = np.asarray(clock.offsets, dtype=np.float64)
    lengths = np.asarray(clock.lengths, dtype=np.float64)
    section_beats = np.concatenate(([0.0], np.cumsum(np.diff(offsets) / lengths[:-1])))
    index = np.maximum(np.searchsorted(offsets, times, side="right") - 1, 0)
    return section_beats[index] + (times - offsets[index]) / lengths[index]


def beats_to_ms(clock, beats):
    beats = np.asarray(beats, dtype=np.float64)
    offsets = np.asarray(clock.offsets, dtype=np.float64)
    lengths = np.asarray(clock.lengths, dtype=np.float64)
    section_beats = np.concatenate(([0.0], np.cumsum(np.diff(offsets) / lengths[:-1])))
    index = np.maximum(np.searchsorted(section_beats, beats, side="right") - 1, 0)
    return offsets[index] + (beats - section_beats[index]) * lengths[index]


def _chart_from_bytes(name, content, chart_path):
    text = content.decode("utf-8-sig", errors="replace")
    metadata, sections, notes, issues = parse_chart_text(text)
    if not is_mania_4k(metadata):
        raise ValueError(f"{name}: only osu!mania 4K charts can be evaluated.")
    if not sections:
        raise ValueError(f"{name}: the chart has no uninherited timing point.")
    if not notes:
        raise ValueError(f"{name}: the chart has no hit objects.")
    return ChartInput(name=name, chart_bytes=content, chart_path=chart_path, metadata=metadata,
                      clock=BeatClock(sections), notes=notes, issues=issues)


def _load_package(path):
    temporary = tempfile.TemporaryDirectory(prefix="keyloom-eval-")
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            charts = [name for name in names if name.lower().endswith(".osu")]
            if not charts:
                raise ValueError(f"{path.name}: the package contains no .osu chart.")
            content = archive.read(charts[0])
            chart_path = Path(temporary.name) / "chart.osu"
            chart_path.write_bytes(content)
            chart = _chart_from_bytes(path.name, content, chart_path)
            if len(charts) > 1:
                chart.issues.append("multiple_charts_in_package_first_evaluated")
            documents = {}
            for document in ("bar-density.json", "density-execution.json", "export-provenance.json"):
                if document in names:
                    documents[document] = json.loads(archive.read(document).decode("utf-8"))
            audio_name = chart.metadata.get("AudioFilename")
            if audio_name in names and Path(audio_name).suffix.lower() in AUDIO_SUFFIXES:
                chart.audio_path = Path(temporary.name) / ("audio" + Path(audio_name).suffix.lower())
                chart.audio_path.write_bytes(archive.read(audio_name))
    except BaseException:
        temporary.cleanup()
        raise
    chart._temporary = temporary
    if "bar-density.json" in documents:
        chart.plan_bars = documents["bar-density.json"].get("bars")
    if "density-execution.json" in documents:
        chart.execution = documents["density-execution.json"].get("summary")
    settings = documents.get("export-provenance.json", {}).get("settings", {})
    if settings.get("target_star") is not None:
        chart.target_star = float(settings["target_star"])
    return chart


def load_chart(path):
    """Load a .osz package, a .osu/extensionless chart file, or a generation directory."""
    path = Path(path)
    if path.is_dir():
        packages = sorted(path.glob("*.osz"), key=lambda item: item.stat().st_mtime)
        charts = sorted(path.glob("*.osu"))
        if packages:
            path = packages[-1]
        elif charts:
            path = charts[0]
        else:
            raise ValueError(f"{path}: no .osz package or .osu chart in this directory.")
    if not path.is_file():
        raise ValueError(f"{path}: chart file does not exist.")
    if zipfile.is_zipfile(path):
        return _load_package(path)
    chart = _chart_from_bytes(path.name, path.read_bytes(), path)
    audio_name = chart.metadata.get("AudioFilename")
    if audio_name:
        sibling = path.parent / audio_name
        if sibling.is_file() and sibling.suffix.lower() in AUDIO_SUFFIXES:
            chart.audio_path = sibling
    return chart


def discover_charts(roots, maximum_bytes=3_000_000):
    """Yield chart files under roots, including osu!lazer's extensionless file store."""
    for root in roots:
        root = Path(root)
        candidates = [root] if root.is_file() else (Path(directory) / name for directory, _, files in os.walk(root) for name in files)
        for candidate in candidates:
            try:
                if candidate.stat().st_size > maximum_bytes:
                    continue
                with candidate.open("rb") as source:
                    head = source.read(4096)
            except OSError:
                continue
            if not HEADER_PATTERN.search(head[:128]):
                continue
            text = head.decode("utf-8", errors="replace")
            if re.search(r"^Mode:\s*3\s*$", text, re.M) and re.search(r"^CircleSize:\s*4(\.0+)?\s*$", text, re.M):
                yield candidate
