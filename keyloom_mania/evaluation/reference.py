"""Human chart reference profiles: aggregate observations only, never chart text.

A profile is calibration data tied to ``FEATURE_SCHEMA``. Rebuild it whenever the
feature semantics change; never reuse a profile across schema versions.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..paths import APP_HOME
from .audio import AUDIO_SCHEMA
from .charts import chart_rows, discover_charts, is_generated, load_chart
from .definitions import chart_context
from .features import FEATURE_SCHEMA, combine, extract_blocks, family_shares

REFERENCE_SCHEMA = "human-chart-reference-v1"
DEFAULT_REFERENCE = APP_HOME / "evaluation" / "human-reference.json.gz"


def star_rating(path):
    try:
        import rosu_pp_py
    except ImportError:
        return None
    try:
        return float(rosu_pp_py.Difficulty().calculate(rosu_pp_py.Beatmap(path=str(path))).stars)
    except Exception:
        return None


def dominant_bpm(clock, times):
    sections = np.maximum(np.searchsorted(np.asarray(clock.offsets), times, side="right") - 1, 0)
    section = int(np.bincount(sections).argmax())
    return 60_000 / clock.sections[section].beat_length_ms


def profile_chart(chart, *, with_audio=True, waveform=None):
    """Observations shared by reference charts and evaluated charts."""
    times, masks, holds = chart_rows(chart.notes)
    blocks = extract_blocks(chart.clock, times, masks, holds)
    span_seconds = max(1e-3, (times[-1] - times[0]) / 1000)
    record = {"sha256": chart.sha256, "star": star_rating(chart.chart_path),
              "bpm": dominant_bpm(chart.clock, times), "rows": int(len(times)), "notes": int(sum(m.bit_count() for m in masks.tolist())),
              "hold_fraction": float(holds.sum() / max(1, len(chart.notes))), "span_seconds": span_seconds,
              "notes_per_second": len(chart.notes) / span_seconds, "timing_sections": len(chart.clock.sections),
              "family": family_shares(combine(blocks)), "blocks": blocks, "audio": None}
    if with_audio and chart.audio_path is not None:
        try:
            from .audio import audio_metrics
            stream = chart_context(chart.clock, times, masks).stream
            record["audio"] = audio_metrics(chart.audio_path, chart.clock, times, masks, stream, waveform)
        except Exception as error:
            record["audio_error"] = f"{type(error).__name__}: {error}"
    return record, (times, masks, holds)


def _reference_worker(task):
    """Profile charts that share one audio file, decoding that audio once."""
    entries, options = task
    waveform, results = None, []
    for path, audio in entries:
        results.append(_reference_chart(path, audio, options, waveform))
        waveform = results[-1].pop("_waveform", waveform)
    return results


def _reference_chart(path, audio, options, waveform):
    try:
        chart = load_chart(path)
    except Exception as error:
        return {"skipped": "unreadable", "detail": str(error)[:120]}
    if audio is not None:
        chart.audio_path = Path(audio)
    with chart:
        if is_generated(chart.metadata):
            return {"skipped": "generated_or_research_chart"}
        if not 30 <= dominant_bpm(chart.clock, chart_rows(chart.notes)[0]) <= 500:
            return {"skipped": "tempo_outside_range"}
        if len(chart.clock.sections) > options["maximum_timing_sections"]:
            return {"skipped": "variable_timing"}
        holds = sum(note.end_ms is not None for note in chart.notes)
        if holds / len(chart.notes) > options["maximum_hold_fraction"]:
            return {"skipped": "long_note_heavy"}
        if len(chart.notes) < options["minimum_notes"]:
            return {"skipped": "too_short"}
        try:
            if options["with_audio"] and chart.audio_path is not None and waveform is None:
                from .audio import decode_audio
                waveform = decode_audio(chart.audio_path)
            record, _ = profile_chart(chart, with_audio=options["with_audio"], waveform=waveform)
        except Exception as error:
            return {"skipped": "feature_error", "detail": f"{type(error).__name__}: {error}"[:120]}
        record["_waveform"] = waveform
        return record


def manifest_entries(manifest, splits=("train",)):
    """(chart, audio) pairs from a corpus manifest; only the named splits are read."""
    manifest = Path(manifest)
    entries = []
    for line in manifest.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("split") in splits and row.get("chart_relative_path") and row.get("audio_relative_path"):
            entries.append((manifest.parent / row["chart_relative_path"], manifest.parent / row["audio_relative_path"]))
    return entries


def build_reference(roots, output=DEFAULT_REFERENCE, *, manifests=(), splits=("train",), maximum_hold_fraction=.1,
                    minimum_notes=200, maximum_timing_sections=32, with_audio=True, workers=None, progress=print):
    """Scan chart folders (osu! Songs, the lazer file store) and corpus manifests into a reference profile."""
    started = time.time()
    candidates, digests = [], set()
    paired = [(path, None) for path in discover_charts(roots)]
    for manifest in manifests:
        paired += manifest_entries(manifest, splits)
    for path, audio in paired:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest not in digests:
            digests.add(digest)
            candidates.append((str(path), None if audio is None else str(audio)))
    options = {"maximum_hold_fraction": maximum_hold_fraction, "minimum_notes": minimum_notes,
               "maximum_timing_sections": maximum_timing_sections, "with_audio": with_audio}
    progress(f"Profiling {len(candidates)} unique 4K charts…")
    charts, skipped = [], Counter()
    groups = {}
    for path, audio in candidates:
        groups.setdefault(audio or path, []).append((path, audio))
    tasks = [(entries, options) for entries in groups.values()]
    if workers == 1:
        results = map(_reference_worker, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_reference_worker, tasks, chunksize=4)
    try:
        done = 0
        for batch in results:
            for result in batch:
                result.pop("_waveform", None)
                done += 1
                if "skipped" in result:
                    skipped[result["skipped"]] += 1
                else:
                    charts.append(result)
                if done % 200 == 0:
                    progress(f"  {done}/{len(candidates)} charts, {len(charts)} accepted")
    finally:
        if pool is not None:
            pool.shutdown()
    if not charts:
        raise ValueError("No human 4K chart passed the reference filters.")
    charts.sort(key=lambda item: item["sha256"])
    profile = {"schema": REFERENCE_SCHEMA, "feature_schema": FEATURE_SCHEMA, "audio_schema": AUDIO_SCHEMA,
               "created_unix": int(time.time()),
               "filters": {**options, "generated_markers_excluded": True, "duplicates": "sha256",
                           "manifest_splits": list(splits) if manifests else []},
               "summary": {"candidates": len(candidates), "accepted": len(charts), "skipped": dict(skipped),
                           "with_audio": sum(item["audio"] is not None for item in charts),
                           "seconds": round(time.time() - started, 1)},
               "corpus_sha256": hashlib.sha256("".join(item["sha256"] for item in charts).encode()).hexdigest(),
               "charts": charts,
               "boundary": "Local human charts of unknown curation; aggregate observations only. A screening reference, not a quality label."}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as target:
        json.dump(profile, target, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return profile


def load_reference(path=DEFAULT_REFERENCE):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Reference profile not found: {path}. Build one with `keyloom evaluate reference <chart folders>`.")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        profile = json.load(source)
    if profile.get("schema") != REFERENCE_SCHEMA or profile.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("The reference profile uses different feature semantics; rebuild it for "
                         f"{FEATURE_SCHEMA} instead of reusing old calibration.")
    if not all(math.isfinite(item["bpm"]) for item in profile["charts"]):
        raise ValueError("The reference profile contains invalid tempo records.")
    # Compact arrays keep several evaluation workers within modest memory.
    for item in profile["charts"]:
        for block in item["blocks"]:
            block["hist"] = {name: np.asarray(values, dtype=np.int32) for name, values in block["hist"].items()}
            block["style_hist"] = {label: {name: np.asarray(values, dtype=np.int32) for name, values in histograms.items()}
                                   for label, histograms in block["style_hist"].items()}
    return profile
