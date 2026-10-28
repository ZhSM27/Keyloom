from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from ._vendor.omtk_pattern.find_patterns import find
from ._vendor.omtk_pattern.osu_parser import parse_osu_mania
from ._vendor.omtk_pattern.primitives import calculate_primitives
from ._vendor.omtk_pattern.summary import from_chart


DETECTOR_COMMIT = "fff821e54e34c7a9649588ec758ead5353e9f2f5"
SCHEMA_VERSION = "omtk-patterns-v1"


def _read_sections(text: str) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current_section = ""
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            sections.setdefault(current_section, {})
        elif ":" in line and current_section in {"General", "Difficulty", "Metadata"}:
            key, value = line.split(":", 1)
            sections[current_section][key.strip()] = value.strip()
    return sections


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def analyze_file(path: str | Path, include_rows: bool = False) -> dict:
    beatmap_path = Path(path)
    source_bytes = beatmap_path.read_bytes()
    sections = _read_sections(source_bytes.decode("utf-8-sig", errors="replace"))
    general = sections.get("General", {})
    difficulty = sections.get("Difficulty", {})
    if int(general.get("Mode", "0")) != 3:
        raise ValueError("Only native osu!mania Mode:3 beatmaps are supported.")
    if float(difficulty.get("CircleSize", "0")) != 4:
        raise ValueError("Only 4K beatmaps are supported.")

    chart = parse_osu_mania(str(beatmap_path))
    if not chart.Notes:
        raise ValueError("The beatmap has no supported hit objects.")
    for row in chart.Notes:
        if not math.isfinite(row.Time):
            raise ValueError("Hit object times must be finite.")
    for timing in chart.BPM:
        if not math.isfinite(timing.Time) or not math.isfinite(timing.Data.MsPerBeat):
            raise ValueError("Timing point values must be finite.")
        if timing.Data.MsPerBeat <= 0:
            raise ValueError("Uninherited timing points must have positive beat lengths.")

    report = from_chart(chart)
    found_patterns = find(chart)
    primitives = calculate_primitives(chart)
    first_note = chart.FirstNote
    last_note = chart.LastNote
    duration = last_note - first_note
    matches = []
    intervals_by_label: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for pattern in found_patterns:
        label = pattern.SpecificType or pattern.Pattern.value
        start = first_note + pattern.Start
        end = first_note + pattern.End
        matches.append(
            {
                "family": pattern.Pattern.value,
                "label": label,
                "start_ms": start,
                "end_ms": end,
                "mixed_spacing": pattern.Mixed,
                "spacing_proxy_ms_per_beat": pattern.MsPerBeat,
            }
        )
        intervals_by_label[label].append((max(first_note, start), min(last_note, end)))

    coverage = {}
    for label, intervals in sorted(intervals_by_label.items()):
        merged = _merge_intervals(intervals)
        covered_duration = sum(end - start for start, end in merged)
        coverage[label] = {
            "duration_ms": covered_duration,
            "fraction_of_chart_span": covered_duration / duration if duration > 0 else 0,
            "intervals_ms": [[start, end] for start, end in merged],
        }

    warnings = [
        "Labels describe chart geometry; they do not evaluate correspondence to audio.",
        "No match is not a verified negative label. Some short or single-lane patterns are missed.",
        "Pattern spacing/BPM proxies describe row spacing, not necessarily the song tempo.",
        "Coverage is a union within each label; different labels may overlap and need not sum to one.",
        "This inherited parser can reject some LN tail/head joins. A parse error does not establish that a chart is invalid.",
    ]
    if len(chart.Notes) < 6:
        warnings.append("The chart is short relative to the detector's recognition windows.")

    result = {
        "schema_version": SCHEMA_VERSION,
        "detector": {
            "name": "nonebot-plugin-osumania-toolkit",
            "commit": DETECTOR_COMMIT,
            "configuration": "upstream default snapshot",
            "recognition_rules_modified": False,
        },
        "source": {
            "path": str(beatmap_path.resolve()),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "audio_filename": general.get("AudioFilename"),
            "metadata": sections.get("Metadata", {}),
        },
        "chart": {
            "keys": chart.Keys,
            "first_note_ms": first_note,
            "last_note_ms": last_note,
            "span_ms": duration,
            "ln_head_fraction": report.LNPercent,
        },
        "upstream_summary": {
            "category": report.Category,
            "mode_tag": report.ModeTag,
            "sv_amount_ms": report.SVAmount,
            "clusters": [
                {
                    "family": cluster.Pattern.value,
                    "specific_types": [
                        {"label": label, "match_fraction": fraction}
                        for label, fraction in cluster.SpecificTypes
                    ],
                    "spacing_proxy_bpm": cluster.BPM,
                    "mixed_spacing": cluster.Mixed,
                    "amount_ms": cluster.Amount,
                    "importance_proxy": cluster.Importance,
                }
                for cluster in report.Clusters
            ],
        },
        "matches": matches,
        "coverage": coverage,
        "warnings": warnings,
    }
    if include_rows:
        result["primitive_rows"] = [
            {
                "time_ms": first_note + row.Time,
                "columns": row.RawNotes,
                "pressed_count": row.Notes,
                "repeated_column_count": row.Jacks,
                "direction": row.Direction.value,
                "is_roll": row.Roll,
                "song_beat_length_ms": row.BeatLength,
                "spacing_proxy_ms_per_beat": row.MsPerBeat,
                "tap_columns": row.NormalNotes,
                "ln_head_columns": row.LNHeads,
                "ln_body_columns": row.LNBodies,
                "ln_tail_columns": row.LNTails,
            }
            for row in primitives
        ]
    return result


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Export OMTK 4K pattern labels as JSON.")
    parser.add_argument("beatmap", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-rows", action="store_true")
    arguments = parser.parse_args()
    try:
        result = analyze_file(arguments.beatmap, include_rows=arguments.include_rows)
        content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if arguments.output:
            if arguments.output.resolve() == arguments.beatmap.resolve():
                raise ValueError("The output path must differ from the input beatmap path.")
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(content, encoding="utf-8")
            print(f"Saved {len(result['matches'])} pattern matches to {arguments.output}")
        else:
            print(content, end="")
    except (OSError, ValueError, AssertionError, IndexError, ZeroDivisionError) as error:
        print(f"Pattern analysis failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
