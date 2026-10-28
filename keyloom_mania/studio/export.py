"""Portable window scheduling and native osu!mania export."""
from pathlib import Path
import re
import numpy as np
from ..segment_labels import LABELS

def beat_plan_window(plan_lookup, section_index, start_beat):
    values = [plan_lookup.get((int(section_index), int(beat // 4)), np.zeros(len(LABELS), dtype=np.float32)) for beat in range(start_beat, start_beat + 8)]
    return np.repeat(np.asarray(values, dtype=np.float32), 24, axis=0)


def section_budget_schedule(lookup, section_index, total_beats, budget_scale):
    result = {}
    for field in ("rows", "notes"):
        values = np.asarray([float(lookup.get((section_index, beat // 4), {}).get(field, 0.0)) for beat in range(total_beats)], dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Section budgets must be finite and nonnegative.")
        values = np.clip(values * budget_scale, 0, 24 if field == "rows" else 96)
        rounded = np.floor(np.cumsum(values) + 0.5)
        result[field] = np.diff(np.concatenate(([0], rounded))).astype(np.float32)
    return result


def controlled_window_starts(total_beats, context_beats=2):
    if context_beats not in (0, 1, 2, 3):
        raise ValueError("Continuation context must be zero through three beats.")
    if total_beats < 8:
        return []
    final_start = total_beats - 8
    starts = list(range(0, final_start + 1, 8 - context_beats))
    if context_beats and starts[-1] != final_start:
        starts.append(final_start)
    return starts


def write_full_osu(path: Path, chart: dict, notes, clock, audio_name: str, creator: str):
    clean = lambda text: str(text).replace("\r", " ").replace("\n", " ")
    timing = "\n".join(f"{section.offset_ms},{section.beat_length_ms},{section.meter},1,0,100,1,0" for section in clock.sections)
    text = f"osu file format v14\n\n[General]\nAudioFilename: {audio_name}\nMode: 3\nPreviewTime: {int(notes[0][0] if notes else 0)}\n\n[Metadata]\nTitle:{clean(chart['metadata'].get('Title', 'Unknown'))}\nArtist:{clean(chart['metadata'].get('Artist', 'Unknown'))}\nCreator:{clean(creator)}\nVersion:{clean(chart['metadata'].get('Version', '4K'))} · Planner + compiled plan\nBeatmapID:0\nBeatmapSetID:-1\n\n[Difficulty]\nHPDrainRate:8\nCircleSize:4\nOverallDifficulty:8\nApproachRate:5\nSliderMultiplier:1.4\nSliderTickRate:1\n\n[TimingPoints]\n{timing}\n\n[HitObjects]\n"
    text += "\n".join(f"{64 + column * 128},192,{time_ms},1,0,0:0:0:0:" for time_ms, column in sorted(set(notes))) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")

def safe_name(text):
    return re.sub(r'[<>:"/\\|?*]+', "_", str(text)).strip()
