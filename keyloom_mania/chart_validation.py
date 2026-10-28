"""Validation of native osu!mania chart text."""
import collections
import math
from .patterns import _read_sections


def chart_statistics(text: str) -> dict:
    section = ""
    notes = []
    bpms = []
    errors = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        try:
            if section == "TimingPoints":
                fields = line.split(",")
                time_ms, beat_length = float(fields[0]), float(fields[1])
                uninherited = len(fields) < 7 or fields[6].strip() == "1"
                if not math.isfinite(time_ms) or not math.isfinite(beat_length):
                    raise ValueError("Non-finite timing point.")
                if uninherited:
                    if beat_length <= 0:
                        raise ValueError("Non-positive beat length.")
                    bpms.append(60000 / beat_length)
            elif section == "HitObjects":
                fields = line.split(",")
                position = float(fields[0])
                start_ms = float(fields[2])
                object_type = int(fields[3])
                if not math.isfinite(position) or not math.isfinite(start_ms):
                    raise ValueError("Non-finite hit object.")
                if position < 0 or position > 512:
                    raise ValueError("Hit object outside the playfield.")
                column = min(3, int(position * 4 / 512))
                hold = bool(object_type & 128)
                if not hold and not object_type & 1:
                    raise ValueError("Unsupported mania hit object type.")
                end_ms = float(fields[5].split(":")[0]) if hold else start_ms
                if not math.isfinite(end_ms) or (hold and end_ms <= start_ms):
                    raise ValueError("Invalid hold end time.")
                notes.append((start_ms, column, end_ms, hold))
        except (ValueError, IndexError, OverflowError) as error:
            errors.append(f"Line {line_number}: {error}")
    notes.sort()
    if not bpms:
        errors.append("No positive uninherited timing point was found.")
    previous_end = {}
    previous_start = {}
    rows: dict[float, list[int]] = collections.defaultdict(list)
    for start_ms, column, end_ms, hold in notes:
        if previous_start.get(column) == start_ms:
            errors.append(f"Duplicate hit at {start_ms} ms in column {column}.")
        if previous_end.get(column, float("-inf")) > start_ms:
            errors.append(f"Overlapping hold at {start_ms} ms in column {column}.")
        previous_start[column] = start_ms
        previous_end[column] = end_ms
        rows[start_ms].append(column)
    hold_count = sum(hold for _, _, _, hold in notes)
    first_note = notes[0][0] if notes else None
    last_note = max(end_ms for _, _, end_ms, _ in notes) if notes else None
    span_ms = last_note - first_note if notes else 0
    chord_counts = collections.Counter(len(columns) for columns in rows.values())
    previous_columns = set()
    repeated_rows = 0
    repeated_columns = 0
    for _, columns in sorted(rows.items()):
        overlap = len(previous_columns.intersection(columns))
        repeated_rows += bool(overlap)
        repeated_columns += overlap
        previous_columns = set(columns)
    return {
        "note_count": len(notes),
        "hold_count": hold_count,
        "ln_head_fraction": hold_count / len(notes) if notes else 0,
        "first_note_ms": first_note,
        "last_note_ms": last_note,
        "span_ms": span_ms,
        "notes_per_second": len(notes) / (span_ms / 1000) if span_ms else 0,
        "onset_row_count": len(rows),
        "chord_row_counts": {str(count): amount for count, amount in sorted(chord_counts.items())},
        "adjacent_repeated_row_fraction": repeated_rows / max(1, len(rows) - 1),
        "adjacent_repeated_column_count": repeated_columns,
        "bpm_min": min(bpms) if bpms else None,
        "bpm_max": max(bpms) if bpms else None,
        "validation_errors": errors,
    }
