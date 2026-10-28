"""Check requested short-jack bounds on the actual exported millisecond chart."""
import bisect
import collections

from .lane_control import absolute_beats


def audit_minijack_export(clock, notes, bars):
    starts = [bar["start_ms"] for bar in bars]
    rows = collections.defaultdict(int)
    for note in notes:
        rows[note.time_ms] |= 1 << note.column
    times = sorted(rows)
    beats = absolute_beats(clock, times)
    last_hits, column_runs = [None] * 4, [0] * 4
    hand_runs, maximum_hands, maximum_columns = [0, 0], [0, 0], [0] * 4
    previous_beat = None
    violations = []
    for time_ms, beat in zip(times, beats):
        mask = rows[time_ms]
        tolerance = 1.01 / clock.sections[clock.section_at(time_ms)].beat_length_ms
        bar_index = bisect.bisect_right(starts, time_ms + 0.51) - 1
        mini = bar_index >= 0 and bars[bar_index].get("style") == "Minijacks"
        continuous = previous_beat is not None and beat - previous_beat <= 0.5 + tolerance
        for hand, bits in enumerate((3, 12)):
            hand_runs[hand] = (hand_runs[hand] + 1 if continuous else 1) if mask & bits else 0
            if mini:
                maximum_hands[hand] = max(maximum_hands[hand], hand_runs[hand])
                if hand_runs[hand] > 5:
                    violations.append({"kind": "hand", "hand": hand, "length": hand_runs[hand], "time_ms": time_ms, "bar": bar_index + 1})
        for column in range(4):
            if not mask & (1 << column):
                continue
            adjacent = last_hits[column] is not None and abs(beat - last_hits[column] - 0.25) <= tolerance
            column_runs[column] = column_runs[column] + 1 if adjacent else 1
            last_hits[column] = beat
            if mini:
                maximum_columns[column] = max(maximum_columns[column], column_runs[column])
                if column_runs[column] > 3:
                    violations.append({"kind": "column", "column": column, "length": column_runs[column], "time_ms": time_ms, "bar": bar_index + 1})
        previous_beat = beat
    return {"passed": not violations, "maximum_minijack_column_runs": maximum_columns,
            "maximum_minijack_hand_runs": maximum_hands, "violations": violations,
            "policy": "Actual planned Minijacks only; exported milliseconds with 1.01 ms rounding tolerance. Same hand participation counts one per occupied row."}
