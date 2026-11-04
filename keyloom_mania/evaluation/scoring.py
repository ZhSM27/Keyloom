"""Human-referenced chart scorecards and A/B comparison.

Every scored metric is a percentile against equally long segments of human
charts with similar difficulty, tempo and style mix. Too little and too much
are judged alike; no axis rewards lower pressure or balanced hands. Scores
screen candidates for listening and playtests; they never promote a model alone.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .charts import ROW_MERGE_MS
from .definitions import DEFINITION_SCHEMA, EVALUATION_LABELS as LABELS, RULES, bar_matches, chart_context
from .features import FEATURE_SCHEMA, HISTOGRAMS, STYLE_FEATURES, combine, scalar_metrics
from .audio import AUDIO_SCHEMA
from .reference import profile_chart

EVALUATION_VERSION = "chart-evaluation-v4"
AXES = {
    "style": "键型兑现",
    "style_natural": "键型内自然度",
    "hand_pressure": "手部压力（2/4 拍）",
    "layout": "排键",
    "rhythm": "节奏",
    "music": "音乐对应",
}
# name, axis, kind, label. Kinds: two_sided, low_bad, high_bad, divergence (higher
# is further from humans), direct (already a 0..1 realization), info (not scored).
METRICS = (
    ("style_self_consistency", "style", "low_bad", "键型清晰度（定义自洽率）"),
    ("plan_style_realization", "style", "direct", "计划键型兑现（相对人类上限）"),
    ("defined_bar_fraction", "style", "info", "符合某一键型定义的小节占比（仅参考）"),
    ("hand_rows_2b_mean", "hand_pressure", "two_sided", "2 拍内单手参与行数·均值"),
    ("hand_notes_4b_mean", "hand_pressure", "two_sided", "4 拍内单手音符数·均值"),
    ("hand_notes_4b_p90", "hand_pressure", "two_sided", "4 拍内单手音符数·p90"),
    ("column_hits_4b_p90", "hand_pressure", "two_sided", "4 拍内单列击打数·p90"),
    ("hand_nps_4b_mean", "hand_pressure", "two_sided", "4 拍内单手每秒音符·均值"),
    ("hand_bias_4b_mean", "hand_pressure", "two_sided", "4 拍局部偏手程度·均值"),
    ("js:hand_rows_2b", "hand_pressure", "divergence", "2 拍单手参与行数分布"),
    ("js:hand_notes_4b", "hand_pressure", "divergence", "4 拍单手音符数分布"),
    ("js:column_hits_4b", "hand_pressure", "divergence", "4 拍单列击打数分布"),
    ("js:hand_bias_4b", "hand_pressure", "divergence", "局部偏手分布"),
    ("js:same_column_gap", "hand_pressure", "divergence", "同列间隔分布"),
    ("js:same_hand_gap", "hand_pressure", "divergence", "同手间隔分布"),
    ("js:hand_run", "hand_pressure", "divergence", "同手连续参与分布"),
    ("js:anchor_share", "hand_pressure", "divergence", "流中单列堆积分布"),
    ("js:chord_size", "layout", "divergence", "押数分布"),
    ("js:transition", "layout", "divergence", "相邻行转移分布"),
    ("jack_transition_fraction", "layout", "two_sided", "叠键转移占比"),
    ("column_entropy", "layout", "low_bad", "列使用均匀度"),
    ("long_jack_runs_per_100_bars", "layout", "high_bad", "普通长叠（≥5 连 1/4）·每百小节"),
    ("near_identical_bar_fraction", "layout", "two_sided", "相邻小节近似复制占比"),
    ("motif_unique_fraction", "layout", "two_sided", "小节排键多样性"),
    ("js:jack_run", "layout", "divergence", "同列连打长度谱（2 小叠 / 3+ 叠 / 5+ 长叠）"),
    ("js:trill_run", "layout", "divergence", "颤长度分布"),
    ("js:roll_run", "layout", "divergence", "楼梯长度分布"),
    ("left_hand_share", "layout", "info", "左手音符占比（仅参考，不计分）"),
    ("js:snap", "rhythm", "divergence", "节奏细分分布"),
    ("js:row_gap", "rhythm", "divergence", "行间隔分布"),
    ("bar_density_cv", "rhythm", "two_sided", "小节密度起伏（变异系数）"),
    ("stream_regularity", "rhythm", "two_sided", "等间隔流占比（≥1 拍）"),
    ("chord_strong_beat_fraction", "rhythm", "two_sided", "流中和弦落在 1/1、1/2 拍的占比"),
    ("grace_per_100_rows", "rhythm", "two_sided", "倚音（孤立的 ≤1/6 拍间隔）·每百行"),
    ("onset_precision", "music", "low_bad", "落点贴合音频起音"),
    ("timing_lift", "music", "low_bad", "时间对齐提升（对比偏移网格）"),
    ("selection_lift", "music", "low_bad", "选拍提升（对比整拍平移）"),
    ("strong_onset_recall", "music", "low_bad", "强起音覆盖率"),
    ("energy_density_spearman", "music", "low_bad", "音频能量与小节密度相关"),
    ("accent_lift", "music", "two_sided", "重音倍数（多压踩强重音 ÷ 单键踩强重音）"),
    ("strong_accent_chord_recall", "music", "two_sided", "强重音由多压覆盖的比例"),
    ("size_strength_spearman", "music", "low_bad", "押数与起音强度的相关"),
    ("unsupported_chords", "music", "high_bad", "无起音支撑的多压比例"),
    ("unsupported_offstream_rows", "music", "high_bad", "流外无起音支撑的行比例"),
    ("beats_rows_exceed_supply", "music", "high_bad", "行数超过可踩起音 +1 的拍子比例"),
)
METRIC_INFO = {name: {"axis": axis, "kind": kind, "label": label} for name, axis, kind, label in METRICS}
STYLE_FEATURE_LABELS = {"hand_rows_2b": "2 拍单手参与行数", "hand_notes_4b": "4 拍单手音符数", "column_hits_4b": "4 拍单列击打数",
                        "same_hand_gap": "同手间隔", "chord_size": "押数", "transition": "相邻行转移", "row_gap": "行间隔", "snap": "节奏细分"}
STYLE_WINDOW_BARS = 32
MINIMUM_STYLE_BARS = 8
DIVERGENCES = tuple(name[3:] for name, *_ in METRICS if name.startswith("js:"))
# Segment-calibrated axes; music is calibrated separately against human charts with audio.
NATURALNESS_AXES = ("hand_pressure", "layout", "rhythm")
AUDIO_METRICS = tuple(name for name, axis, *_ in METRICS if axis == "music")
MINIMUM_SAMPLES = 8
MINIMUM_DIVERGENCE_OBSERVATIONS = 16
IMPOSSIBLE_JACK_MS = 40.0
FLAM_MS = 10.0
BOUNDARY = ("Screening against local human charts of similar difficulty, tempo and style mix. Percentiles are "
            "descriptive and depend on the community-definition pattern rules; listening, phrase review and playtests decide musical naturalness. "
            "Never promote a model on these scores alone, and never for lower pressure alone.")


def js_divergence(first, second, prior=.5):
    p = np.asarray(first, dtype=np.float64) + prior
    q = np.asarray(second, dtype=np.float64) + prior
    p, q = p / p.sum(), q / q.sum()
    middle = (p + q) / 2

    def relative(first):
        support = first > 0
        return np.sum(first[support] * np.log2(first[support] / middle[support]))
    return float(.5 * relative(p) + .5 * relative(q))


def _clean(samples):
    return np.asarray([item for item in samples if item is not None and math.isfinite(item)], dtype=np.float64)


def tail_probability(percentile, kind):
    """Probability of a human value at least this far out; the favourable half of one-sided metrics is 1."""
    if kind == "two_sided":
        return 2 * min(percentile, 1 - percentile)
    if kind == "low_bad":
        return min(1., 2 * percentile)
    return min(1., 2 * (1 - percentile))


def typicality(value, samples, kind):
    """Place a value in the human sample: percentile, tail probability and a status."""
    samples = _clean(samples)
    if value is None or len(samples) < MINIMUM_SAMPLES:
        return None
    percentile = float((np.sum(samples < value) + .5 * np.sum(samples == value)) / len(samples))
    tail = tail_probability(percentile, kind)
    # Two-sided central 90% (p05..p95) is typical; one-sided metrics use the same tail mass.
    status = "typical" if tail >= .1 else "unusual" if tail >= .02 else "atypical"
    p05, median, p95 = np.quantile(samples, (.05, .5, .95))
    return {"percentile": percentile, "tail": tail, "status": status,
            "reference": {"median": float(median), "p05": float(p05), "p95": float(p95), "samples": int(len(samples))}}


def _surprise(tail, count):
    return -math.log(max(tail, 1 / (count + 1)))


def human_likeness(values, rows, names, kinds=None):
    """Share of human segments at least as atypical as the chart over the same metrics.

    Each metric contributes -log(tail probability); human segments are scored
    leave-one-out against the other segments, so a human chart is uniform here.
    """
    columns = {}
    for name in names:
        kind = (kinds or {}).get(name) or METRIC_INFO[name]["kind"]
        samples = np.asarray([np.nan if row.get(name) is None else row[name] for row in rows], dtype=np.float64)
        finite = np.sort(samples[np.isfinite(samples)])
        if values.get(name) is None or len(finite) < MINIMUM_SAMPLES:
            continue
        columns[name] = (kind, samples, finite)
    if not columns:
        return None, {}
    chart_terms, human_terms = [], []
    for name, (kind, samples, finite) in columns.items():
        value = values[name]
        percentile = (np.sum(finite < value) + .5 * np.sum(finite == value)) / len(finite)
        chart_terms.append(_surprise(tail_probability(percentile, kind), len(finite)))
        less = np.searchsorted(finite, samples, side="left")
        equal = np.searchsorted(finite, samples, side="right") - less - 1
        loo = np.where(np.isfinite(samples), (less + .5 * equal) / max(1, len(finite) - 1), np.nan)
        human_terms.append([np.nan if not math.isfinite(p) else _surprise(tail_probability(p, kind), len(finite) - 1) for p in loo])
    chart_deviation = float(np.mean(chart_terms))
    terms = np.asarray(human_terms, dtype=np.float64)
    present = np.isfinite(terms).sum(axis=0)
    human_deviation = np.where(np.isfinite(terms), terms, 0).sum(axis=0)[present > 0] / present[present > 0]
    if len(human_deviation) < MINIMUM_SAMPLES:
        return None, {}
    score = float(np.mean(human_deviation >= chart_deviation - 1e-9))
    return round(100 * score, 1), {"deviation": chart_deviation, "human_median_deviation": float(np.median(human_deviation)),
                                   "metrics": len(columns), "human_segments": int(len(human_deviation))}


def _distance(profile, item):
    distance = 0.
    if profile["star"] is not None and item["star"] is not None:
        distance += abs(profile["star"] - item["star"]) / .75
    else:
        distance += abs(math.log(max(profile["notes_per_second"], .1) / max(item["notes_per_second"], .1))) / .3
    distance += abs(math.log2(profile["bpm"] / item["bpm"])) / .5
    if profile["family"] and item["family"]:
        distance += sum(abs(a - b) for a, b in zip(profile["family"], item["family"])) / .5
    return distance


def select_neighbors(reference, profile, count=40):
    candidates = [item for item in reference["charts"] if item["sha256"] != profile["sha256"]]
    return sorted(candidates, key=lambda item: _distance(profile, item))[:count]


def _segments(item, length):
    blocks = item["blocks"]
    if len(blocks) <= length:
        return [combine(blocks)]
    starts = sorted({int(round(value)) for value in np.linspace(0, len(blocks) - length, min(3, len(blocks) - length + 1))})
    return [combine(blocks[start:start + length]) for start in starts]


def _divergences(segment, pooled):
    result = {}
    for name in DIVERGENCES:
        histogram = segment["hist"][name]
        result["js:" + name] = js_divergence(histogram, pooled[name]) if np.sum(histogram) >= MINIMUM_DIVERGENCE_OBSERVATIONS else None
    return result


def calibrate(profile, reference, count=40):
    """Chart values plus matched-length human samples for every segment metric."""
    segment = combine(profile["blocks"])
    neighbors = select_neighbors(reference, profile, count)
    full = [combine(item["blocks"]) for item in neighbors]
    pooled = {name: sum((record["hist"][name] for record in full), np.zeros_like(segment["hist"][name])) for name in segment["hist"]}
    rows = []
    for item, whole in zip(neighbors, full):
        excluded = {name: pooled[name] - whole["hist"][name] for name in pooled}
        for part in _segments(item, len(profile["blocks"])):
            rows.append({**scalar_metrics(part), **_divergences(part, excluded)})
    values = {**scalar_metrics(segment), **_divergences(segment, pooled)}
    values.update({name: (profile.get("audio") or {}).get(name) for name in AUDIO_METRICS})
    # Audio metrics depend on difficulty and style too: calibrate against the nearest charts that have audio.
    audio_charts = [item for item in reference["charts"] if item.get("audio") and item["audio"].get("schema") == AUDIO_SCHEMA]
    audio_rows = [item["audio"] for item in select_neighbors({"charts": audio_charts}, profile, count)] if profile.get("audio") else []
    stars = [item["star"] for item in neighbors if item["star"] is not None]
    summary = {"count": len(neighbors), "star_range": [min(stars), max(stars)] if stars else None,
               "bpm_range": [min(item["bpm"] for item in neighbors), max(item["bpm"] for item in neighbors)] if neighbors else None,
               "segment_blocks": len(profile["blocks"]), "human_segments": len(rows), "audio_reference_charts": len(audio_charts),
               "audio_neighbors": len(audio_rows)}
    return values, rows, audio_rows, neighbors, summary


def _style_ceiling(neighbors, reference, label):
    for pool in (neighbors, reference["charts"]):
        bars = passed = 0
        for item in pool:
            for block in item["blocks"]:
                record = block["style"].get(label)
                if record:
                    bars += record[0]
                    passed += record[1]
        if bars >= 30:
            return passed / bars, bars
    return None, 0


def plan_adherence(chart, times, masks):
    """Planner intent versus the exported chart: the planned label's definition must hold in the bar."""
    if not chart.plan_bars:
        return None
    context = chart_context(chart.clock, times, masks)
    records = []
    for index, bar in enumerate(chart.plan_bars):
        start, end, style = float(bar["start_ms"]), float(bar["end_ms"]), bar.get("style")
        if end <= start or style not in LABELS or bar.get("quiet"):
            continue
        length = chart.clock.sections[chart.clock.section_at(start)].beat_length_ms
        if end - start < 4 * length - 1:
            continue
        rows = np.flatnonzero((times >= start - .5) & (times < end - .5))
        matches = bar_matches(context, rows)
        records.append({"bar": index + 1, "planned_style": style, "pass": bool(matches[style]),
                        "matched_styles": [label for label in LABELS if matches[label]]})
    per_style = {}
    for label in sorted({record["planned_style"] for record in records}):
        group = [record["pass"] for record in records if record["planned_style"] == label]
        per_style[label] = {"bars": len(group), "pass_rate": float(np.mean(group)), "rule": RULES[label]}
    return {"eligible_bars": len(records), "passed_bars": sum(record["pass"] for record in records),
            "bar_pass_rate": float(np.mean([record["pass"] for record in records])) if records else None,
            "per_style": per_style, "failed_bars": [record for record in records if not record["pass"]][:64],
            "policy": f"{DEFINITION_SCHEMA}: the planned label's definition must hold in the complete exported bar"}


def _style_windows(blocks, label, need, limit=3):
    """Consecutive-block windows holding at least ``need`` bars of one style."""
    counts = [block["style_bars"].get(label, 0) for block in blocks]
    windows = []
    for start in range(len(blocks)):
        total, stop = 0, start
        while stop < len(blocks) and total < need:
            total += counts[stop]
            stop += 1
        if total >= need and counts[start]:
            windows.append((start, stop))
    if len(windows) > limit:
        windows = [windows[int(round(position))] for position in np.linspace(0, len(windows) - 1, limit)]
    return [combine(blocks[start:stop])["style_hist"][label] for start, stop in windows]


def style_naturalness(profile, neighbors):
    """Compare each style's bars with human bars of the same style, in matched-size windows."""
    chart_bars = combine(profile["blocks"])["style_bars"]
    full = [combine(item["blocks"]) for item in neighbors]
    metrics, labels = {}, {}
    for label, bars in sorted(chart_bars.items(), key=lambda item: -item[1]):
        if bars < MINIMUM_STYLE_BARS:
            continue
        need = min(bars, STYLE_WINDOW_BARS)
        pooled = {name: sum((whole["style_hist"][label][name] for whole in full if label in whole["style_hist"]),
                            np.zeros(HISTOGRAMS[name], dtype=np.int64)) for name in STYLE_FEATURES}
        names = [f"style:{label}:{name}" for name in STYLE_FEATURES]
        chart_windows = _style_windows(profile["blocks"], label, need, limit=8)
        values = {key: float(np.median([js_divergence(window[name], pooled[name]) for window in chart_windows]))
                  for key, name in zip(names, STYLE_FEATURES)}
        rows = []
        for item, whole in zip(neighbors, full):
            if whole["style_bars"].get(label, 0) < need:
                continue
            excluded = {name: pooled[name] - whole["style_hist"][label][name] for name in STYLE_FEATURES}
            for window in _style_windows(item["blocks"], label, need):
                rows.append({key: js_divergence(window[name], excluded[name]) for key, name in zip(names, STYLE_FEATURES)})
        kinds = dict.fromkeys(names, "divergence")
        score, detail = human_likeness(values, rows, names, kinds)
        labels[label] = {"bars": bars, "window_bars": need, "score": score, **detail}
        for key, name in zip(names, STYLE_FEATURES):
            entry = {"value": values[key], "axis": "style_natural", "kind": "divergence",
                     "label": f"{label}·{STYLE_FEATURE_LABELS[name]}分布"}
            calibrated = typicality(values[key], [row[key] for row in rows], "divergence")
            if calibrated:
                entry.update(calibrated)
            metrics[key] = entry
    scored = [(item["score"], item["bars"]) for item in labels.values() if item["score"] is not None]
    score = round(sum(value * weight for value, weight in scored) / sum(weight for _, weight in scored), 1) if scored else None
    return score, labels, metrics


def validity(chart, times, masks):
    columns = (np.asarray(masks)[:, None] >> np.arange(4)) & 1
    impossible, minimum = 0, None
    for column in range(4):
        gaps = np.diff(times[np.flatnonzero(columns[:, column])])
        if len(gaps):
            impossible += int(np.sum(gaps < IMPOSSIBLE_JACK_MS))
            minimum = float(gaps.min()) if minimum is None else min(minimum, float(gaps.min()))
    row_gaps = np.diff(times)
    audit = None
    if chart.plan_bars and any(bar.get("style") == "Minijacks" for bar in chart.plan_bars):
        from ..playstyle_audit import audit_minijack_export
        audit = audit_minijack_export(chart.clock, chart.notes, chart.plan_bars)
        audit = {key: audit[key] for key in ("passed", "maximum_minijack_column_runs", "maximum_minijack_hand_runs")} | {"violations": len(audit["violations"])}
    issues = defaultdict(int)
    for issue in chart.issues:
        issues[issue] += 1
    return {"passed": impossible == 0 and (audit is None or audit["passed"]),
            "impossible_jacks_under_40ms": impossible, "minimum_same_column_gap_ms": minimum,
            "flam_rows_2_to_10ms": int(np.sum((row_gaps > ROW_MERGE_MS) & (row_gaps <= FLAM_MS))),
            "parse_issues": dict(issues), "minijack_audit": audit}


EXECUTION_FIELDS = ("music_notes", "prepared_notes", "layout_notes", "export_notes", "music_rows", "export_rows",
                    "prepare_removed_notes", "layout_removed_notes", "export_removed_notes",
                    "prepare_added_notes", "layout_added_notes", "export_added_notes",
                    "prepare_dropped_rows", "layout_dropped_rows", "export_dropped_rows", "notes_beat_mae", "rows_beat_mae")


def evaluate_chart(chart, reference=None, *, with_audio=True, neighbor_count=40):
    profile, (times, masks, holds) = profile_chart(chart, with_audio=with_audio)
    metrics = {}
    neighbors, summary, rows, audio_rows = [], None, [], []
    if reference is not None:
        values, rows, audio_rows, neighbors, summary = calibrate(profile, reference, neighbor_count)
    else:
        values = {**scalar_metrics(combine(profile["blocks"])), **(profile.get("audio") or {})}
    for name, info in METRIC_INFO.items():
        if name == "plan_style_realization":
            continue
        entry = {"value": values.get(name), **info}
        source = audio_rows if info["axis"] == "music" else rows
        calibrated = typicality(values.get(name), [row.get(name) for row in source], "two_sided" if info["kind"] == "info" else info["kind"])
        if calibrated:
            entry.update(calibrated)
        metrics[name] = entry
    style_score, style_labels = None, {}
    if neighbors:
        style_score, style_labels, style_metrics = style_naturalness(profile, neighbors)
        metrics.update(style_metrics)
    adherence = plan_adherence(chart, times, masks)
    realization = None
    if adherence and adherence["eligible_bars"] and reference is not None:
        weighted = total = 0.
        for label, record in adherence["per_style"].items():
            ceiling, bars = _style_ceiling(neighbors, reference, label)
            record["human_ceiling"] = ceiling
            record["human_ceiling_bars"] = bars
            if ceiling:
                weighted += record["bars"] * min(1., record["pass_rate"] / ceiling)
                total += record["bars"]
        if total:
            realization = weighted / total
            metrics["plan_style_realization"] = {"value": realization, **METRIC_INFO["plan_style_realization"],
                "status": "typical" if realization >= .9 else "unusual" if realization >= .7 else "atypical"}
    axes = {}
    for axis, label in AXES.items():
        names = [name for name, entry in metrics.items() if entry["axis"] == axis and entry["kind"] not in ("info", "direct")]
        if axis == "style":
            score, detail = (None if realization is None else round(100 * realization, 1)), {"basis": "plan_style_realization"}
        elif axis == "style_natural":
            score, detail = style_score, {"styles": style_labels}
        else:
            score, detail = human_likeness(values, audio_rows if axis == "music" else rows, names)
        concerns = sorted((name for name in metrics if metrics[name]["axis"] == axis and metrics[name]["kind"] != "info"
                           and metrics[name].get("status") not in (None, "typical")), key=lambda name: metrics[name].get("tail", metrics[name]["value"]))
        axes[axis] = {"label": label, "score": score, **detail, "concerns": concerns}
    natural_names = [name for name, entry in metrics.items() if entry["axis"] in NATURALNESS_AXES and entry["kind"] not in ("info", "direct")]
    whole_chart, naturalness_detail = human_likeness(values, rows, natural_names) if rows else (None, {})
    parts = [score for score in (whole_chart, style_score, axes["music"]["score"]) if score is not None]
    naturalness = round(float(np.mean(parts)), 1) if parts else None
    check = validity(chart, times, masks)
    execution = None
    if chart.execution:
        execution = {key: chart.execution.get(key) for key in EXECUTION_FIELDS if key in chart.execution}
        losses = sum(int(execution.get(key) or 0) for key in EXECUTION_FIELDS if "removed" in key or "dropped" in key)
        execution["downstream_losses"] = losses
    metadata = chart.metadata
    return {
        "version": EVALUATION_VERSION, "feature_schema": FEATURE_SCHEMA,
        "chart": {"name": chart.name, "sha256": chart.sha256, "title": metadata.get("Title"), "artist": metadata.get("Artist"),
                  "version": metadata.get("Version"), "creator": metadata.get("Creator"), "rows": profile["rows"],
                  "notes": profile["notes"], "hold_fraction": profile["hold_fraction"], "bpm": round(profile["bpm"], 3),
                  "span_seconds": round(profile["span_seconds"], 2), "blocks": len(profile["blocks"]),
                  "audio_error": profile.get("audio_error")},
        "difficulty": {"star": profile["star"], "target_star": chart.target_star,
                       "delta": profile["star"] - chart.target_star if profile["star"] is not None and chart.target_star is not None else None},
        "validity": check, "execution": execution, "plan_adherence": adherence,
        "metrics": metrics, "axes": axes,
        "overall": {"score": naturalness, "naturalness": naturalness, "whole_chart_naturalness": whole_chart,
                    "style_naturalness": style_score, "music_naturalness": axes["music"]["score"], **naturalness_detail,
                    "style_realization": axes["style"]["score"], "gate_passed": check["passed"],
                    "pattern_rules": DEFINITION_SCHEMA,
                    "definition": "Naturalness: mean of whole-chart human likeness (hand pressure, layout, rhythm against matched-length "
                                  "human segments), within-style human likeness (each style's bars against human bars of that style) and, "
                                  "when calibrated, music correspondence against the nearest human charts with audio. "
                                  "Style realization is reported beside it, never averaged in."},
        "reference": None if reference is None else {"corpus_sha256": reference["corpus_sha256"], "charts": len(reference["charts"]), "neighbors": summary},
        "boundary": BOUNDARY,
    }


def _format(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3g}" if abs(value) < 1000 else f"{value:.0f}"
    return str(value)


def _direction(entry):
    kind, percentile = entry["kind"], entry.get("percentile")
    if kind == "divergence":
        return "偏离人类"
    if kind == "direct":
        return "兑现不足"
    if kind == "low_bad" or (kind == "two_sided" and percentile is not None and percentile < .5):
        return "偏低"
    return "偏高"


STATUS_LABELS = {"typical": "典型", "unusual": "偏离", "atypical": "明显偏离"}


def render_markdown(report):
    chart, overall = report["chart"], report["overall"]
    title = " - ".join(item for item in (chart["artist"], chart["title"]) if item) or chart["name"]
    lines = [f"## {title} [{chart['version'] or chart['name']}]", ""]
    reference = report["reference"]
    if reference:
        neighbors = reference["neighbors"]
        stars = neighbors["star_range"]
        span = f"{stars[0]:.2f}–{stars[1]:.2f}★" if stars else "未知星级"
        realization = overall["style_realization"]
        lines.append(f"自然度 **{_format(overall['naturalness'])}** / 100 · 键型兑现 **{_format(realization)}** · "
                     f"硬约束 {'通过' if overall['gate_passed'] else '**未通过**'}")
        styles = report["axes"]["style_natural"].get("styles", {})
        within = " · ".join(f"{label} {_format(item['score'])}（{item['bars']} 小节）" for label, item in styles.items()) or "无足够同键型小节"
        lines.append(f"自然度 = 整谱 {_format(overall['whole_chart_naturalness'])}、键型内 {_format(overall['style_naturalness'])}"
                     f"、音乐对应 {_format(overall['music_naturalness'])} 中已校准部分的平均；键型内：{within}")
        lines.append(f"参照 {neighbors['count']} 张相近人类谱面（{span}），切成 {neighbors['human_segments']} 段 "
                     f"{neighbors['segment_blocks']}×16 小节的等长片段；键型内按同键型的人类小节比较。分数 = 至少与本谱同样不典型的人类样本占比，人类谱面约均匀分布在 0–100。")
    else:
        lines.append("未提供人类参照库：只列出原始指标，不计分。")
    difficulty = report["difficulty"]
    lines += ["", f"{chart['notes']} 音 / {chart['rows']} 行 · {chart['span_seconds']} s · {chart['bpm']} BPM · "
              f"实算 {_format(difficulty['star'])}★" + (f"（目标 {difficulty['target_star']}★）" if difficulty["target_star"] is not None else ""), ""]
    if reference:
        lines += ["| 维度 | 得分 | 主要偏离 |", "| --- | --- | --- |"]
        for axis in report["axes"].values():
            concerns = []
            for name in axis["concerns"][:2]:
                entry = report["metrics"][name]
                where = f"，人类 {entry['percentile'] * 100:.0f} 百分位" if entry.get("percentile") is not None else ""
                concerns.append(f"{entry['label']}{_direction(entry)}（{_format(entry['value'])}{where}）")
            score = _format(axis["score"]) if axis["score"] is not None else "未评"
            lines.append(f"| {axis['label']} | {score} | {'；'.join(concerns) or '—'} |")
    check = report["validity"]
    notes = [f"同列 <40 ms 连击 {check['impossible_jacks_under_40ms']}", f"2–10 ms 错位行 {check['flam_rows_2_to_10ms']}"]
    if check["minijack_audit"]:
        notes.append("短叠审计 " + ("通过" if check["minijack_audit"]["passed"] else "未通过"))
    lines += ["", "硬约束：" + " · ".join(notes)]
    adherence = report["plan_adherence"]
    if adherence:
        lines += ["", f"计划键型通过 {adherence['passed_bars']}/{adherence['eligible_bars']} 小节（{_format(adherence['bar_pass_rate'])}）"]
        for label, record in adherence["per_style"].items():
            ceiling = record.get("human_ceiling")
            lines.append(f"- {label}: {record['bars']} 小节，通过率 {_format(record['pass_rate'])}"
                         + (f"，人类上限 {_format(ceiling)}" if ceiling is not None else "") + f"（规则：{record['rule']}）")
    execution = report["execution"]
    if execution:
        lines += ["", f"计划/导出：音乐计划 {execution.get('music_notes')} 音 → 导出 {execution.get('export_notes')} 音 · "
                  f"下游减键/丢行 {execution['downstream_losses']} · 逐拍音符 MAE {_format(execution.get('notes_beat_mae'))}"]
    if reference:
        lines += ["", "| 指标 | 值 | 人类中位 [p5, p95] | 百分位 | 判断 |", "| --- | --- | --- | --- | --- |"]
        for name, entry in report["metrics"].items():
            if entry["value"] is None or (entry.get("reference") is None and entry["kind"] != "direct" and entry["axis"] != "music"):
                continue
            span = entry.get("reference")
            human = f"{_format(span['median'])} [{_format(span['p05'])}, {_format(span['p95'])}]" if span else "—"
            percentile = f"{entry['percentile'] * 100:.0f}" if entry.get("percentile") is not None else "—"
            status = "参考" if entry["kind"] == "info" else STATUS_LABELS.get(entry.get("status"), "未校准")
            lines.append(f"| {entry['label']} | {_format(entry['value'])} | {human} | {percentile} | {status} |")
    lines += ["", f"> {report['boundary']}"]
    return "\n".join(lines)


def render_comparison(reports):
    names = [report["chart"]["version"] or report["chart"]["name"] for report in reports]
    lines = ["## 对比", "", "| 项目 | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(reports)]
    lines.append("| 自然度 | " + " | ".join(_format(report["overall"]["naturalness"]) for report in reports) + " |")
    lines.append("| 　整谱 | " + " | ".join(_format(report["overall"]["whole_chart_naturalness"]) for report in reports) + " |")
    lines.append("| 　键型内 | " + " | ".join(_format(report["overall"]["style_naturalness"]) for report in reports) + " |")
    lines.append("| 　音乐对应 | " + " | ".join(_format(report["overall"]["music_naturalness"]) for report in reports) + " |")
    lines.append("| 键型兑现 | " + " | ".join(_format(report["overall"]["style_realization"]) for report in reports) + " |")
    lines.append("| 硬约束 | " + " | ".join("通过" if report["overall"]["gate_passed"] else "未通过" for report in reports) + " |")
    for axis, label in AXES.items():
        lines.append(f"| {label} | " + " | ".join(_format(report["axes"][axis]["score"]) for report in reports) + " |")
    lines.append("| 实算星级 | " + " | ".join(_format(report["difficulty"]["star"]) for report in reports) + " |")
    lines.append("| 计划键型通过率 | " + " | ".join(_format((report["plan_adherence"] or {}).get("bar_pass_rate")) for report in reports) + " |")
    for name, info in METRIC_INFO.items():
        cells = []
        for report in reports:
            entry = report["metrics"].get(name)
            if not entry or entry.get("value") is None:
                cells.append("—")
                continue
            mark = {"unusual": " △", "atypical": " ×"}.get(entry.get("status"), "")
            cells.append(_format(entry["value"]) + mark)
        if any(cell != "—" for cell in cells):
            lines.append(f"| {info['label']} | " + " | ".join(cells) + " |")
    lines += ["", "△ 偏离人类常见范围 · × 明显偏离。各谱面分别与各自相近的人类谱面比较。"]
    return "\n".join(lines)
