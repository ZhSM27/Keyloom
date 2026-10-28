"""Deterministic audio structure cues; no semantic labels or model training."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MusicStructurePolicy:
    minimum_bars: int = 2
    boundary_threshold: float = 0.30
    repetition_threshold: float = 0.85


def _standardized(values):
    values = np.asarray(values, dtype=np.float32)
    deviation = values.std(axis=0)
    # Keep nearly constant channels from dominating the structure signal.
    floor = max(float(deviation.mean()), float(np.abs(values).mean()) * 0.01, 1e-5)
    return np.clip((values - values.mean(axis=0)) / np.maximum(deviation, floor), -4, 4)


def _novelty(values, sections):
    result = np.zeros(len(values), dtype=np.float32)
    for position in range(1, len(values)):
        if sections[position] != sections[position - 1]:
            continue
        distances = []
        for width in (1, 2, 4, 8):
            lower, upper = max(0, position - width), min(len(values), position + width)
            if np.any(sections[lower:upper] != sections[position]):
                continue
            left, right = values[lower:position].mean(axis=0), values[position:upper].mean(axis=0)
            distances.append(min(1.0, float(np.sqrt(np.mean((left - right) ** 2))) / 2))
        if distances:
            result[position] = 0.5 * max(distances) + 0.5 * float(np.mean(distances))
    return result


def music_structure(notes, rows, probabilities, quiet, *, music=None, acoustic=None,
                    rhythm=None, timing_sections=None, owner_scores=None, policy=MusicStructurePolicy()):
    """Return shared zero-based half-open segments, respecting timing and silence."""
    count = len(notes)
    if (not isinstance(policy.minimum_bars, int) or policy.minimum_bars < 1
            or not 0 < policy.boundary_threshold <= 1 or not 0 < policy.repetition_threshold <= 1):
        raise ValueError("Invalid music structure policy.")
    sections = np.zeros(count, dtype=np.int64) if timing_sections is None else np.asarray(timing_sections)
    if sections.shape != (count,) or not np.issubdtype(sections.dtype, np.integer):
        raise ValueError("Timing section IDs must align with bars.")
    views = []
    for values in (music, acoustic, rhythm):
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != count or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError("Audio structure features must be finite aligned bar matrices.")
        views.append(_standardized(values))
    audio_score = np.maximum.reduce([_novelty(view, sections) for view in views]) if views else np.zeros(count, dtype=np.float32)
    evidence_score = np.zeros(count, dtype=np.float32)
    evidence_score[1:] = np.abs(np.diff(probabilities, axis=0)).max(axis=1)
    density = np.log1p(np.stack((notes, rows), axis=-1))
    density_score = np.zeros(count, dtype=np.float32)
    density_score[1:] = np.minimum(1, np.abs(np.diff(density, axis=0)).mean(axis=1))
    fallback = np.maximum(evidence_score, density_score)
    score = np.maximum(audio_score, (0.35 if views else 1.0) * fallback)
    learned_score = np.zeros(count, dtype=np.float32)
    if owner_scores is not None:
        owner_scores = np.asarray(owner_scores, dtype=np.float32)
        if owner_scores.shape != probabilities.shape or not np.isfinite(owner_scores).all():
            raise ValueError("Learned boundary evidence must align with bars.")
        for position in range(2, count - 1):
            left, right = owner_scores[position - 2:position], owner_scores[position:position + 2]
            left_id, right_id = left.argmax(axis=-1), right.argmax(axis=-1)
            if (left_id[0] == left_id[1] and right_id[0] == right_id[1] and left_id[0] != right_id[0]
                    and left.max(axis=-1).min() >= .55 and right.max(axis=-1).min() >= .55):
                learned_score[position] = .75 * np.abs(left.mean(axis=0) - right.mean(axis=0)).sum() / 2
        score = np.maximum(score, learned_score)
    forced = {0, count}
    for position in range(1, count):
        if quiet[position] != quiet[position - 1] or sections[position] != sections[position - 1]:
            forced.add(position)
            score[position] = 1
    boundaries = set(forced)
    for lower, upper in zip(sorted(forced)[:-1], sorted(forced)[1:]):
        if quiet[lower]:
            continue
        candidates = [position for position in range(lower + policy.minimum_bars, upper - policy.minimum_bars + 1)
                      if score[position] >= policy.boundary_threshold
                      and score[position] >= score[position - 1]
                      and (position + 1 == count or score[position] >= score[position + 1])]
        for position in sorted(candidates, key=lambda index: (-float(score[index]), index)):
            if all(abs(position - existing) >= policy.minimum_bars for existing in boundaries):
                boundaries.add(position)
    ids = np.full(count, -1, dtype=np.int32)
    repeats = np.full(count, -1, dtype=np.int32)
    segments, representatives = [], []
    ordered = sorted(boundaries)
    for lower, upper in zip(ordered[:-1], ordered[1:]):
        if quiet[lower]:
            continue
        ordinal = len(segments)
        ids[lower:upper] = ordinal
        parts = min(4, upper - lower)
        descriptor = None if not views else np.concatenate([
            np.concatenate([view[part].mean(axis=0) for part in np.array_split(np.arange(lower, upper), parts)])
            / np.sqrt(view.shape[1]) for view in views])
        group, similarity = len(representatives), 0.0
        if descriptor is not None:
            for candidate, (previous, length, section) in enumerate(representatives):
                if previous is None or section != sections[lower] or previous.shape != descriptor.shape or not 0.5 <= (upper - lower) / length <= 2:
                    continue
                affinity = float(np.exp(-np.sum((descriptor - previous) ** 2) / (len(views) * parts)))
                if affinity >= policy.repetition_threshold and affinity > similarity:
                    group, similarity = candidate, affinity
        if group == len(representatives):
            representatives.append((descriptor, upper - lower, sections[lower]))
        repeats[lower:upper] = group
        segments.append({"id": ordinal, "start_bar": lower, "end_bar": upper,
                         "timing_section": int(sections[lower]), "repeat_group": group,
                         "repeat_confidence": similarity, "boundary_confidence": float(score[lower]),
                         "boundary_source": "timing_or_silence" if lower in forced else "learned_owner" if learned_score[lower] > audio_score[lower] else "audio" if views and audio_score[lower] >= policy.boundary_threshold else "planner_fallback"})
    return {"segment_ids": ids, "repeat_group_ids": repeats, "boundary_scores": score,
            "segments": segments, "source": "multiscale_audio_with_planner_fallback" if views else "planner_evidence_fallback",
            "policy": asdict(policy)}
