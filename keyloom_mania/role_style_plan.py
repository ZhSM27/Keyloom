from dataclasses import asdict, dataclass

import numpy as np

from .main_style import one_hot_plan
from .music_structure import MusicStructurePolicy, music_structure
from .plan_compiler import compile_plan_v2
from .segment_labels import LABELS
from .style_hierarchy import LOWER_STYLES, compatible_styles


ROLE_STYLE_VERSION = "role-style-plan-v4-directional-lower-styles"


@dataclass(frozen=True)
class RoleStylePolicy:
    minimum_segment_bars: int = 2
    boundary_threshold: float = 0.30
    repetition_threshold: float = 0.85
    primary_bias: float = 0.16
    secondary_bias: float = 0.20
    minimum_secondary_evidence: float = 0.12
    switch_penalty: float = 0.18
    primary_target_fraction: float = 0.55
    primary_share_penalty: float = 2.0
    repetition_weight: float = 0.20
    maximum_lower_fraction: float = 0.25
    lower_minimum_evidence: float = 0.22
    energy_density_weight: float = 0.0


def validate_style_selection(primary_label, secondary_labels=None):
    if primary_label not in LABELS:
        raise ValueError("Select a supported primary osu!mania style.")
    if secondary_labels is None or secondary_labels == "":
        secondary_labels = ()
    elif isinstance(secondary_labels, str):
        secondary_labels = (secondary_labels,)
    elif isinstance(secondary_labels, (list, tuple)):
        secondary_labels = tuple(secondary_labels)
    else:
        raise ValueError("Secondary styles must be a list of up to three supported 4K styles.")
    if any(not isinstance(label, str) or label not in LABELS for label in secondary_labels):
        raise ValueError("Select supported secondary osu!mania styles.")
    if len(secondary_labels) > 3 or len(set(secondary_labels)) != len(secondary_labels):
        raise ValueError("Choose up to three different secondary styles.")
    if primary_label in secondary_labels:
        raise ValueError("Primary and secondary styles must be different.")
    return primary_label, secondary_labels


def _schedule(emissions, lengths, boundaries, intervals, policy):
    """Exact DP over last style and secondary duration, with a soft share cost."""
    segments, styles = emissions.shape
    if not segments:
        return np.empty(0, dtype=np.int64)
    if styles == 1:
        return np.zeros(segments, dtype=np.int64)
    total = int(lengths.sum())
    scores = np.full((styles, total + 1), -np.inf)
    scores[0, 0] = 0
    back, used = [], 0
    for ordinal in range(segments):
        length = int(lengths[ordinal])
        next_scores = np.full_like(scores, -np.inf)
        previous_style = np.full(scores.shape, -1, dtype=np.int8)
        penalty = policy.switch_penalty * (1 - 0.75 * boundaries[ordinal])
        connected = ordinal > 0 and intervals[ordinal, 0] == intervals[ordinal - 1, 1]
        for selected in range(styles):
            if not np.isfinite(emissions[ordinal, selected]):
                continue
            delta = length if selected else 0
            alternatives = scores[:, :used + 1].copy()
            if connected:
                alternatives -= penalty * (np.arange(styles) != selected)[:, None]
            best = alternatives.argmax(axis=0)
            values = alternatives[best, np.arange(used + 1)] + emissions[ordinal, selected] * length
            next_scores[selected, delta:used + delta + 1] = values
            previous_style[selected, delta:used + delta + 1] = best
        scores = next_scores
        back.append(previous_style)
        used += length
    fraction = np.arange(total + 1) / max(1, total)
    excess = np.maximum(0, fraction - (1 - policy.primary_target_fraction))
    scores -= policy.primary_share_penalty * total * excess[None] ** 2
    selected, secondary_bars = np.unravel_index(scores.argmax(), scores.shape)
    result = np.zeros(segments, dtype=np.int64)
    for ordinal in range(segments - 1, -1, -1):
        result[ordinal] = selected
        previous = int(back[ordinal][selected, secondary_bars])
        secondary_bars -= int(lengths[ordinal]) if selected else 0
        selected = previous
    return result


def active_style_plan(probabilities, notes_per_beat, rows_per_beat, thresholds, config,
                      primary_label, secondary_labels=None, policy=RoleStylePolicy(), *,
                      music=None, acoustic=None, rhythm=None, timing_sections=None,
                      owner_scores=None, style_evidence=None, density_reference=None, structure_override=None):
    primary_label, secondary_labels = validate_style_selection(primary_label, secondary_labels)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    notes, rows = np.asarray(notes_per_beat, dtype=np.float32), np.asarray(rows_per_beat, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if notes.ndim != 1 or not len(notes) or rows.shape != notes.shape or probabilities.shape != (len(notes), len(LABELS)) or thresholds.shape != (len(LABELS),):
        raise ValueError("Role-style plans require aligned nonempty bar arrays and fifteen labels.")
    if not all(np.isfinite(values).all() for values in (probabilities, notes, rows, thresholds)) or np.any((probabilities < 0) | (probabilities > 1)) or np.any(notes < 0) or np.any(rows < 0):
        raise ValueError("Role-style probabilities and budgets must be finite and in range.")
    parameters = asdict(policy)
    if (not all(np.isfinite(value) for value in parameters.values())
            or any(parameters[name] < 0 for name in ("primary_bias", "secondary_bias", "switch_penalty", "primary_share_penalty", "repetition_weight", "energy_density_weight"))
            or not 0 <= policy.minimum_secondary_evidence <= 1 or not 0 < policy.primary_target_fraction <= 1
            or not 0 <= policy.maximum_lower_fraction < 0.5 or not 0 <= policy.lower_minimum_evidence <= 1):
        raise ValueError("Invalid role-style policy.")
    raw = compile_plan_v2(probabilities, notes, rows, thresholds, config)
    quiet = np.asarray(raw["quiet"], dtype=bool)
    owner_values = probabilities if owner_scores is None else np.asarray(owner_scores, dtype=np.float32)
    lower_values = probabilities if style_evidence is None else np.asarray(style_evidence, dtype=np.float32)
    for value in (owner_values, lower_values):
        if value.shape != probabilities.shape or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError("Learned owner/style scores must align with finite fifteen-label bars.")
    structural_notes = notes if owner_scores is None else np.ones(len(notes))
    structural_rows = rows if owner_scores is None else np.ones(len(rows))
    structural_evidence = probabilities if owner_scores is None else np.zeros_like(probabilities)
    structure = music_structure(structural_notes, structural_rows, structural_evidence, quiet, music=music, acoustic=acoustic,
                                rhythm=rhythm, timing_sections=timing_sections,
                                owner_scores=owner_scores,
                                policy=MusicStructurePolicy(policy.minimum_segment_bars, policy.boundary_threshold, policy.repetition_threshold))
    if structure_override is not None:
        from .hierarchical_planner import validate_hierarchy
        structure = validate_hierarchy(structure_override, quiet)
    allowed = [primary_label, *sorted(secondary_labels, key=LABELS.index)]
    columns = [LABELS.index(label) for label in allowed]
    segments = structure["segments"]
    evidence = np.asarray([owner_values[item["start_bar"]:item["end_bar"]].mean(axis=0)[columns] for item in segments], dtype=np.float32).reshape(-1, len(allowed))
    density_cost = np.zeros_like(probabilities)
    if policy.energy_density_weight:
        from .density_coordination import style_density_cost
        reference = np.stack((notes,rows),axis=-1) if density_reference is None else np.asarray(density_reference)
        if reference.shape != (len(notes),2) or not np.isfinite(reference).all() or (reference<0).any():
            raise ValueError('Coordinated density reference must be a finite aligned nonnegative bar matrix.')
        density_cost = policy.energy_density_weight * style_density_cost(reference[:,0], reference[:,1], quiet, rhythm)
    emissions = evidence.copy()
    if segments:
        emissions -= np.asarray([density_cost[item['start_bar']:item['end_bar']].mean(0)[columns] for item in segments])
    emissions[:, 0] += policy.primary_bias
    emissions[:, 1:] += policy.secondary_bias
    if segments:
        active_mean = max(float((notes[~quiet] + rows[~quiet]).mean()), 1e-6)
        levels = np.asarray([(notes[item["start_bar"]:item["end_bar"]] + rows[item["start_bar"]:item["end_bar"]]).mean() / active_mean for item in segments])
        # Density is a small preference, never a gate excluding strong sections.
        emissions[:, 1:] += (0.15 * np.clip(1 - levels, -1, 1))[:, None]
        eligible = evidence >= policy.minimum_secondary_evidence
        eligible[:, 0] = True
        for group in sorted({item["repeat_group"] for item in segments}):
            indices = [index for index, item in enumerate(segments) if item["repeat_group"] == group]
            if len(indices) > 1:
                mean = evidence[indices].mean(axis=0)
                emissions[indices] += policy.repetition_weight * (mean - evidence[indices])
        emissions[~eligible] = -np.inf
    if structure_override is not None and structure['sections']:
        # First schedule section intentions, then softly guide phrase choices.
        # Phrase-level support gates still apply, so a coarse prior cannot
        # introduce an unsupported secondary or an upper automatic style.
        parents = structure['sections']
        parent_evidence = np.asarray([owner_values[s['start_bar']:s['end_bar']].mean(0)[columns] for s in parents])
        parent_emissions = parent_evidence.copy()
        parent_emissions -= np.asarray([density_cost[s['start_bar']:s['end_bar']].mean(0)[columns] for s in parents])
        parent_emissions[:, 0] += policy.primary_bias
        parent_emissions[:, 1:] += policy.secondary_bias
        parent_allowed = parent_evidence >= policy.minimum_secondary_evidence
        parent_allowed[:, 0] = True
        parent_emissions[~parent_allowed] = -np.inf
        parent_selected = _schedule(parent_emissions,
            np.asarray([s['end_bar'] - s['start_bar'] for s in parents]),
            np.asarray([s['boundary_confidence'] for s in parents]),
            np.asarray([(s['start_bar'], s['end_bar']) for s in parents]), policy)
        for index, parent in enumerate(parents):
            choice = int(parent_selected[index])
            others = np.delete(parent_emissions[index], choice)
            margin = float(parent_emissions[index, choice] - np.max(others)) if len(others) else 1.
            confidence = float(np.clip(margin / .25, 0, 1))
            parent.update(style_intent=allowed[choice], role_intent='main' if choice == 0 else 'support',
                          role_confidence=confidence,
                          evidence={label: float(value) for label, value in zip(allowed, parent_evidence[index])})
            for ordinal, phrase in enumerate(segments):
                if phrase['section_id'] == index:
                    emissions[ordinal, choice] += structure['policy']['section_role_prior'] * confidence
    lengths = np.asarray([item["end_bar"] - item["start_bar"] for item in segments], dtype=np.int64)
    selected = _schedule(emissions, lengths, np.asarray([item["boundary_confidence"] for item in segments]),
                         np.asarray([(item["start_bar"], item["end_bar"]) for item in segments]), policy)
    style_ids = np.zeros(len(notes), dtype=np.int64)
    owner_ids = np.zeros(len(notes), dtype=np.int64)
    for ordinal, item in enumerate(segments):
        choice = int(selected[ordinal])
        item.update(style=allowed[choice], role="main" if choice == 0 else "support",
                    evidence={label: round(float(value), 6) for label, value in zip(allowed, evidence[ordinal])},
                    reason="primary_identity_and_global_schedule" if choice == 0 else "selected_secondary_evidence_and_global_schedule")
        style_ids[item["start_bar"]:item["end_bar"]] = columns[choice] + 1
        owner_ids[item["start_bar"]:item["end_bar"]] = columns[choice] + 1
        item["requested_style"] = allowed[choice]
        if policy.energy_density_weight:
            item['density_compatibility_cost'] = {label:round(float(density_cost[item['start_bar']:item['end_bar'],column].mean()),6)
                                                 for label,column in zip(allowed,columns)}
    # A lower variant keeps its owner's role and budget. The reverse direction
    # and unrelated families can never be introduced by this optional mixture.
    for owner in allowed:
        lower = LOWER_STYLES.get(owner, ())
        owner_column = LABELS.index(owner)
        limit = int(np.sum(owner_ids == owner_column + 1) * policy.maximum_lower_fraction)
        candidates = []
        for item in segments:
            if item["requested_style"] != owner or not lower:
                continue
            start, stop = item["start_bar"], item["end_bar"]
            local = (lower_values[start:stop] - density_cost[start:stop]).mean(axis=0)
            label = max(lower, key=lambda name: float(local[LABELS.index(name)]))
            strength = float(local[LABELS.index(label)])
            if strength >= max(policy.lower_minimum_evidence, float(local[owner_column]) * 0.9):
                candidates.append((strength - float(local[owner_column]), start, stop, label, item))
        used = 0
        for _, start, stop, label, item in sorted(candidates, key=lambda value: (-value[0], value[1])):
            if used + stop - start <= limit:
                style_ids[start:stop] = LABELS.index(label) + 1
                item.update(style=label, reason="same_family_lower_variant", lower_variant_of=owner)
                used += stop - start
    main = owner_ids == LABELS.index(primary_label) + 1
    secondary = (~quiet) & ~main
    roles = np.where(quiet, "quiet", np.where(main, "main", "support"))
    unused = []
    for label in secondary_labels:
        if not np.any(owner_ids == LABELS.index(label) + 1):
            supported = any(item["evidence"][label] >= policy.minimum_secondary_evidence for item in segments)
            reason = "used_as_lower_variant_not_independent_secondary" if np.any(style_ids == LABELS.index(label) + 1) else "global_identity_or_transition_cost" if supported else "insufficient_local_evidence"
            unused.append({"style": label, "reason": reason})
    permitted = sorted({name for owner in allowed for name in compatible_styles(owner)}, key=LABELS.index)
    hierarchy_fields = {} if structure_override is None else {
        'section_ids': structure['section_ids'], 'sections': structure['sections'],
        'hierarchy_version': structure['version'], 'hierarchy': structure}
    return {**raw, **hierarchy_fields, "patterns": one_hot_plan(style_ids).astype(np.uint8), "style_ids": style_ids, "owner_style_ids": owner_ids,
            "primary_style": primary_label, "secondary_styles": list(secondary_labels),
            "main_section": main.astype(np.uint8), "secondary_section": secondary.astype(np.uint8),
            "phrase_ids": structure["segment_ids"], "segment_ids": structure["segment_ids"],
            "repeat_group_ids": structure["repeat_group_ids"], "boundary_scores": structure["boundary_scores"],
            "segments": segments, "unused_secondary_styles": unused, "roles": roles,
            "structure_policy": structure["policy"],
            "style_policy": {"schema_version": ROLE_STYLE_VERSION, "maximum_secondary_styles": 3,
                             "automatic_variants": "same_family_lower_only", "requested_styles": allowed,
                             "allowed_styles": permitted, "lower_style_map": {name: list(LOWER_STYLES.get(name, ())) for name in allowed}, **parameters},
            "role_source": ("learned_conditional_owner_" if owner_scores is not None else "") + structure["source"] + "_not_semantic_section_labels"}
