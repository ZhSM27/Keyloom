"""Audio-evidence section/phrase hierarchy and bounded musical density shaping.

No chorus/verse ground truth is implied. Learned Planner/adapter contexts stay
frozen; this module acts after inference and never edits generator onset logits.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

from .music_structure import _standardized

HIERARCHY_VERSION = 'section-phrase-hierarchy-v1'
STRUCTURE_OPTIONS = (
    {'id': 'legacy', 'label': '原音乐分段', 'status': 'baseline', 'available': True},
    {'id': 'hierarchical', 'label': '段落—乐句分层', 'status': 'research', 'available': True},
)


@dataclass(frozen=True)
class HierarchyPolicy:
    section_minimum_bars: int = 8
    phrase_minimum_bars: int = 2
    section_threshold: float = .32
    phrase_threshold: float = .28
    repetition_threshold: float = .85
    section_role_prior: float = .18
    section_density_strength: float = .08
    phrase_density_strength: float = .04
    maximum_density_change: float = .12


def validate_structure_id(value='legacy'):
    if not isinstance(value, str) or value not in ('legacy', 'hierarchical'):
        raise ValueError('请选择有效的段落与乐句规划方案。')
    return value


def _novelty(views, lower, upper, widths):
    score = np.zeros(upper - lower, dtype=np.float32)
    for position in range(lower + 1, upper):
        distances = []
        for view in views:
            for width in widths:
                left, right = max(lower, position - width), min(upper, position + width)
                if min(position - left, right - position) < min(width, 2):
                    continue
                a, b = view[left:position].mean(0), view[position:right].mean(0)
                distances.append(min(1., float(np.sqrt(np.mean((a - b) ** 2))) / 2))
        if distances:
            score[position - lower] = .5 * max(distances) + .5 * float(np.mean(distances))
    return score


def _boundaries(score, lower, upper, minimum, threshold):
    selected = {lower, upper}
    candidates = [position for position in range(lower + minimum, upper - minimum + 1)
        if score[position - lower] >= threshold
        and score[position - lower] >= score[position - lower - 1]
        and score[position - lower] >= score[min(len(score) - 1, position - lower + 1)]]
    for position in sorted(candidates, key=lambda i: (-float(score[i - lower]), i)):
        if all(abs(position - other) >= minimum for other in selected):
            selected.add(position)
    return selected


def _repeat_groups(items, views, threshold):
    """Order-sensitive normalized-time descriptors; different lengths can match."""
    representatives = []
    for item in items:
        indices = np.arange(item['start_bar'], item['end_bar'])
        descriptor = np.concatenate([np.concatenate([view[chunk].mean(0) for chunk in np.array_split(indices, min(4, len(indices)))])
            / np.sqrt(view.shape[1]) for view in views])
        group, confidence = len(representatives), 0.
        for candidate, (previous, length, timing) in enumerate(representatives):
            if timing != item['timing_section'] or previous.shape != descriptor.shape or not .5 <= len(indices) / length <= 2:
                continue
            affinity = float(np.exp(-np.sum((descriptor - previous) ** 2) / (len(views) * min(4, len(indices)))))
            if affinity >= threshold and affinity > confidence:
                group, confidence = candidate, affinity
        if group == len(representatives):
            representatives.append((descriptor, len(indices), item['timing_section']))
        item.update(repeat_group=group, repeat_confidence=confidence)


def _periodic_boundaries(views, lower, upper):
    """A bar-grid subdivision is allowed only with nonconstant repeated audio."""
    length = upper - lower
    for lag in (4, 8):
        if length < 3 * lag:
            continue
        qualities = []
        for view in views:
            values = view[lower:upper]
            variance = float(np.mean((values - values.mean(0)) ** 2))
            if variance <= .01:
                continue
            error = float(np.mean((values[lag:] - values[:-lag]) ** 2))
            qualities.append(1 - error / (2 * variance))
        if qualities and np.median(qualities) >= .85:
            return set(range(lower + lag, upper, lag)), float(np.median(qualities))
    return set(), 0.


def section_phrase_structure(quiet, *, music=None, acoustic=None, rhythm=None, timing_sections=None, policy=HierarchyPolicy()):
    quiet = np.asarray(quiet, dtype=bool)
    if quiet.ndim != 1 or not len(quiet):
        raise ValueError('Hierarchy requires a nonempty aligned quiet mask.')
    parameters = asdict(policy)
    for name in ('section_minimum_bars', 'phrase_minimum_bars'):
        if type(parameters[name]) is not int or parameters[name] < 1:
            raise ValueError('Hierarchy minimum lengths must be positive integers.')
    if (not all(np.isfinite(value) for value in parameters.values())
        or not 0 < policy.section_threshold <= 1 or not 0 < policy.phrase_threshold <= 1
        or not 0 < policy.repetition_threshold <= 1 or not 0 <= policy.maximum_density_change <= .25
        or min(policy.section_role_prior, policy.section_density_strength, policy.phrase_density_strength) < 0):
        raise ValueError('Invalid hierarchy policy.')
    count = len(quiet)
    timing = np.zeros(count, dtype=np.int64) if timing_sections is None else np.asarray(timing_sections)
    if timing.shape != (count,) or not np.issubdtype(timing.dtype, np.integer):
        raise ValueError('Timing section IDs must align with hierarchy bars.')
    raw, views = {}, []
    for name, value in (('music', music), ('acoustic', acoustic), ('rhythm', rhythm)):
        if value is None:
            continue
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 2 or value.shape[0] != count or not value.shape[1] or not np.isfinite(value).all():
            raise ValueError('Hierarchy audio views must be finite aligned bar matrices.')
        raw[name] = value
        views.append(_standardized(value))
    if not views:
        raise ValueError('Hierarchical planning requires audio evidence.')
    energy_source = 'unavailable'
    energy = np.zeros(count, dtype=np.float64)
    if 'rhythm' in raw and raw['rhythm'].shape[1] == 40:
        energy, energy_source = raw['rhythm'][:, 32:36].mean(-1).astype(np.float64), 'raw_rhythm_broadband_energy'
    elif 'acoustic' in raw:
        energy, energy_source = raw['acoustic'].mean(-1).astype(np.float64), 'mean_logmel_power'
    scale = float(energy[~quiet].std()) if (~quiet).any() else 0.
    energy_z = np.zeros(count) if scale < 1e-6 else np.clip((energy - energy[~quiet].mean()) / scale, -2, 2)
    forced = [0, *[i for i in range(1, count) if quiet[i] != quiet[i - 1] or timing[i] != timing[i - 1]], count]
    section_ids, phrase_ids = np.full(count, -1, dtype=np.int32), np.full(count, -1, dtype=np.int32)
    scores = np.zeros(count, dtype=np.float32)
    sections, phrases = [], []
    for lower, upper in zip(forced[:-1], forced[1:]):
        if quiet[lower]:
            continue
        coarse = _novelty(views, lower, upper, (4, 8, 16))
        edges = sorted(_boundaries(coarse, lower, upper, policy.section_minimum_bars, policy.section_threshold))
        for start, stop in zip(edges[:-1], edges[1:]):
            section_id = len(sections)
            section_ids[start:stop] = section_id
            sections.append({'id': section_id, 'start_bar': start, 'end_bar': stop,
                'timing_section': int(timing[start]), 'boundary_confidence': 1. if start in forced else float(coarse[start - lower]),
                'boundary_source': 'timing_or_silence' if start in forced else 'long_context_audio',
                'energy_level': float(energy_z[start:stop].mean()), 'phrase_ids': []})
            fine = _novelty(views, start, stop, (1, 2, 4))
            fine_edges = _boundaries(fine, start, stop, policy.phrase_minimum_bars, policy.phrase_threshold)
            periodic, periodic_confidence = _periodic_boundaries(views, start, stop)
            # Supported repeated periods are retained as musical units; do not
            # subdivide them again just because individual timbres alternate.
            if periodic:
                fine_edges = {start, stop, *periodic}
            for a, b in zip(sorted(fine_edges)[:-1], sorted(fine_edges)[1:]):
                phrase_id = len(phrases)
                phrase_ids[a:b] = phrase_id
                sections[-1]['phrase_ids'].append(phrase_id)
                slope = float(energy_z[max(a, b - 2):b].mean() - energy_z[a:min(b, a + 2)].mean())
                cue = 'build' if slope > .3 else 'release' if slope < -.3 else 'hold' if scale > 1e-6 else 'neutral'
                confidence = 1. if a == start else periodic_confidence if periodic else float(fine[a - start])
                scores[a] = confidence
                phrases.append({'id': phrase_id, 'section_id': section_id, 'start_bar': a, 'end_bar': b,
                    'timing_section': int(timing[a]), 'boundary_confidence': confidence,
                    'boundary_source': 'parent_section' if a == start else 'audio_periodicity' if periodic else 'short_context_audio',
                    'function_cue': cue, 'function_confidence': min(1., abs(slope)) if cue in ('build', 'release') else 0.,
                    'energy_level': float(energy_z[a:b].mean())})
    _repeat_groups(sections, views, policy.repetition_threshold)
    _repeat_groups(phrases, views, policy.repetition_threshold)
    repeats = np.full(count, -1, dtype=np.int32)
    for phrase in phrases:
        repeats[phrase['start_bar']:phrase['end_bar']] = phrase['repeat_group']
    return {'version': HIERARCHY_VERSION, 'section_ids': section_ids, 'phrase_ids': phrase_ids,
        'segment_ids': phrase_ids.copy(), 'sections': sections, 'segments': phrases,
        'repeat_group_ids': repeats, 'boundary_scores': scores, 'quiet': quiet.copy(),
        'energy': energy, 'energy_z': energy_z, 'energy_source': energy_source,
        'source': 'audio_section_phrase_hierarchy', 'policy': parameters,
        'boundary': 'Audio-relative sections and phrase cues; not verified verse/chorus or cadences.'}


def validate_hierarchy(structure, quiet):
    """Reject overlapping, crossing or silent spans before role scheduling."""
    quiet = np.asarray(quiet, dtype=bool)
    count = len(quiet)
    if not np.array_equal(structure.get('quiet'), quiet):
        raise ValueError('Hierarchy must preserve frozen silence.')
    for name, records in (('section_ids', structure['sections']), ('phrase_ids', structure['segments'])):
        ids = np.asarray(structure[name])
        expected = np.full(count, -1, dtype=np.int32)
        for ordinal, item in enumerate(records):
            a, b = item['start_bar'], item['end_bar']
            if (type(a) is not int or type(b) is not int or type(item['id']) is not int
                or item['id'] != ordinal or not 0 <= a < b <= count or (expected[a:b] != -1).any() or quiet[a:b].any()):
                raise ValueError('Invalid hierarchy coverage or silence overlap.')
            expected[a:b] = ordinal
            if name == 'phrase_ids':
                if type(item['section_id']) is not int or not 0 <= item['section_id'] < len(structure['sections']):
                    raise ValueError('Invalid parent section ID.')
                parent = structure['sections'][item['section_id']]
                if not parent['start_bar'] <= a < b <= parent['end_bar'] or parent['timing_section'] != item['timing_section']:
                    raise ValueError('A phrase crosses its parent section.')
        if ids.shape != (count,) or not np.issubdtype(ids.dtype, np.integer) or not np.array_equal(ids, expected) or not np.array_equal(ids >= 0, ~quiet):
            raise ValueError('Hierarchy must cover each active bar exactly once.')
    if not np.array_equal(structure['segment_ids'], structure['phrase_ids']):
        raise ValueError('Role scheduling segments must be hierarchy phrases.')
    for parent in structure['sections']:
        children = [p['id'] for p in structure['segments'] if p['section_id'] == parent['id']]
        if parent['phrase_ids'] != children:
            raise ValueError('Section children must match phrase parent references.')
    return deepcopy(structure)


def shape_hierarchical_density(beat_density, structure):
    density = np.asarray(beat_density)
    quiet = np.asarray(structure['quiet'], dtype=bool)
    validate_hierarchy(structure, quiet)
    if density.shape != (len(quiet), 4, 2) or not np.isfinite(density).all() or (density < 0).any():
        raise ValueError('Hierarchy density must be a finite aligned [bars, 4, 2] matrix.')
    policy = structure['policy']
    desired = np.ones(len(quiet), dtype=np.float64)
    for section in structure['sections']:
        a, b = section['start_bar'], section['end_bar']
        desired[a:b] += policy['section_density_strength'] * np.clip(section['energy_level'], -1, 1)
    for phrase in structure['segments']:
        a, b = phrase['start_bar'], phrase['end_bar']
        local = structure['energy_z'][a:b]
        desired[a:b] += policy['phrase_density_strength'] * np.clip(local - local.mean(), -1, 1)
    weights = density[..., 0].sum(-1).astype(np.float64)
    active = (~quiet) & (weights > 0)
    bound = policy['maximum_density_change']
    factors = np.ones(len(quiet), dtype=np.float64)
    if active.any():
        lower, upper = 1 - bound, np.full(len(quiet), 1 + bound)
        # Do not increase conditions beyond physical grid capacity; do not
        # pretend an already-infeasible original target became feasible either.
        peak = np.maximum(density[..., 0].max(1) / 96, density[..., 1].max(1) / 24)
        upper = np.minimum(upper, np.maximum(1., 1 / np.maximum(peak, 1e-8)))
        left, right = .5, 1.5
        for _ in range(50):
            middle = (left + right) / 2
            total = (np.clip(desired * middle, lower, upper)[active] * weights[active]).sum()
            if total < weights[active].sum(): left = middle
            else: right = middle
        factors[active] = np.clip(desired * ((left + right) / 2), lower, upper)[active]
        factors[np.abs(factors - 1) < 1e-12] = 1
    shaped = (density * factors[:, None, None]).astype(np.result_type(density.dtype, np.float32))
    # Quiet and zero budgets remain byte-for-byte as supplied.
    shaped[~active] = density[~active]
    return shaped, {'version': HIERARCHY_VERSION, 'multipliers': factors.tolist(),
        'original_note_budget_sum': float(density[..., 0].sum(dtype=np.float64)),
        'shaped_note_budget_sum': float(shaped[..., 0].sum(dtype=np.float64)),
        'original_row_budget_sum': float(density[..., 1].sum(dtype=np.float64)),
        'shaped_row_budget_sum': float(shaped[..., 1].sum(dtype=np.float64)),
        'note_budget_policy': 'preserve_total_requested_notes_across_complete_nominal_beats',
        'row_budget_policy': 'shared_multipliers_preserve_local_note_to_row_ratio; total_rows_may_change',
        'maximum_relative_change': bound, 'changed_bars': int(np.count_nonzero(np.abs(factors - 1) > 1e-6)),
        'original_beat_density': density.tolist(), 'shaped_beat_density': shaped.tolist()}


def json_hierarchy(structure):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in structure.items()}
