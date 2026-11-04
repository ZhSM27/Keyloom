"""Peak-anchored section density allocation on the audio section hierarchy.

osu!mania star rating is dominated by a chart's hardest stretch, not its mean.
The base Planner receives one song-wide star value for every bar and tends to
give verses the same density as the climax. This research option keeps the
most intense sections at the Planner's own density and only lowers the other
sections whose density exceeds the human peak-relative ratio for the target
star. It never raises a bar, never edits quiet or zero budgets, and does not
preserve the total budget: lower sections lose notes by design.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import json
from importlib import resources

import numpy as np

from .hierarchical_planner import STRUCTURE_OPTIONS as HIERARCHY_OPTIONS, validate_hierarchy

PEAK_VERSION = 'peak-anchored-density-v2'
CALIBRATION_SCHEMA = 'peak-allocation-calibration-v2'
LOCAL_STAR_VERSION = 'section-local-star-v1'
CALIBRATION_RESOURCE = 'peak-allocation.json'
# Bar-weighted section ranks below the peak fraction (0 = most intense).
RANK_GRID = (.35, .45, .55, .65, .75, .85, .95)
STRUCTURE_OPTIONS = (*HIERARCHY_OPTIONS,
    {'id': 'peak-anchored', 'label': '峰值锚定难度分配', 'status': 'research', 'available': True})
STAR_CONDITION_OPTIONS = (
    {'id': 'global', 'label': '全曲统一难度条件', 'status': 'baseline', 'available': True},
    {'id': 'section', 'label': '按段落强度的局部难度条件', 'status': 'research', 'available': True,
     'requires_structure_id': 'peak-anchored'},
)


@dataclass(frozen=True)
class PeakPolicy:
    peak_fraction: float = .3
    energy_weight: float = 1.
    density_weight: float = .75
    repeat_weight: float = .25
    density_spread_floor: float = .1
    minimum_multiplier: float = .6


def validate_structure_id(value='legacy'):
    if not isinstance(value, str) or value not in {item['id'] for item in STRUCTURE_OPTIONS}:
        raise ValueError('请选择有效的段落与乐句规划方案。')
    return value


def validate_star_condition(value='global', structure_id=None):
    if not isinstance(value, str) or value not in {item['id'] for item in STAR_CONDITION_OPTIONS}:
        raise ValueError('请选择有效的生成器难度条件方案。')
    if value == 'section' and structure_id is not None and structure_id != 'peak-anchored':
        raise ValueError('按段落强度的局部难度条件需要“峰值锚定难度分配”段落规划。')
    return value


def _chart_curve(blocks, peak_fraction, minimum_active_bars):
    """Density-sorted block ratios to the chart's own peak-fraction mean."""
    values = [(block['scalars']['notes'] / block['scalars']['active_bars'], block['scalars']['active_bars'])
              for block in blocks if block['scalars']['active_bars'] >= minimum_active_bars]
    if len(values) < 4:
        return None
    density, weight = np.asarray(sorted(values, key=lambda value: -value[0]), dtype=np.float64).T
    share = weight / weight.sum()
    rank = np.cumsum(share) - share / 2
    head = rank < peak_fraction
    head[0] = True
    peak = float(np.average(density[head], weights=weight[head]))
    if peak <= 0:
        return None
    return np.interp(RANK_GRID, rank, density / peak), peak


def calibrate(profile, *, peak_fraction=PeakPolicy.peak_fraction, minimum_active_bars=12, minimum_charts=30):
    """Star-bucketed human section-ratio curves from an evaluation reference profile."""
    curves, peaks = [], []
    for chart in profile['charts']:
        star = chart.get('star')
        if star is None or not np.isfinite(star):
            continue
        result = _chart_curve(chart['blocks'], peak_fraction, minimum_active_bars)
        if result is not None:
            curves.append((float(star), result[0]))
            bpm = chart.get('bpm')
            if bpm is not None and np.isfinite(bpm) and bpm > 0:
                # Notes per active 4/4 bar to notes per second at the dominant tempo.
                peaks.append((float(star), np.log(result[1] * bpm / 240)))
    if not curves:
        raise ValueError('The reference profile has no charts usable for peak calibration.')
    stars = np.asarray([star for star, _ in curves])
    values = np.asarray([curve for _, curve in curves])
    buckets = []
    for lower in range(2, 9):
        selected = (stars >= lower) & ((stars < lower + 1) if lower < 8 else True)
        if selected.sum() < minimum_charts:
            continue
        chosen = values[selected]
        buckets.append({'star': round(float(np.median(stars[selected])), 4), 'star_range': [lower, lower + 1 if lower < 8 else None],
                        'charts': int(selected.sum()),
                        **{name: np.round(np.percentile(chosen, q, axis=0), 4).tolist()
                           for name, q in (('ratio_q25', 25), ('ratio_median', 50), ('ratio_q75', 75))}})
    if not buckets:
        raise ValueError('Too few reference charts in every star bucket.')
    if len(peaks) < minimum_charts:
        raise ValueError('Too few reference charts with tempo for the star-density relation.')
    target, log_nps = np.asarray(peaks).T
    slope, intercept = np.polyfit(log_nps, target, 1)
    fitted = intercept + slope * log_nps
    return {'schema': CALIBRATION_SCHEMA, 'version': PEAK_VERSION,
            'source_schema': profile['schema'], 'source_feature_schema': profile['feature_schema'],
            'source_corpus_sha256': profile['corpus_sha256'], 'source_splits': profile.get('filters', {}).get('manifest_splits'),
            'block_bars': 16, 'minimum_active_bars': minimum_active_bars, 'peak_fraction': peak_fraction,
            'rank_grid': list(RANK_GRID), 'charts': len(curves), 'buckets': buckets,
            'star_from_peak_nps': {'intercept': round(float(intercept), 4), 'slope': round(float(slope), 4),
                'r2': round(float(1 - np.sum((target - fitted) ** 2) / np.sum((target - target.mean()) ** 2)), 4),
                'mae': round(float(np.abs(target - fitted).mean()), 4), 'charts': len(peaks),
                'definition': 'star ~ intercept + slope * ln(top-fraction block notes per second)'},
            'boundary': 'Human 16-bar blocks sorted by notes per active bar, as a ratio to the chart\'s own top-fraction '
                        'density. A section-contrast prior from charts of unknown curation, not a quality label.'}


def load_calibration():
    calibration = json.loads(resources.files('keyloom_mania.resources').joinpath(CALIBRATION_RESOURCE).read_text(encoding='utf-8'))
    validate_calibration(calibration)
    return calibration


def validate_calibration(calibration, policy=PeakPolicy()):
    if (calibration.get('schema') != CALIBRATION_SCHEMA or calibration.get('version') != PEAK_VERSION
            or calibration.get('rank_grid') != list(RANK_GRID) or calibration.get('peak_fraction') != policy.peak_fraction
            or not calibration.get('buckets')):
        raise ValueError('Peak allocation calibration does not match this allocator version.')
    stars = [bucket['star'] for bucket in calibration['buckets']]
    for bucket in calibration['buckets']:
        ratio = np.asarray(bucket['ratio_median'], dtype=np.float64)
        if ratio.shape != (len(RANK_GRID),) or not np.isfinite(ratio).all() or (ratio <= 0).any() or (ratio > 1).any():
            raise ValueError('Peak allocation ratios must be finite values in (0, 1].')
    if stars != sorted(stars):
        raise ValueError('Peak allocation buckets must be ordered by star.')
    relation = calibration.get('star_from_peak_nps') or {}
    if not all(np.isfinite(relation.get(name, np.nan)) for name in ('intercept', 'slope')) or relation['slope'] <= 0:
        raise ValueError('Peak allocation calibration needs a positive star-density slope.')
    return calibration


def ratio_curve(calibration, target_star):
    """Median human ratio at each rank, interpolated between star buckets."""
    stars = np.asarray([bucket['star'] for bucket in calibration['buckets']])
    ratios = np.asarray([bucket['ratio_median'] for bucket in calibration['buckets']])
    return np.asarray([np.interp(float(target_star), stars, ratios[:, column]) for column in range(len(RANK_GRID))])


def section_intensity(structure, notes_per_beat, policy=PeakPolicy()):
    """Song-relative section scores from audio energy, Planner density and repetition."""
    sections = structure['sections']
    lengths = np.asarray([s['end_bar'] - s['start_bar'] for s in sections], dtype=np.float64)
    energy = np.clip([s['energy_level'] for s in sections], -2, 2)
    level = np.log1p([notes_per_beat[s['start_bar']:s['end_bar']].mean() for s in sections])
    mean = np.average(level, weights=lengths)
    spread = max(float(np.sqrt(np.average((level - mean) ** 2, weights=lengths))), policy.density_spread_floor)
    density = np.clip((level - mean) / spread, -2, 2)
    # Choruses usually return; a single repetition already counts, more add little.
    groups = Counter(s['repeat_group'] for s in sections)
    repeat = np.asarray([min(groups[s['repeat_group']] - 1, 2) / 2 for s in sections])
    score = policy.energy_weight * energy + policy.density_weight * density + policy.repeat_weight * repeat
    order = sorted(range(len(sections)), key=lambda index: (-score[index], sections[index]['start_bar']))
    rank = np.zeros(len(sections))
    covered = 0.
    for index in order:
        rank[index] = (covered + lengths[index] / 2) / lengths.sum()
        covered += lengths[index]
    return {'energy': energy, 'density': density, 'repeat': repeat, 'score': score, 'rank': rank, 'order': order}


def section_tiers(structure, notes_per_beat, policy=PeakPolicy()):
    """Peak sections and repeat-consistent ranks; shared by allocation and diagnostics."""
    sections = structure['sections']
    if not sections:
        return {'peak': set(), 'rank': np.zeros(0), 'intensity': None}
    intensity = section_intensity(structure, notes_per_beat, policy)
    groups = {}
    for index, section in enumerate(sections):
        groups.setdefault(section['repeat_group'], []).append(index)
    # Most intense first until the fraction is covered: a long section crossing
    # the line must count, or a few short bars would become the only anchor.
    peak, covered = set(), 0
    target = policy.peak_fraction * sum(s['end_bar'] - s['start_bar'] for s in sections)
    for index in intensity['order']:
        peak.add(index)
        covered += sections[index]['end_bar'] - sections[index]['start_bar']
        if covered >= target:
            break
    # A returning section keeps the treatment of its strongest occurrence,
    # and repeated lower sections share one rank instead of a tie order.
    peak = {member for index in peak for member in groups[sections[index]['repeat_group']]}
    rank = np.asarray([np.mean(intensity['rank'][groups[section['repeat_group']]]) for section in sections])
    return {'peak': peak, 'rank': rank, 'intensity': intensity}


def shape_peak_anchored_density(beat_density, structure, target_star, policy=PeakPolicy(), calibration=None):
    density = np.asarray(beat_density)
    quiet = np.asarray(structure['quiet'], dtype=bool)
    validate_hierarchy(structure, quiet)
    if density.shape != (len(quiet), 4, 2) or not np.isfinite(density).all() or (density < 0).any():
        raise ValueError('Peak allocation density must be a finite aligned [bars, 4, 2] matrix.')
    parameters = asdict(policy)
    if (not all(np.isfinite(value) for value in parameters.values()) or not 0 < policy.peak_fraction < RANK_GRID[0]
            or not 0 < policy.minimum_multiplier <= 1 or policy.density_spread_floor <= 0
            or min(policy.energy_weight, policy.density_weight, policy.repeat_weight) < 0):
        raise ValueError('Invalid peak allocation policy.')
    if not np.isfinite(target_star):
        raise ValueError('Peak allocation needs a finite target star.')
    calibration = validate_calibration(load_calibration() if calibration is None else calibration, policy)
    curve = ratio_curve(calibration, target_star)
    notes = density[..., 0].mean(1).astype(np.float64)
    active = (~quiet) & (density[..., 0].sum(-1) > 0)
    factors = np.ones(len(quiet), dtype=np.float64)
    records, reference = [], None
    sections = structure['sections']
    if sections:
        tiers = section_tiers(structure, notes, policy)
        intensity, peak, rank = tiers['intensity'], tiers['peak'], tiers['rank']
        peak_bars = np.zeros(len(quiet), dtype=bool)
        for index in peak:
            peak_bars[sections[index]['start_bar']:sections[index]['end_bar']] = True
        if (peak_bars & active).any():
            reference = float(notes[peak_bars & active].mean())
        for index, section in enumerate(sections):
            a, b = section['start_bar'], section['end_bar']
            local = active[a:b]
            before = float(notes[a:b][local].mean()) if local.any() else 0.
            ratio = cap = None
            multiplier = 1.
            if index not in peak and reference and before > 0:
                ratio = float(np.interp(rank[index], RANK_GRID, curve))
                cap = ratio * reference
                multiplier = float(np.clip(cap / before, policy.minimum_multiplier, 1))
                if multiplier > 1 - 1e-9:
                    multiplier = 1.
                factors[a:b][local] = multiplier
            records.append({'id': section['id'], 'start_bar': a, 'end_bar': b,
                'tier': 'peak' if index in peak else 'reduced' if multiplier < 1 else 'within_ratio',
                'rank': round(float(rank[index]), 4), 'own_rank': round(float(intensity['rank'][index]), 4), 'score': round(float(intensity['score'][index]), 4),
                'energy_level': round(float(intensity['energy'][index]), 4), 'density_z': round(float(intensity['density'][index]), 4),
                'repeat': round(float(intensity['repeat'][index]), 4),
                'human_ratio': None if ratio is None else round(ratio, 4),
                'cap_notes_per_beat': None if cap is None else round(cap, 4),
                'notes_per_beat_before': round(before, 4), 'multiplier': round(multiplier, 4),
                'notes_per_beat_after': round(before * multiplier, 4)})
    shaped = (density * factors[:, None, None]).astype(np.result_type(density.dtype, np.float32))
    # Quiet and zero budgets remain byte-for-byte as supplied.
    shaped[~active] = density[~active]
    original_notes = float(density[..., 0].sum(dtype=np.float64))
    shaped_notes = float(shaped[..., 0].sum(dtype=np.float64))
    return shaped, {'version': PEAK_VERSION, 'hierarchy_version': structure['version'], 'target_star': float(target_star),
        'calibration': {key: calibration[key] for key in ('schema', 'source_schema', 'source_feature_schema', 'source_corpus_sha256', 'charts')},
        'policy': parameters, 'ratio_curve': {'rank_grid': list(RANK_GRID), 'ratio': np.round(curve, 4).tolist()},
        'peak_reference_notes_per_beat': None if reference is None else round(reference, 4),
        'sections': records,
        'original_note_budget_sum': original_notes, 'shaped_note_budget_sum': shaped_notes,
        'original_row_budget_sum': float(density[..., 1].sum(dtype=np.float64)),
        'shaped_row_budget_sum': float(shaped[..., 1].sum(dtype=np.float64)),
        'note_budget_change': 0. if original_notes <= 0 else shaped_notes / original_notes - 1,
        'note_budget_policy': 'peak_sections_unchanged; other sections capped at the human peak-relative ratio; never raised; total may decrease',
        'row_budget_policy': 'shared_multipliers_preserve_local_note_to_row_ratio',
        'changed_bars': int(np.count_nonzero(np.abs(factors - 1) > 1e-6)),
        'multipliers': factors.tolist(), 'original_beat_density': density.tolist(), 'shaped_beat_density': shaped.tolist()}


def section_local_stars(shaping, bar_count, target_star, calibration=None, *, maximum_drop=1.5, minimum_star=2.):
    """Per-bar generator star condition from the shaped section densities.

    Peak sections keep the target star. Every other section is given the star a
    human chart would have if that section were its peak, using the calibrated
    star-density slope; the drop is bounded and never goes below the workbench
    range. Quiet bars keep the target star (they emit no notes).
    """
    calibration = validate_calibration(load_calibration() if calibration is None else calibration)
    if shaping.get('version') != PEAK_VERSION:
        raise ValueError('Local star conditions require a peak-anchored density report.')
    if not np.isfinite(target_star) or not 0 <= maximum_drop <= 3:
        raise ValueError('Invalid local star policy.')
    slope = float(calibration['star_from_peak_nps']['slope'])
    reference = shaping.get('peak_reference_notes_per_beat')
    stars = np.full(bar_count, float(target_star))
    records = []
    for section in shaping['sections']:
        drop = 0.
        after = section['notes_per_beat_after']
        if section['tier'] != 'peak' and reference and after > 0:
            drop = float(np.clip(-slope * np.log(min(1., after / reference)), 0, maximum_drop))
        star = max(float(minimum_star), float(target_star) - drop) if drop else float(target_star)
        stars[section['start_bar']:section['end_bar']] = star
        records.append({'id': section['id'], 'tier': section['tier'], 'local_star': round(star, 4)})
    return stars, {'version': LOCAL_STAR_VERSION, 'density_version': shaping['version'], 'target_star': float(target_star),
        'star_density_slope': slope, 'maximum_drop': maximum_drop, 'minimum_star': minimum_star, 'sections': records,
        'boundary': "The generator was trained with each chart's global star; a lowered window condition means "
                    '"like a chart whose peak is this dense", not a verified local difficulty.'}


def window_star(local_stars, beats):
    """Mean local star over the candidate slots a window will newly decide.

    ``beats`` are absolute beats from the first bar line; prefix slots that the
    previous window already decided must be excluded by the caller.
    """
    beats = np.asarray(beats, dtype=np.float64)
    if not len(beats) or not np.isfinite(beats).all():
        raise ValueError('A window star needs finite candidate beats.')
    bars = np.clip(np.floor(beats / 4).astype(np.int64), 0, len(local_stars) - 1)
    return float(np.mean(np.asarray(local_stars)[bars]))
