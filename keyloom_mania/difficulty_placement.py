"""Where the exported chart's hardest strain sections fall relative to the musical sections.

Observational only: it never feeds back into planning or decoding. osu!mania
star rating sums per-400 ms strain peaks sorted in descending order with 0.9^i
weights, so the top ~20 peaks carry most of it. If those land in sections the
planner ranked below the peak, difficulty is misallocated, whatever the star.
"""
import math

import numpy as np

from .peak_planner import PeakPolicy, section_tiers

PLACEMENT_VERSION = 'difficulty-placement-v2'
# Bars planned in these styles stay out of every metric. Stream decoding has a
# known generator bug; remove the style here once the fix has been re-measured.
EXCLUDED_STYLES = ('Stream',)


def _median(values):
    return round(float(np.median(values)), 4) if values else None


def mania_strains(osu_path):
    """Per-section strain peaks and each section's start time in milliseconds."""
    import rosu_pp_py

    beatmap = rosu_pp_py.Beatmap(path=str(osu_path))
    strains = rosu_pp_py.Difficulty().strains(beatmap)
    return np.asarray(strains.strains, dtype=np.float64), float(strains.section_length)


def strain_section_starts(note_times_ms, count, section_length_ms):
    """Sections end at multiples of the length, starting at the second object (the first has no strain)."""
    times = np.sort(np.asarray(note_times_ms, dtype=np.float64))
    if len(times) < 2 or count <= 0:
        return np.zeros(0)
    first_end = math.ceil(times[1] / section_length_ms) * section_length_ms
    return first_end - section_length_ms + section_length_ms * np.arange(count)


def difficulty_placement(strains, section_length_ms, note_times_ms, bar_starts_ms, structure, planned_notes_per_beat,
                         shaped_notes_per_beat=None, *, top=20, policy=PeakPolicy(), bar_styles=None, excluded_styles=EXCLUDED_STYLES):
    """Strain placement by the same peak tiers the peak-anchored allocator uses.

    ``planned_notes_per_beat`` is the unshaped Planner density, so legacy,
    hierarchical and peak-anchored runs of one song are tiered identically.
    Tiers still use every section; only the reported metrics skip excluded bars.
    """
    strains = np.asarray(strains, dtype=np.float64)
    bar_starts = np.asarray(bar_starts_ms, dtype=np.float64)
    planned = np.asarray(planned_notes_per_beat, dtype=np.float64)
    quiet = np.asarray(structure['quiet'], dtype=bool)
    if strains.ndim != 1 or not np.isfinite(strains).all() or planned.shape != quiet.shape or bar_starts.shape != quiet.shape:
        raise ValueError('Difficulty placement needs finite strains and aligned bar arrays.')
    shaped = planned if shaped_notes_per_beat is None else np.asarray(shaped_notes_per_beat, dtype=np.float64)
    excluded = np.zeros(len(quiet), dtype=bool)
    if bar_styles is not None:
        if len(bar_styles) != len(quiet):
            raise ValueError('Difficulty placement bar styles must align with the bars.')
        excluded = np.asarray([style in excluded_styles for style in bar_styles], dtype=bool)
    sections = structure['sections']
    tiers = section_tiers(structure, planned, policy)
    tier = np.full(len(quiet), 'quiet', dtype=object)
    for index, section in enumerate(sections):
        tier[section['start_bar']:section['end_bar']] = 'peak' if index in tiers['peak'] else 'lower'
    times = np.sort(np.asarray(note_times_ms, dtype=np.float64))
    starts = strain_section_starts(times, len(strains), section_length_ms)
    # Strain sections before the first bar are counted in bar 0.
    strain_bar = np.clip(np.searchsorted(bar_starts, starts + section_length_ms / 2, side='right') - 1, 0, len(quiet) - 1)
    note_bar = np.clip(np.searchsorted(bar_starts, times, side='right') - 1, 0, len(quiet) - 1)
    exported = np.bincount(note_bar, minlength=len(quiet)).astype(np.float64) / 4
    kept_strains = ~excluded[strain_bar] if len(strains) else np.zeros(0, dtype=bool)
    top_index = [index for index in np.argsort(-strains, kind='stable') if kept_strains[index]][:top]
    top_tiers = [str(tier[strain_bar[index]]) for index in top_index]
    active = (tier != 'quiet') & ~excluded
    peak_bars = (tier == 'peak') & ~excluded
    weights = 0.9 ** np.arange(len(top_index))

    def high(values):
        return float(np.percentile(values, 90)) if len(values) else None

    peak_strain = high(strains[peak_bars[strain_bar]]) if len(strains) else None
    records = []
    for index, section in enumerate(sections):
        a, b = section['start_bar'], section['end_bar']
        keep = ~excluded[a:b]
        if not keep.any():
            records.append({'id': section['id'], 'start_bar': a, 'end_bar': b, 'tier': str(tier[a]), 'excluded': True,
                            'rank': round(float(tiers['rank'][index]), 4)})
            continue
        own = strains[(strain_bar >= a) & (strain_bar < b) & kept_strains]
        records.append({'id': section['id'], 'start_bar': a, 'end_bar': b, 'tier': str(tier[a]), 'excluded': False,
            'excluded_bars': int((~keep).sum()), 'rank': round(float(tiers['rank'][index]), 4),
            'planned_notes_per_beat': round(float(planned[a:b][keep].mean()), 4),
            'shaped_notes_per_beat': round(float(shaped[a:b][keep].mean()), 4),
            'exported_notes_per_beat': round(float(exported[a:b][keep].mean()), 4),
            'strain_p90': None if not len(own) else round(high(own), 4),
            'strain_max': None if not len(own) else round(float(own.max()), 4),
            'top_strain_sections': int(sum(strain_bar[i] >= a and strain_bar[i] < b for i in top_index))})
    exported_peak, planned_peak, shaped_peak = (float(values[peak_bars].mean()) if peak_bars.any() else 0.
                                                for values in (exported, planned, shaped))
    lower = [r for r in records if r['tier'] == 'lower' and not r['excluded']]
    return {'version': PLACEMENT_VERSION, 'strain_section_ms': section_length_ms, 'top_sections': int(len(top_index)),
        'excluded_styles': list(excluded_styles) if bar_styles is not None else [],
        'excluded_bars': int(excluded.sum()), 'excluded_strain_sections': int((~kept_strains).sum()),
        'alignment': 'rosu-pp strain sections end at multiples of the section length from the second hit object; bar mapping by section midpoint',
        'peak_bar_share': round(float(peak_bars.sum() / max(1, active.sum())), 4),
        'top_strain_peak_share': round(float(np.mean([t == 'peak' for t in top_tiers])), 4) if top_tiers else None,
        'top_strain_peak_weight_share': round(float(weights[[t == 'peak' for t in top_tiers]].sum() / weights.sum()), 4) if top_tiers else None,
        'top_strain_tiers': {name: top_tiers.count(name) for name in sorted(set(top_tiers))},
        'peak_strain_p90': None if peak_strain is None else round(peak_strain, 4),
        'lower_to_peak_strain_p90_median': _median([r['strain_p90'] / peak_strain for r in lower
                                                    if r['strain_p90'] is not None and peak_strain]),
        'lower_to_peak_exported_density_median': _median([r['exported_notes_per_beat'] / exported_peak
                                                          for r in lower if exported_peak > 0]),
        'lower_to_peak_planned_density_median': _median([r['planned_notes_per_beat'] / planned_peak
                                                         for r in lower if planned_peak > 0]),
        'lower_to_peak_shaped_density_median': _median([r['shaped_notes_per_beat'] / shaped_peak
                                                        for r in lower if shaped_peak > 0]),
        'sections': records,
        'top_strains': [{'start_ms': round(float(starts[i]), 1), 'strain': round(float(strains[i]), 4),
                         'bar': int(strain_bar[i]) + 1, 'tier': str(tier[strain_bar[i]])} for i in top_index],
        'boundary': 'Diagnostic only. Section tiers come from audio energy and Planner density, not verified verse/chorus labels.'}
