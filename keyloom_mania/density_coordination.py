"""Bounded music-preserving density coordination; no hard allocation budget."""
import numpy as np

from .segment_labels import LABELS

COORDINATION_VERSION = 'energy-density-coordination-v2-preserve-onsets'
COORDINATION_OPTIONS = (
    {'id':'legacy', 'label':'原编排与密度执行', 'status':'baseline'},
    {'id':'coordinated', 'label':'能量与密度协调', 'status':'research'},
)


def validate_coordination(value='legacy'):
    if not isinstance(value, str) or value not in ('legacy', 'coordinated'):
        raise ValueError('请选择有效的编排与密度协调方案。')
    return value


def density_execution_trace(beat_density, bars, note_times, energy, beat_length_ms):
    """Count rounded integer-timestamp exports on half-open beat intervals."""
    times=np.sort(np.asarray(note_times,dtype=np.float64))
    starts=np.asarray([bar['start_ms'] for bar in bars])
    edges=np.rint(starts[:,None]+np.arange(5)*beat_length_ms)
    actual=np.stack([np.diff(np.searchsorted(values,edges,side='left'),axis=-1) for values in (times,np.unique(times))],axis=-1)
    target=np.asarray(beat_density,dtype=np.float64)
    complete=np.asarray([bar['end_ms']-bar['start_ms']>=4*beat_length_ms-1 for bar in bars])
    energy=np.asarray(energy)
    active=complete & ~np.asarray([bar['quiet'] for bar in bars])
    high=active & (energy>=np.quantile(energy[active],.75)) if active.any() else active
    low=active & (energy<=np.quantile(energy[active],.25)) if active.any() else active
    summary={'notes_beat_mae':float(np.abs(actual[complete,:,0]-target[complete,:,0]).mean()) if complete.any() else None,
             'rows_beat_mae':float(np.abs(actual[complete,:,1]-target[complete,:,1]).mean()) if complete.any() else None,
             'high_energy_underfill_fraction':float((actual[high,:,0].mean(-1)<.65*target[high,:,0].mean(-1)).mean()) if high.any() else None,
             'active_minus_low_energy_notes_per_beat':float(actual[high,:,0].mean()-actual[low,:,0].mean()) if high.any() and low.any() else None}
    trace=[{**bar,'audio_energy':float(energy[i]),'target_beat_density':target[i].tolist(),
            'actual_beat_density':actual[i].tolist(),'complete':bool(complete[i]),'high_energy':bool(high[i])} for i,bar in enumerate(bars)]
    return {'version':COORDINATION_VERSION,'summary':summary,'bars':trace,'edge_policy':'round theoretical beat edges to exported integer milliseconds, half-open intervals'}


def style_density_cost(notes, rows, quiet, rhythm=None):
    """Soft chord-ratio shortfall at the requested row rate, not a hard gate."""
    notes, rows = np.asarray(notes), np.asarray(rows)
    energy = notes + rows
    if rhythm is not None:
        rhythm = np.asarray(rhythm)
        if rhythm.ndim != 2 or rhythm.shape[0] != len(notes) or rhythm.shape[1] != 40 or not np.isfinite(rhythm).all():
            raise ValueError('Density coordination needs aligned raw 40-channel rhythm features.')
        energy = rhythm[:,32:36].mean(-1)
    active = ~np.asarray(quiet, dtype=bool)
    reference = np.sort(energy[active])
    if len(reference):
        rank = (np.searchsorted(reference,energy,'left') + np.searchsorted(reference,energy,'right')) / (2*len(reference))
    else:
        rank = np.zeros(len(notes))
    capacity = np.full(len(LABELS),4.,dtype=np.float32)
    for label in ('Stream','Rolls','Trills','Minitrills'): capacity[LABELS.index(label)] = 1
    for label in ('Jumpstream','Jumptrill','Split Trill'): capacity[LABELS.index(label)] = 2
    capacity[LABELS.index('Handstream')] = 3
    shortfall = np.maximum(0, 1 - rows[:,None]*capacity[None]/np.maximum(notes[:,None],1e-6))
    salience = np.clip((rank-.25)/.5,0,1)
    return (shortfall * salience[:,None] * active[:,None]).astype(np.float32)


def coordinate_onsets(probability, density, beats, prefix_masks, *, candidates_per_beat=24, max_logit_shift=1.5):
    """Bound expected rows per beat while preserving music rank and prefix.

    A common bounded log-odds offset calibrates probabilities. It neither
    selects top-k rows nor guarantees that a target density will be filled.
    """
    import torch
    if not isinstance(candidates_per_beat,int) or candidates_per_beat<1 or not np.isfinite(max_logit_shift) or max_logit_shift<=0:
        raise ValueError('Density calibration needs positive grid size and logit bound.')
    if density is None or beats is None or density.shape != (*probability.shape,2) or beats.shape != probability.shape:
        raise ValueError('Density coordination requires aligned density and absolute beat positions.')
    if (not torch.isfinite(density).all() or (density<0).any() or not torch.isfinite(beats).all()
            or not torch.isfinite(probability).all() or ((probability<0)|(probability>1)).any()):
        raise ValueError('Density coordination requires finite probabilities and nonnegative targets.')
    result = probability.clone()
    prefix = prefix_masks.shape[1]
    original_rows, adjusted_rows, limit_hits, groups = [], [], 0, 0
    for batch in range(len(probability)):
        ids = torch.floor(beats[batch]+1e-6).long()
        for identifier in ids.unique():
            positions = torch.where(ids==identifier)[0]
            # Partial windows lack a complete beat denominator.
            if len(positions) != candidates_per_beat: continue
            editable = positions[positions>=prefix]
            if not len(editable): continue
            fixed = positions[positions<prefix]
            fixed_rows = (prefix_masks[batch,fixed]>0).sum().to(probability.dtype)
            target = density[batch,positions,1].mean() - fixed_rows
            p = probability[batch,editable]
            flexible = (p>0) & (p<1)
            logits = torch.logit(p.clamp(1e-7,1-1e-7))
            def shifted(value):
                return torch.where(flexible,(logits+value).sigmoid(),p)
            lower, upper = p.new_tensor(-max_logit_shift),p.new_tensor(max_logit_shift)
            for _ in range(20):
                middle = (lower+upper)/2
                less = shifted(middle).sum() < target
                lower,upper = torch.where(less,middle,lower),torch.where(less,upper,middle)
            shift=(lower+upper)/2
            adjusted = shifted(shift)
            result[batch,editable] = adjusted
            original_rows.append(p.sum());adjusted_rows.append(adjusted.sum())
            limit_hits += int(shift.abs()>=max_logit_shift-1e-4)
            groups += 1
    summary = {'version':COORDINATION_VERSION,'calibrated_beats':groups,'bounded_beats':limit_hits,
               'max_logit_shift':max_logit_shift,'hard_row_budget':False,
               'expected_rows_before':float(torch.stack(original_rows).sum()) if original_rows else 0.,
               'expected_rows_after':float(torch.stack(adjusted_rows).sum()) if adjusted_rows else 0.}
    return result, summary
