"""Absolute phase-grid consensus, avoiding cumulative missed-beat indices."""
import numpy as np


def _candidate(beats,downbeats,center):
    bpms=np.arange(max(30.,center*.88),min(400.,center*1.12)+.01,.02)
    if not len(bpms):return None
    best=None
    for start in range(0,len(bpms),128):
        values=bpms[start:start+128]
        vectors=np.exp(2j*np.pi*values[:,None]*beats[None]/60).mean(-1)
        index=int(np.argmax(np.abs(vectors)))
        item=(float(abs(vectors[index])),float(values[index]),float(np.angle(vectors[index])/(2*np.pi)%1))
        if best is None or item[0]>best[0]:best=item
    _,bpm,phase=best;period=60/bpm;origin=phase*period
    inliers=np.ones(len(beats),dtype=bool)
    for _ in range(5):
        indices=np.rint((beats-origin)/period)
        residual=beats-(origin+indices*period)
        center_error=np.median(residual)
        mad=np.median(np.abs(residual-center_error))
        inliers=np.abs(residual-center_error)<=max(.025,min(period*.2,4*mad))
        if inliers.sum()<8:return None
        period,origin=np.polyfit(indices[inliers],beats[inliers],1)
        if period<=0:return None
    residual=np.abs(beats-(origin+np.rint((beats-origin)/period)*period))
    error=float(np.quantile(residual,.9)*1000)
    downbeat_agreement=0.;bar_phase=0
    if len(downbeats):
        index=np.rint((downbeats-origin)/period).astype(np.int64)
        aligned=np.abs(downbeats-(origin+index*period))<max(.04,period*.15)
        if aligned.any():
            votes=np.bincount(index[aligned]%4,minlength=4);bar_phase=int(votes.argmax())
            downbeat_agreement=float(votes.max()/len(downbeats))
    return {'bpm':float(60/period),'offset_ms':float(((origin+bar_phase*period)%(4*period))*1000),
        'coherence':best[0],'inlier_fraction':float(inliers.mean()),'grid_error_p90_ms':error,
        'downbeat_agreement':downbeat_agreement,'confidence':float(inliers.mean()*np.exp(-error/60))}


def fit_robust_grid(beats,downbeats=(),phase_bias_ms=0.):
    beats=np.asarray(beats,dtype=np.float64);downbeats=np.asarray(downbeats,dtype=np.float64)
    if len(beats)<8 or not np.isfinite(beats).all() or (np.diff(beats)<=0).any():raise ValueError('Not enough strictly increasing beats.')
    center=60/float(np.median(np.diff(beats)))
    candidates=[c for f in (1.,.5,2.) if 25<=center*f<=450 and (c:=_candidate(beats,downbeats,center*f)) is not None]
    if not candidates:raise ValueError('No stable absolute beat grid.')
    primary=candidates[0]
    # A higher octave needs a clear consensus gain; an equally fitting doubled
    # grid remains an alternative, not a silent tempo normalization.
    best=max(candidates,key=lambda c:c['coherence'])
    if best['coherence']>primary['coherence']+.15 and best['inlier_fraction']>.85:primary=best
    result=dict(primary)
    estimated=result['grid_error_p90_ms']<45 and result['inlier_fraction']>.8
    applied=float(phase_bias_ms) if estimated else 0.
    result['offset_ms']=(result['offset_ms']-applied)%(240000/result['bpm'])
    alternatives=[]
    for candidate in candidates:
        if candidate is primary:continue
        stable=candidate['grid_error_p90_ms']<45 and candidate['inlier_fraction']>.8
        bias=float(phase_bias_ms) if stable else 0.
        alternatives.append({'bpm':round(candidate['bpm'],3),
            'offset_ms':round((candidate['offset_ms']-bias)%(240000/candidate['bpm']),1),
            'coherence':round(candidate['coherence'],4),'status':'estimated' if stable else 'uncertain',
            'estimator_phase_bias_applied_ms':bias})
    result.update(bpm=round(result['bpm'],3),offset_ms=round(result['offset_ms'],1),confidence=round(result['confidence'],3),
        status='estimated' if estimated else 'uncertain',method='beat-this-final0',fit_policy='absolute-consensus-v4',
        estimator_phase_bias_applied_ms=applied,alternatives=alternatives,beat_count=len(beats),downbeat_count=len(downbeats),
        confidence_kind='grid_fit_quality_not_model_probability',
        note='已按稳定拍点拟合 BPM/offset；倍速解释仍可能有歧义，请结合歌曲检查。' if estimated else '恒定拍点网格不稳定，请手动校准；变速音乐需分段处理。')
    return result
