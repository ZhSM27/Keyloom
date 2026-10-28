"""Beat This! inference and robust fitting of a constant osu timing grid."""

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np


from .paths import ASSET_ROOT
CHECKPOINT = ASSET_ROOT / "audio/beat-this/final0.ckpt"
CHECKPOINT_SHA256 = '8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331'
PHASE_CALIBRATION = ASSET_ROOT / "audio/beat-this/calibration.json"
PHASE_CALIBRATION_SHA256 = '62797757a00528e77d8ec5c401578bcd148431f7a6e6a8cb2fe0844295e8fb96'
INFERENCE_LOCK = threading.Lock()


def fit_beat_grid(beats, downbeats=()):
    beats = np.asarray(beats, dtype=np.float64)
    downbeats = np.asarray(downbeats, dtype=np.float64)
    if beats.ndim != 1 or len(beats) < 8 or not np.isfinite(beats).all() or np.any(np.diff(beats) <= 0):
        raise ValueError('未检测到足够的连续拍点，请手动输入 BPM 和 offset。')
    if downbeats.ndim != 1 or not np.isfinite(downbeats).all():
        raise ValueError('Invalid downbeat positions.')
    gaps = np.diff(beats)
    period = float(np.median(gaps))
    indices = np.concatenate(([0], np.cumsum(np.maximum(1, np.rint(gaps / period))))).astype(np.float64)
    inliers = np.ones(len(beats), dtype=bool)
    for _ in range(5):
        if inliers.sum() < 8:
            raise ValueError('拍点无法拟合成稳定网格，请手动输入 BPM 和 offset。')
        period, origin = np.polyfit(indices[inliers], beats[inliers], 1)
        if period <= 0:
            raise ValueError('Invalid fitted beat interval.')
        residual = beats - (origin + indices * period)
        center = np.median(residual[inliers])
        scale = np.median(np.abs(residual[inliers] - center))
        inliers = np.abs(residual - center) <= max(.025, min(period * .2, 4 * scale))
    period, origin = np.polyfit(indices[inliers], beats[inliers], 1)
    residual = np.abs(beats - (origin + indices * period))
    error_ms = float(np.quantile(residual, .9) * 1000)
    bar_phase = 0
    downbeat_agreement = 0.
    if len(downbeats):
        downbeat_indices = np.rint((downbeats - origin) / period).astype(np.int64)
        aligned = np.abs(downbeats - (origin + downbeat_indices * period)) < max(.04, period * .15)
        if aligned.any():
            votes = np.bincount(downbeat_indices[aligned] % 4, minlength=4)
            bar_phase = int(votes.argmax())
            downbeat_agreement = float(votes.max() / len(downbeats))
    offset = float((origin + bar_phase * period) % (4 * period))
    confidence = float(inliers.mean() * np.exp(-error_ms / 60))
    status = 'estimated' if error_ms < 45 and inliers.mean() > .8 else 'uncertain'
    if not 30 <= 60 / period <= 400:
        raise ValueError('测得拍速超出 30–400 BPM，请手动校正。')
    note = 'Beat This! 已检测拍点并拟合 BPM/offset；可直接修改下方数值。'
    if status == 'uncertain':
        note = '拍点与恒定网格偏差较大，可能变速或存在漏拍；请试听并手动校准。'
    if downbeat_agreement < .6:
        note += ' 小节首拍不确定，请检查四拍分组。'
    return {
        'bpm': round(float(60 / period), 3), 'offset_ms': round(offset * 1000, 1),
        'confidence': round(confidence, 3), 'status': status, 'method': 'beat-this-final0',
        'note': note, 'alternatives': [], 'beat_count': len(beats), 'downbeat_count': len(downbeats),
        'grid_error_p90_ms': round(error_ms, 2), 'downbeat_agreement': round(downbeat_agreement, 3),
        'confidence_kind': 'grid_fit_quality_not_model_probability',
    }


def detect_timing(waveform, sample_rate=24_000):
    import torch
    from beat_this.inference import Audio2Beats

    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or sample_rate <= 0:
        raise ValueError('Expected finite mono audio and a positive sample rate.')
    if not CHECKPOINT.is_file():
        raise RuntimeError('Beat This! 权重缺失，请运行 keyloom assets fetch-audio；也可手动填写 BPM/offset。')
    with CHECKPOINT.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != CHECKPOINT_SHA256:
            raise RuntimeError('Beat This! 权重校验失败，请重新运行 keyloom assets fetch-audio。')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with INFERENCE_LOCK:
        tracker = Audio2Beats(checkpoint_path=str(CHECKPOINT), device=device, dbn=False)
        try:
            with torch.inference_mode():
                beats, downbeats = tracker(audio, sample_rate)
            from .robust_beat_grid import fit_robust_grid
            bias=0.;calibration_hash=None
            if PHASE_CALIBRATION.is_file():
                content=PHASE_CALIBRATION.read_bytes();calibration_hash=hashlib.sha256(content).hexdigest()
                if calibration_hash!=PHASE_CALIBRATION_SHA256:raise RuntimeError('拍点相位校准记录校验失败。')
                bias=float(json.loads(content)['estimator_phase_bias_ms'])
            fitted = fit_robust_grid(beats, downbeats, bias)
            fitted['phase_calibration_sha256']=calibration_hash
            fitted.update(beats_seconds=np.asarray(beats).tolist(), downbeats_seconds=np.asarray(downbeats).tolist())
            return fitted
        finally:
            del tracker
            if device == 'cuda':
                torch.cuda.empty_cache()
