"""Offline constant-tempo and beat-phase estimation, using NumPy only."""

import numpy as np


def _onsets(audio, rate):
    hop = max(1, round(rate * .005))
    size = 1 << max(8, round(rate * .035).bit_length())
    padded = np.pad(audio, (size // 2, size // 2 + hop))
    count = (len(audio) + hop - 1) // hop
    window = np.hanning(size).astype(np.float32)
    flux = np.zeros(count, dtype=np.float64)
    previous = None
    for start in range(0, count, 256):
        indices = np.arange(start, min(count, start + 256)) * hop
        frames = padded[indices[:, None] + np.arange(size)]
        spectrum = np.log1p(np.abs(np.fft.rfft(frames * window)) * 10)
        # Suppress stationary tones and their small frequency fluctuations.
        reference = np.concatenate((spectrum[:1] if previous is None else previous, spectrum[:-1]))
        reference = np.maximum.reduce((reference, np.roll(reference, 1, axis=1), np.roll(reference, -1, axis=1)))
        flux[start:start + len(indices)] = np.maximum(spectrum - reference, 0).mean(axis=1)
        previous = spectrum[-1:]
    if flux.max(initial=0) < 1e-8:
        return np.empty(0), np.empty(0)
    baseline = np.convolve(flux, np.ones(41) / 41, mode="same")[:len(flux)]
    strength = np.maximum(flux - baseline * .5, 0)
    peaks = np.flatnonzero((strength[1:-1] > strength[:-2]) & (strength[1:-1] >= strength[2:])) + 1
    peaks = peaks[strength[peaks] > max(np.quantile(strength, .7) * 1.3, strength.max() * .035)]
    selected = []
    for peak in peaks[np.argsort(strength[peaks])[::-1]]:
        if all(abs(int(peak) - other) * hop / rate >= .085 for other in selected):
            selected.append(int(peak))
        if len(selected) >= 1600:
            break
    selected.sort()
    # Refine each spectral attack against physical 1 ms waveform energy.
    fine_hop = max(1, round(rate * .001))
    fine = np.pad(audio, (0, (-len(audio)) % fine_hop)).reshape(-1, fine_hop)
    energy = np.mean(fine.astype(np.float64) ** 2, axis=1)
    derivative = np.maximum(np.diff(energy, prepend=0), 0)
    times, weights = [], []
    for peak in selected:
        center = round(peak * hop / fine_hop)
        lower, upper = max(0, center - 35), min(len(derivative), center + 36)
        attack = lower + int(np.argmax(derivative[lower:upper]))
        times.append(attack * fine_hop / rate)
        weights.append(strength[peak])
    return np.asarray(times), np.asarray(weights)


def _score_grid(bpms, times, weights):
    scores, phases = [], []
    span = max(1., times[-1] - times[0])
    for start in range(0, len(bpms), 64):
        rates = bpms[start:start + 64] / 60
        angles = times[None] * rates[:, None] * (2 * np.pi)
        vectors = np.exp(1j * angles) @ weights / weights.sum()
        coherence = np.abs(vectors)
        # Penalize a doubled grid with most beats missing.
        coverage = np.minimum(1., len(times) / (span * rates))
        scores.extend(coherence * np.sqrt(coverage))
        phases.extend((np.angle(vectors) / (2 * np.pi)) % 1)
    return np.asarray(scores), np.asarray(phases)


def estimate_tempo(waveform, sample_rate=24_000):
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or sample_rate <= 0:
        raise ValueError("Tempo estimation needs finite mono audio and a positive sample rate.")
    # Bound work on very long tracks; never move the audio time origin.
    audio = audio[:sample_rate * 180]
    fallback = {"bpm": 120., "offset_ms": 0., "confidence": 0., "alternatives": [],
                "status": "insufficient_signal", "method": "spectral-flux-grid-v2",
                "note": "起音不足，未测得可靠 BPM/offset；120 BPM 为占位值，请手动校准。"}
    if len(audio) < sample_rate * 5 or np.max(np.abs(audio)) < 1e-7:
        return fallback
    times, weights = _onsets(audio, sample_rate)
    if len(times) < 6 or times[-1] - times[0] < 3:
        return fallback
    weights = np.minimum(weights, np.quantile(weights, .9))
    grid = np.arange(60., 220.001, .1)
    scores, phases = _score_grid(grid, times, weights)
    candidates = []
    local_peaks = np.flatnonzero((scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:])) + 1
    local_peaks = np.unique(np.concatenate(([0, len(grid) - 1], local_peaks)))
    for index in local_peaks[np.argsort(scores[local_peaks])[::-1]]:
        if all(abs(grid[index] - value) > max(3, value * .03) for value in candidates):
            candidates.append(float(grid[index]))
        if len(candidates) >= 6:
            break
    refined = []
    for center in candidates:
        fine_grid = np.arange(max(60, center - .15), min(220, center + .15) + .001, .002)
        fine_scores, fine_phases = _score_grid(fine_grid, times, weights)
        best = int(np.argmax(fine_scores))
        bpm = float(fine_grid[best])
        offset = float(fine_phases[best] * 60_000 / bpm)
        refined.append({"bpm": round(bpm, 3), "offset_ms": round(offset, 1), "score": float(fine_scores[best])})
    refined.sort(key=lambda item: item["score"], reverse=True)
    best = refined[0]
    period = 60_000 / best["bpm"]
    residual = np.abs((times * 1000 - best["offset_ms"] + period / 2) % period - period / 2)
    alignment = float(np.sum(weights[residual < 30]) / weights.sum())
    ambiguity = refined[1]["score"] / max(best["score"], 1e-8) if len(refined) > 1 else 0
    confidence = float(np.clip(best["score"] * alignment * (1 - .45 * ambiguity), 0, 1))
    status = "estimated" if confidence >= .3 else "uncertain"
    alternatives = [item for item in refined if item['score'] >= max(.15, best['score'] * .4)][:3]
    return {
        "bpm": best["bpm"], "offset_ms": best["offset_ms"], "confidence": round(confidence, 3),
        "alternatives": [{**item, "score": round(item["score"], 4)} for item in alternatives],
        "status": status, "method": "spectral-flux-grid-v2", "onset_count": len(times),
        "analysis_seconds": round(len(audio) / sample_rate, 2), "aligned_onset_fraction": round(alignment, 3),
        "note": "已自动测量 BPM/offset；offset 是拍网格偏移，不保证为小节第一拍。" if status == "estimated" else "拍速可能存在半速/倍速或变速歧义，请试听候选速度。",
    }
