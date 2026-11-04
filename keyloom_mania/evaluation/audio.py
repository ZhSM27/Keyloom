"""Reference-free correspondence between exported rows and audio onsets.

Each lift compares the chart with copies of itself shifted along its own beat
grid, so it asks whether notes follow this recording rather than whether they
sit on a grid. Accent measures ask whether chords (rows of two or more notes)
sit on the recording's strongest onsets, and how often chords or rows appear
where no onset supports them. Thresholds are descriptive; only a reference
with enough human audio calibrates them into scores.
"""
from __future__ import annotations

import numpy as np

from .charts import absolute_beats, beats_to_ms

AUDIO_SCHEMA = "onset-alignment-v2-accent"
STRONG_QUANTILE = .75     # strong accents: the top quarter of onset strengths within the chart span
SAMPLE_RATE = 24_000
HOP = 120
N_FFT = 1024
HOP_MS = 1000 * HOP / SAMPLE_RATE
TOLERANCE_MS = 30.0
TIMING_SHIFTS_BEATS = (-3 / 8, -1 / 8, 1 / 8, 3 / 8)
SELECTION_SHIFTS_BEATS = (-2., -1., 1., 2.)


def onset_envelope(waveform):
    """Positive log-spectral flux and frame energy on a 5 ms hop."""
    waveform = np.asarray(waveform, dtype=np.float32)
    padded = np.pad(waveform, (N_FFT // 2, N_FFT // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]
    window = np.hanning(N_FFT).astype(np.float32)
    flux = np.zeros(len(frames), dtype=np.float32)
    energy = np.zeros(len(frames), dtype=np.float32)
    previous = None
    for start in range(0, len(frames), 2048):
        block = frames[start:start + 2048]
        energy[start:start + len(block)] = np.mean(block.astype(np.float32) ** 2, axis=1)
        magnitude = np.log1p(100 * np.abs(np.fft.rfft(block * window, axis=1)))
        before = magnitude[:1] if previous is None else previous[None]
        flux[start:start + len(block)] = np.maximum(np.diff(magnitude, axis=0, prepend=before), 0).mean(axis=1)
        previous = magnitude[-1]
    return flux, energy


def onset_peaks(flux):
    """Local novelty maxima within +/-25 ms above the median positive novelty."""
    width = int(500 / HOP_MS)
    smooth = np.convolve(flux, np.ones(width) / width, mode="same")
    novelty = flux - smooth
    radius = int(25 / HOP_MS)
    padded = np.pad(novelty, radius, constant_values=-np.inf)
    local = novelty >= np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1).max(axis=1)
    positive = novelty[novelty > 0]
    threshold = float(np.median(positive)) if len(positive) else 0.
    peaks = np.flatnonzero(local & (novelty > threshold))
    return peaks * HOP_MS, novelty[peaks]


def _precision(times, peaks):
    if not len(times) or not len(peaks):
        return 0.
    index = np.clip(np.searchsorted(peaks, times), 1, len(peaks) - 1)
    distance = np.minimum(np.abs(times - peaks[index - 1]), np.abs(times - peaks[index]))
    return float(np.mean(distance <= TOLERANCE_MS))


def _rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    for value in np.unique(values):
        same = values == value
        ranks[same] = ranks[same].mean()
    return ranks


def spearman(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if len(first) < 8 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return None
    return float(np.corrcoef(_rank(first), _rank(second))[0, 1])


def _nearest(times, peaks):
    """Distance to and index of the nearest peak for each time."""
    if len(peaks) < 2 or not len(times):
        return np.full(len(times), np.inf), np.zeros(len(times), dtype=np.int64)
    index = np.clip(np.searchsorted(peaks, times), 1, len(peaks) - 1)
    left, right = np.abs(times - peaks[index - 1]), np.abs(times - peaks[index])
    return np.minimum(left, right), np.where(left <= right, index - 1, index)


def accent_metrics(clock, times, masks, peaks, strength, stream=None):
    """Chords on strong onsets versus singles, and chords or rows without onset support."""
    sizes = np.asarray([mask.bit_count() for mask in np.asarray(masks).tolist()])
    chords = sizes >= 2
    span = (peaks >= times.min() - 100) & (peaks <= times.max() + 100)
    result = dict.fromkeys(("accent_lift", "strong_accent_chord_recall", "size_strength_spearman", "unsupported_chords",
                            "unsupported_offstream_rows", "beats_rows_exceed_supply"))
    if span.sum() < 16:
        return result
    cut = np.quantile(strength[span], STRONG_QUANTILE)
    distance, index = _nearest(times, peaks)
    supported = distance <= TOLERANCE_MS
    strong = supported & (strength[index] >= cut)
    if chords.sum() >= 8 and (~chords).sum() >= 8:
        # Half-count smoothing keeps the ratio finite when no single lands on a strong onset.
        chord_rate = (strong[chords].sum() + .5) / (chords.sum() + 1)
        single_rate = (strong[~chords].sum() + .5) / ((~chords).sum() + 1)
        result["accent_lift"] = float(chord_rate / single_rate)
    strong_peaks = peaks[span][strength[span] >= cut]
    if chords.sum() >= 2 and len(strong_peaks) >= 8:
        result["strong_accent_chord_recall"] = float(np.mean(_nearest(strong_peaks, times[chords])[0] <= TOLERANCE_MS))
    result["size_strength_spearman"] = spearman(sizes, np.where(supported, strength[index], 0.))
    if chords.sum() >= 8:
        result["unsupported_chords"] = float(1 - supported[chords].mean())
    if stream is not None:
        result["unsupported_offstream_rows"] = float(np.mean(~supported & ~np.asarray(stream, dtype=bool)))
    # Rows per beat against onsets available in that beat (half-open, shifted by the tolerance).
    beats = absolute_beats(clock, times)
    beat_ids = np.floor(beats + 1e-6).astype(np.int64)
    ids = np.arange(beat_ids.min(), beat_ids.max() + 1)
    edges = beats_to_ms(clock, np.concatenate((ids, [ids[-1] + 1]))) - TOLERANCE_MS
    supply = np.diff(np.searchsorted(peaks, edges))
    placed = np.bincount(beat_ids - ids[0], minlength=len(ids))
    active = placed > 0
    if active.sum() >= 16:
        result["beats_rows_exceed_supply"] = float(np.mean(placed[active] > supply[active] + 1))
    return result


def alignment_metrics(waveform, clock, times, masks=None, stream=None):
    times = np.asarray(times, dtype=np.float64)
    flux, energy = onset_envelope(waveform)
    peaks, strength = onset_peaks(flux)
    beats = absolute_beats(clock, times)
    observed = _precision(times, peaks)

    def lift(shifts):
        baseline = np.mean([_precision(beats_to_ms(clock, beats + shift), peaks) for shift in shifts])
        return float(observed / baseline) if baseline > 0 else None

    span = (peaks >= times.min() - 100) & (peaks <= times.max() + 100)
    strong = peaks[span][strength[span] >= np.quantile(strength[span], .75)] if span.sum() >= 8 else np.empty(0)
    recall = None
    if len(strong):
        index = np.clip(np.searchsorted(times, strong), 1, max(1, len(times) - 1))
        nearest = np.minimum(np.abs(strong - times[index - 1]), np.abs(strong - times[np.minimum(index, len(times) - 1)]))
        recall = float(np.mean(nearest <= TOLERANCE_MS))
    bars = np.floor(beats / 4).astype(np.int64)
    indices = np.arange(bars.min(), bars.max() + 1)
    edges = beats_to_ms(clock, np.concatenate((indices, [indices[-1] + 1])) * 4) / HOP_MS
    edges = np.clip(np.rint(edges).astype(np.int64), 0, len(energy))
    bar_energy = np.asarray([10 * np.log10(energy[a:b].mean() + 1e-10) if b > a else np.nan for a, b in zip(edges[:-1], edges[1:])])
    bar_rows = np.bincount(bars - indices[0], minlength=len(indices))
    valid = np.isfinite(bar_energy)
    return {"schema": AUDIO_SCHEMA, "onset_peaks": int(len(peaks)), "tolerance_ms": TOLERANCE_MS,
            "onset_precision": observed, "timing_lift": lift(TIMING_SHIFTS_BEATS),
            "selection_lift": lift(SELECTION_SHIFTS_BEATS), "strong_onset_recall": recall,
            "energy_density_spearman": spearman(bar_energy[valid], bar_rows[valid]),
            **(accent_metrics(clock, times, masks, peaks, strength, stream) if masks is not None else {})}


def decode_audio(path):
    from ..features import AcousticConfig, decode_waveform

    return decode_waveform(path, AcousticConfig(sample_rate=SAMPLE_RATE))[0]


def audio_metrics(path, clock, times, masks=None, stream=None, waveform=None):
    return alignment_metrics(decode_audio(path) if waveform is None else waveform, clock, times, masks, stream)
