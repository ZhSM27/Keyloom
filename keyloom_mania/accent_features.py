"""Accent descriptors: percussive band onsets, harmonic and pitch-class onsets.

Median-filter harmonic/percussive separation (time median keeps sustained
tones, frequency median keeps transients) splits the spectrum before onset
detection, so a kick, a snare/hat hit and a new melody note become separate
channels. Every channel is standardised within the song (median and MAD), as
accents are relative to the rest of the recording. Frames use a 5 ms hop,
matching the local rhythm descriptors; candidates are sampled at the same
offsets.
"""
import numpy as np

from .local_rhythm import OFFSETS_MS
from .tiny_data import interpolate_features

ACCENT_SCHEMA = "accent-hpss-v1"
SAMPLE_RATE = 24_000
N_FFT = 1024
ANALYSIS_HOP = 240        # 10 ms analysis frames
OUTPUT_HOP_MS = 5.0
MEDIAN_FRAMES = 17        # harmonic: time median over 170 ms
MEDIAN_BINS = 17          # percussive: frequency median over ~400 Hz
PERCUSSIVE_BANDS_HZ = ((30, 150), (150, 2000), (2000, 12000))
PITCH_RANGE_HZ = (60, 5000)
CHANNELS = ("full_flux", "kick_flux", "snare_flux", "hat_flux", "harmonic_flux", "pitch_flux",
            "percussive_level", "harmonic_level")
ACCENT_WIDTH = len(CHANNELS)
ACCENT_LOCAL_WIDTH = len(OFFSETS_MS) * ACCENT_WIDTH


def _spectrogram(waveform):
    padded = np.pad(np.asarray(waveform, dtype=np.float32), (N_FFT // 2, N_FFT // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::ANALYSIS_HOP]
    window = np.hanning(N_FFT).astype(np.float32)
    return np.concatenate([np.abs(np.fft.rfft(frames[start:start + 2048] * window, axis=1)).astype(np.float32)
                           for start in range(0, len(frames), 2048)])


def _median(values, size, axis):
    pad = [(0, 0), (0, 0)]
    pad[axis] = (size // 2, size // 2)
    padded = np.pad(values, pad, mode="edge")
    result = np.empty_like(values)
    step = 4096
    if axis == 0:
        for start in range(0, len(values), step):
            window = np.lib.stride_tricks.sliding_window_view(padded[start:start + step + size - 1], size, axis=0)
            result[start:start + step] = np.median(window, axis=-1)
    else:
        for start in range(0, len(values), step):
            window = np.lib.stride_tricks.sliding_window_view(padded[start:start + step], size, axis=1)
            result[start:start + step] = np.median(window, axis=-1)
    return result


def _flux(magnitude):
    logarithm = np.log1p(100 * magnitude)
    return np.maximum(np.diff(logarithm, axis=0, prepend=logarithm[:1]), 0).mean(axis=1)


def _standardise(values):
    median = np.median(values)
    spread = np.median(np.abs(values - median)) * 1.4826
    return ((values - median) / max(spread, 1e-6)).astype(np.float32)


def accent_frames(waveform):
    """[frames at 5 ms, ACCENT_WIDTH] within-song standardised accent channels."""
    magnitude = _spectrogram(waveform)
    harmonic = _median(magnitude, MEDIAN_FRAMES, axis=0)
    percussive = _median(magnitude, MEDIAN_BINS, axis=1)
    total = harmonic ** 2 + percussive ** 2 + 1e-12
    harmonic_part = magnitude * harmonic ** 2 / total
    percussive_part = magnitude * percussive ** 2 / total
    frequencies = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    channels = [_flux(magnitude)]
    for lower, upper in PERCUSSIVE_BANDS_HZ:
        band = (frequencies >= lower) & (frequencies < upper)
        channels.append(_flux(percussive_part[:, band]))
    channels.append(_flux(harmonic_part))
    pitched = (frequencies >= PITCH_RANGE_HZ[0]) & (frequencies < PITCH_RANGE_HZ[1])
    pitch_class = np.mod(np.rint(12 * np.log2(frequencies[pitched] / 440) + 69), 12).astype(np.int64)
    chroma = np.zeros((len(magnitude), 12), dtype=np.float32)
    for index in range(12):
        chroma[:, index] = harmonic_part[:, pitched][:, pitch_class == index].sum(axis=1)
    chroma /= chroma.sum(axis=1, keepdims=True) + 1e-6
    channels.append(np.maximum(np.diff(chroma, axis=0, prepend=chroma[:1]), 0).sum(axis=1)
                    * np.log1p(harmonic_part[:, pitched].sum(axis=1)))
    channels.append(np.log1p(percussive_part.sum(axis=1)))
    channels.append(np.log1p(harmonic_part.sum(axis=1)))
    analysis = np.stack([_standardise(channel) for channel in channels], axis=1)
    # Resample 10 ms analysis frames onto the 5 ms grid used by local descriptors.
    times = np.arange(int(np.ceil(len(analysis) * ANALYSIS_HOP / SAMPLE_RATE * 1000 / OUTPUT_HOP_MS))) * OUTPUT_HOP_MS
    source = np.arange(len(analysis)) * ANALYSIS_HOP / SAMPLE_RATE * 1000
    return np.stack([np.interp(times, source, analysis[:, column]) for column in range(ACCENT_WIDTH)], axis=1).astype(np.float32)


def sample_accents(frames, times_ms):
    """Accent channels at each time and the local-rhythm offsets: [len(times), ACCENT_LOCAL_WIDTH]."""
    times = np.asarray(times_ms, dtype=np.float64)
    positions = np.clip(times[:, None] + np.asarray(OFFSETS_MS, dtype=np.float64)[None], 0, (len(frames) - 1) * OUTPUT_HOP_MS)
    return interpolate_features(frames, positions.reshape(-1), OUTPUT_HOP_MS).reshape(len(times), ACCENT_LOCAL_WIDTH)


def build_accent_cache(manifest, output, splits=("train", "validation"), workers=4, progress=print):
    """Cache accent frames per audio file for the named manifest splits; resumable, never reads other splits."""
    import hashlib
    import json
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    manifest, output = Path(manifest), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audio = {}
    for line in manifest.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("split") in splits and row.get("audio_relative_path"):
            audio.setdefault(row["audio_hash"], (str(manifest.parent / row["audio_relative_path"]), row["split"]))
    tasks = [(key, path, str(output / f"{key}.npz")) for key, (path, _) in sorted(audio.items())
             if not (output / f"{key}.npz").is_file()]
    progress(f"{len(audio)} audio files, {len(tasks)} to compute")
    failures = {}
    with ProcessPoolExecutor(workers) as pool:
        for index, (key, error) in enumerate(pool.map(_cache_one, tasks, chunksize=2), start=1):
            if error:
                failures[key] = error
            if index % 100 == 0:
                progress(f"  {index}/{len(tasks)}")
    files = {key: hashlib.sha256((output / f"{key}.npz").read_bytes()).hexdigest()
             for key in audio if (output / f"{key}.npz").is_file()}
    record = {"schema": ACCENT_SCHEMA, "channels": CHANNELS, "hop_ms": OUTPUT_HOP_MS, "sample_rate": SAMPLE_RATE,
              "dtype": "float16", "splits": list(splits), "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "audio_files": len(audio), "cached": len(files), "failures": failures,
              "split_of_audio": {key: split for key, (_, split) in audio.items()}, "files": files}
    (output / "manifest.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def _cache_one(task):
    key, path, target = task
    try:
        from pathlib import Path
        from .features import AcousticConfig, decode_waveform

        waveform, _ = decode_waveform(Path(path), AcousticConfig(sample_rate=SAMPLE_RATE))
        frames = accent_frames(waveform).astype(np.float16)
        temporary = target + ".tmp.npz"
        np.savez_compressed(temporary, frames=frames)
        Path(temporary).replace(target)
        return key, None
    except Exception as error:
        return key, f"{type(error).__name__}: {error}"[:200]
