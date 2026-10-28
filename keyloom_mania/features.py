from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np

from .audio import MAX_DURATION_SECONDS, TimestampTracker, asset_path


SCHEMA_VERSION = "acoustic-logmel-v1"


@dataclass(frozen=True)
class AcousticConfig:
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 120
    n_mels: int = 128
    f_min: float = 30.0
    f_max: float = 12000.0
    power: float = 2.0
    center: bool = True
    pad_mode: str = "constant"
    mel_scale: str = "slaney"
    norm: str = "slaney"
    log_floor: float = 1e-10

    def __post_init__(self):
        if self.sample_rate <= 0 or self.n_fft < 2 or self.hop_length < 1 or not 1 <= self.n_mels <= self.n_fft // 2 + 1:
            raise ValueError("Invalid acoustic dimensions.")
        if not 0 <= self.f_min < self.f_max <= self.sample_rate / 2 or not 0 < self.log_floor < 1:
            raise ValueError("Invalid acoustic frequency range or log floor.")
        if not self.center or self.pad_mode != "constant" or self.power != 2:
            raise ValueError("This timestamp schema requires centered constant-padding power spectrograms.")


def feature_frame_times(frame_count: int, config: AcousticConfig = AcousticConfig()) -> np.ndarray:
    return np.arange(frame_count, dtype=np.float64) * config.hop_length / config.sample_rate


def feature_frame_span(start_ms: float, end_ms: float, frame_count: int, config: AcousticConfig = AcousticConfig()) -> tuple[int, int]:
    if not math.isfinite(start_ms) or not math.isfinite(end_ms) or end_ms <= start_ms:
        raise ValueError("Feature window must have finite increasing bounds.")
    scale = config.sample_rate / (config.hop_length * 1000)
    start = max(0, min(frame_count, math.ceil(start_ms * scale - 1e-9)))
    end = max(0, min(frame_count, math.ceil(end_ms * scale - 1e-9)))
    return start, end


def waveform_levels(waveform: np.ndarray) -> tuple[float, float]:
    if waveform.ndim != 1 or not waveform.size:
        raise ValueError("Waveform levels require nonempty one-dimensional audio.")
    peak = 0.0
    square_sum = 0.0
    for start in range(0, waveform.size, 262144):
        block = waveform[start:start + 262144].astype(np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Audio samples must be finite.")
        peak = max(peak, float(np.max(np.abs(block))))
        square_sum += float(np.dot(block, block))
    return peak, math.sqrt(square_sum / waveform.size)


def decode_waveform(path: Path, config: AcousticConfig = AcousticConfig()) -> tuple[np.ndarray, dict]:
    pieces = []
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=config.sample_rate)
    native_sample_count = 0
    analysis_sample_count = 0
    with av.open(str(path)) as source:
        if not source.streams.audio:
            raise ValueError("Audio file has no audio stream.")
        stream = source.streams.audio[0]
        tracker = TimestampTracker(stream.codec_context.name)
        native_rate = None
        native_channels = None
        for frame in source.decode(stream):
            if native_rate is None:
                native_rate = frame.sample_rate
                native_channels = len(frame.layout.channels)
            if frame.sample_rate != native_rate or len(frame.layout.channels) != native_channels:
                raise ValueError("Source audio format changes during decoding.")
            if getattr(frame, "is_corrupt", False):
                raise ValueError("Decoder marked an audio frame as corrupt.")
            timestamp = float(frame.pts * frame.time_base) if frame.pts is not None and frame.time_base is not None else None
            tracker.observe(timestamp, frame.samples, frame.sample_rate)
            native_sample_count += frame.samples
            frame.pts = None
            for output_frame in resampler.resample(frame):
                samples = output_frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                pieces.append(samples)
                analysis_sample_count += samples.size
                if analysis_sample_count > config.sample_rate * MAX_DURATION_SECONDS:
                    raise ValueError("Audio exceeds the permitted duration.")
        for output_frame in resampler.resample(None):
            samples = output_frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)
            pieces.append(samples)
            analysis_sample_count += samples.size
        tracker.finish()
    if not pieces or not native_rate:
        raise ValueError("Audio has no decoded samples.")
    waveform = np.concatenate(pieces)
    if waveform.size > config.sample_rate * MAX_DURATION_SECONDS:
        raise ValueError("Audio exceeds the permitted duration.")
    if abs(waveform.size / config.sample_rate - native_sample_count / native_rate) > 2 / config.sample_rate:
        raise ValueError("Resampling did not preserve the decoded sample timeline.")
    peak, rms = waveform_levels(waveform)
    return waveform, {
        "native_sample_rate": native_rate,
        "native_channels": native_channels,
        "native_sample_count": native_sample_count,
        "analysis_sample_count": int(waveform.size),
        "first_source_pts_seconds": tracker.origin_seconds,
        "max_timestamp_drift_ms": tracker.max_drift_ms,
        "decoded_duration_seconds": waveform.size / config.sample_rate,
        "peak_amplitude": peak,
        "rms_amplitude": rms,
    }


def logmel_features(waveform: np.ndarray, config: AcousticConfig = AcousticConfig(), device: str = "cpu") -> np.ndarray:
    import torch
    import torchaudio

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("Expected a nonempty finite mono waveform.")
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        f_min=config.f_min,
        f_max=config.f_max,
        power=config.power,
        center=config.center,
        pad_mode=config.pad_mode,
        mel_scale=config.mel_scale,
        norm=config.norm,
    ).to(device)
    with torch.inference_mode():
        tensor = torch.from_numpy(waveform.copy()).to(device)
        mel = transform(tensor).clamp_min(config.log_floor).log()
        frame_count = (waveform.size + config.hop_length - 1) // config.hop_length
        result = mel[:, :frame_count].transpose(0, 1).contiguous().cpu().numpy()
    if result.shape != (frame_count, config.n_mels) or not np.isfinite(result).all():
        raise ValueError("Invalid log-Mel output.")
    return result


def hash_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
