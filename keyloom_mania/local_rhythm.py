"""Audio-local rhythm descriptors used by the released generators."""
import numpy as np
from .tiny_data import SEQUENCE_LENGTH, interpolate_features
OFFSETS_MS = (-80, -40, -20, -10, 0, 10, 20, 40, 80)
DESCRIPTOR_WIDTH = 40
LOCAL_WIDTH = len(OFFSETS_MS) * DESCRIPTOR_WIDTH


def rhythm_descriptors(logmel: np.ndarray) -> np.ndarray:
    if logmel.ndim != 2 or logmel.shape[1] != 128 or not len(logmel) or not np.isfinite(logmel).all():
        raise ValueError("Local rhythm requires finite 128-bin log-power Mel frames.")
    values = logmel.astype(np.float32)
    pooled = values.reshape(len(values), 32, 4).mean(axis=-1)
    bands = values.reshape(len(values), 4, 32)
    maximum = bands.max(axis=-1)
    energy = maximum + np.log(np.exp(bands - maximum[..., None]).mean(axis=-1))
    delta = np.maximum(np.diff(values, axis=0, prepend=values[:1]), 0)
    flux = np.log1p(delta.reshape(len(values), 4, 32).mean(axis=-1))
    return np.concatenate((pooled, energy, flux), axis=-1).astype(np.float32)

def sample_neighborhood(descriptors: np.ndarray, times_ms: np.ndarray) -> np.ndarray:
    if descriptors.ndim != 2 or descriptors.shape[1] != DESCRIPTOR_WIDTH or times_ms.ndim != 2 or times_ms.shape[1] != SEQUENCE_LENGTH:
        raise ValueError("Local neighborhoods require complete grid windows and 40-channel frames.")
    positions = times_ms[..., None] + np.asarray(OFFSETS_MS, dtype=np.float64)
    sampled = interpolate_features(descriptors, positions.reshape(-1), 5.0)
    return sampled.reshape(len(times_ms), SEQUENCE_LENGTH, LOCAL_WIDTH)
