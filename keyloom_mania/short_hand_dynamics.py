"""Ordered observations of the last four beats; no persistent pressure debt."""
import numpy as np
import torch

SCHEMA = 'short-hand-dynamics-v1-2-4-ordered'
WINDOWS = (2., 4.)
BIN_BEATS = .125
BINS = 32
CHANNELS = 9
SCALARS = 5


def ordered_hand_context(previous, beats, seconds_per_beat, seed=None):
    """Return oldest-to-newest eighth-beat observations, excluding this note.

    The left endpoint is open, including the endpoint of the newer bin. No
    previous-hit state survives four beats. Structural model history is separate.
    """
    size, length = previous.shape
    if beats.shape != previous.shape:
        raise ValueError('History and times must align.')
    masks = torch.cat((torch.zeros_like(previous[:, :1]), previous[:, 1:]), 1)
    times = torch.cat((beats[:, :1] - 1e-4, beats[:, :-1]), 1)
    prefix = 0
    if seed is not None:
        prefix = seed[0].shape[1]
        masks = torch.cat((seed[0], masks), 1)
        times = torch.cat((seed[1], times), 1)
    columns = (masks[..., None] & (1 << torch.arange(4, device=masks.device))) != 0
    hands = torch.stack((columns[..., :2].any(-1), columns[..., 2:].any(-1)), -1)
    doubles = torch.stack((columns[..., :2].all(-1), columns[..., 2:].all(-1)), -1)
    event = torch.cat((columns, hands, doubles, (masks > 0)[..., None]), -1).float()
    cumulative = torch.cat((event.new_zeros(size, 1, CHANNELS), event.cumsum(1)), 1)
    upper = prefix + torch.arange(length, device=masks.device) + 1
    edges = beats[..., None] - torch.arange(BINS, -1, -1, device=beats.device) * BIN_BEATS
    indices = torch.searchsorted(times.contiguous(), edges.reshape(size, -1).contiguous(), right=True).reshape(size, length, BINS + 1)
    indices = torch.minimum(indices, upper[None, :, None])
    totals = cumulative.gather(1, indices.reshape(size, -1, 1).expand(-1, -1, CHANNELS)).reshape(size, length, BINS + 1, CHANNELS)
    bins = totals[:, :, 1:] - totals[:, :, :-1]
    last = torch.where(hands, times[..., None], torch.full_like(times[..., None], -10000.)).cummax(1).values
    last = last[:, upper - 1]
    seen = (last > beats[..., None] - 4) & (last < beats[..., None])
    gap = torch.where(seen, (beats[..., None] - last).clamp(0, 4) / 4, torch.ones_like(last))
    scalars = torch.cat((gap.float(), seen.float(), seconds_per_beat[:, :length, None].float().clamp(0, 2) / 2), -1)
    return bins, scalars


def numpy_signature(masks, beats, now, bpm):
    """Same finite-horizon observations for source-prefix matching, without torch."""
    masks = np.asarray(masks, np.int64); beats = np.asarray(beats, float)
    selected = (beats > now - 4 + 1e-7) & (beats < now - 1e-7) & (masks > 0)
    masks, beats = masks[selected], beats[selected]
    out = np.zeros((BINS, CHANNELS), np.float32)
    last = np.full(2, -np.inf)
    for mask, time in zip(masks, beats):
        # Open-left bins: an exact edge belongs to its older bin.
        index = int(np.ceil((time - now + 4) / BIN_BEATS - 1e-7)) - 1
        if not 0 <= index < BINS:
            continue
        columns = np.array([(int(mask) >> c) & 1 for c in range(4)])
        hands = np.array([columns[:2].any(), columns[2:].any()])
        doubles = np.array([columns[:2].all(), columns[2:].all()])
        out[index] += np.r_[columns, hands, doubles, 1]
        last[hands] = time
    seen = np.isfinite(last)
    gap = np.where(seen, np.minimum(now - last, 4) / 4, 1)
    return np.r_[out.reshape(-1), gap, seen.astype(float), min(60 / bpm, 2) / 2].astype(np.float32)


def prefix_distance(first, second):
    """Similarity of observations, never an absolute low-pressure preference."""
    a, b = np.asarray(first), np.asarray(second)
    bins_a, bins_b = a[:BINS * CHANNELS].reshape(BINS, CHANNELS), b[:BINS * CHANNELS].reshape(BINS, CHANNELS)
    # Preserve hand identity and timing order; normalize by observed activity.
    scale = max(1., float(np.abs(bins_a).sum() + np.abs(bins_b).sum()))
    event_distance = float(np.abs(bins_a - bins_b).sum()) / scale
    gap_distance = float(np.abs(a[-5:-1] - b[-5:-1]).mean())
    return .8 * event_distance + .2 * gap_distance


def hand_allocation(masks):
    """Joint left/right note counts, nine categorical outcomes, no balance target."""
    if isinstance(masks, torch.Tensor):
        return ((masks & 1) > 0).long() * 3 + ((masks & 2) > 0).long() * 3 + ((masks & 4) > 0).long() + ((masks & 8) > 0).long()
    masks = np.asarray(masks)
    return ((masks & 1) > 0).astype(int) * 3 + ((masks & 2) > 0).astype(int) * 3 + ((masks & 4) > 0).astype(int) + ((masks & 8) > 0).astype(int)
