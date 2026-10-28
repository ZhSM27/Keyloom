"""Opt-in music-count reference alignment; native onsets and weights stay intact."""
from functools import partial
from weakref import ref, proxy

import torch
from .segment_labels import LABELS

from .expression_generator import selected_onset_rows

COUNT_DENSITY_VERSION = 'observed-row-count-reference-v3-js-only'


def bounded_count_distribution(probability, target_mean, max_shift=.5, active=None):
    """Reduce overshooting count means within native support; never fill weak rows."""
    if not 0 <= max_shift <= 1 or probability.ndim != 2 or probability.shape[1] != 4 or target_mean.shape != probability.shape[:1]:
        raise ValueError('Invalid count sampling distribution or shift bound.')
    if not torch.isfinite(probability).all() or (probability < 0).any() or not torch.allclose(probability.sum(-1), torch.ones_like(target_mean)) or not torch.isfinite(target_mean).all():
        raise ValueError('Count sampling probabilities and targets must be finite and normalized.')
    values = torch.arange(1, 5, device=probability.device, dtype=probability.dtype)
    # A budget ceiling must not turn musically weak/single rows into extra chords.
    target_mean = torch.minimum(target_mean, (probability * values).sum(-1))
    logp = probability.log()  # log(0)=-inf preserves zero support without flooring tiny probabilities.
    low, high = torch.full_like(target_mean, -max_shift), torch.full_like(target_mean, max_shift)
    for _ in range(12):
        middle = (low + high) / 2
        mean = ((logp + middle[:, None] * values).softmax(-1) * values).sum(-1)
        low, high = torch.where(mean < target_mean, middle, low), torch.where(mean >= target_mean, middle, high)
    shift = (low + high) / 2
    unchanged = ((probability * values).sum(-1) - target_mean).abs() <= 1e-6
    shift = torch.where(unchanged, 0., shift)
    result = (logp + shift[:, None] * values).softmax(-1)
    return result if active is None else torch.where(active[:, None], result, probability)


def enable_count_density_reference(model, mix=.5, enabled=True):
    """Blend the count-only row denominator with already selected musical rows.

    No onset logits, neural density inputs, counts outside JS/HS, or prefix
    trajectories are rewritten. This deliberately retains the native count
    sampler, including its temperature, seed and probability guard.
    """
    if not 0 <= mix <= 1:
        raise ValueError('Count density reference mix must lie in [0, 1].')
    if (getattr(model, 'count_policy', None) != 'music-density-sample'
            or not hasattr(model, '_music_density') or not hasattr(model, '_reference_pass')):
        raise ValueError('This generator does not use the supported native music-density sampler.')
    if hasattr(model, 'count_density_reference'):
        raise ValueError('Count density reference was already configured.')
    def weak_bound(method):
        # Native D policies also install instance-only method functions. Keep
        # those functions alive while holding only a weak reference to the owner.
        function, owner = method.__func__, ref(method.__self__)
        def call(*args, **kwargs):
            instance = owner()
            if instance is None:
                raise RuntimeError('Count density generator was released.')
            return function(instance, *args, **kwargs)
        return call
    original_onsets, original_generate = weak_bound(model.generation_onset_logits), weak_bound(model.generate)
    model.count_density_reference = {'version': COUNT_DENSITY_VERSION, 'mix': float(mix), 'enabled': bool(enabled),
        'scope': ['Jumpstream'], 'neural_density_input': 'unchanged',
        'onsets': 'native_unmodified', 'prefix': 'committed_music_counts_then_masks', 'max_log_odds_shift_per_note': .5,
        'sampling_direction': 'reduce_overfill_only'}
    if mix and enabled:
        model.count_density_adjustment = bounded_count_distribution
    model.count_density_windows = []
    model._count_density_prefix = None

    def onsets(self, memory, condition, plan):
        logits = original_onsets(memory, condition, plan)
        if self._reference_pass or self._music_density is None or not mix or not enabled:
            return logits
        beats = self._density_beats
        if beats is None or beats.shape != logits.shape:
            raise ValueError('Observed count density requires aligned absolute beats.')
        occupancy = (logits.sigmoid() >= .5).to(logits.dtype)
        prefix = self._music_prefix
        if prefix is None:
            prefix = self._count_density_prefix
        prefix_slots = 0 if prefix is None else prefix.shape[1]
        if prefix_slots:
            occupancy[:, :prefix_slots] = (prefix > 0).to(occupancy.dtype)
        observed = selected_onset_rows(occupancy, beats).to(self._music_density.dtype)
        density = self._music_density
        scoped = plan[..., LABELS.index('Jumpstream')] > .5
        reference_rows = torch.where(scoped, (1 - mix) * density[..., 1] + mix * observed, density[..., 1])
        # Only generation_count_choice reads this field; memory was conditioned
        # before onset scoring using the original tensor passed to generate().
        self._music_density = torch.stack((density[..., 0], reference_rows), dim=-1)
        fresh = torch.arange(logits.shape[1], device=logits.device) >= prefix_slots
        ids = torch.floor(beats[0] + 1e-7).long()
        detail = []
        for beat in ids[fresh].unique(sorted=True):
            selected = ids == beat
            detail.append({'beat': int(beat), 'planned_rows': float(density[0, selected, 1].mean()),
                'selected_music_rows': float(observed[0, selected].mean()),
                'count_reference_rows': float(reference_rows[0, selected].mean())})
        self.count_density_windows.append({'prefix_slots': prefix_slots, 'beats': detail})
        return logits

    def generate(self, *args, **kwargs):
        self._count_density_prefix = kwargs.get('prefix_masks')
        try:
            return original_generate(*args, **kwargs)
        finally:
            self._count_density_prefix = None

    # Weak bindings avoid retaining CUDA models through instance-hook cycles.
    model.generation_onset_logits = partial(onsets, proxy(model))
    model.generate = partial(generate, proxy(model))
    return dict(model.count_density_reference)
