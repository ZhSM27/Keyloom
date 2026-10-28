from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import importlib.util
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .audio import asset_path
from .features import decode_waveform, hash_file
from .audio_assets import DEFAULT_ASSET_ROOT, SOURCE_REVISION, WEIGHTS_REVISION, WEIGHTS_SHA256, asset_specification, check_asset


SCHEMA_VERSION = "musicfm-fma-context-v1"
SAMPLE_RATE = 24000
FRAME_HOP = 960


@dataclass(frozen=True)
class MusicFeatureConfig:
    core_seconds: int = 12
    context_seconds: int = 2
    layer_index: int = 7
    precision: str = "float32"

    def __post_init__(self):
        if not isinstance(self.core_seconds, int) or not 1 <= self.core_seconds <= 24:
            raise ValueError("MusicFM core length must be an integer from 1 to 24 seconds.")
        if not isinstance(self.context_seconds, int) or not 0 <= self.context_seconds <= 6:
            raise ValueError("MusicFM context must be an integer from 0 to 6 seconds.")
        if not 1 <= self.layer_index <= 12 or self.precision not in ("float32", "bfloat16"):
            raise ValueError("Unsupported MusicFM layer or precision.")


def chunk_plan(sample_count: int, config: MusicFeatureConfig) -> list[dict]:
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("MusicFM chunking requires a positive integer sample count.")
    frame_count = math.ceil(sample_count / FRAME_HOP)
    core_frames = config.core_seconds * 25
    context_frames = config.context_seconds * 25
    result = []
    for core_start in range(0, frame_count, core_frames):
        core_end = min(frame_count, core_start + core_frames)
        input_start = max(0, core_start - context_frames)
        input_end = min(frame_count, core_end + context_frames)
        result.append({"core_start_frame": core_start, "core_end_frame": core_end, "input_start_frame": input_start, "input_end_frame": input_end, "sample_start": input_start * FRAME_HOP, "sample_end": min(sample_count, input_end * FRAME_HOP), "padded_samples": max(2, input_end - input_start) * FRAME_HOP})
    return result


def pool_global(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or not features.shape[0] or not np.isfinite(features).all():
        raise ValueError("Global context requires finite frame embeddings.")
    means = []
    times = []
    for start in range(0, len(features), 25):
        end = min(len(features), start + 25)
        means.append(features[start:end].mean(axis=0, dtype=np.float32))
        times.append((start + end - 1) * 0.5 * FRAME_HOP / SAMPLE_RATE)
    return np.stack(means), np.asarray(times, dtype=np.float64)


def load_source_module(asset_root: Path, filename: str):
    path = asset_root / "source/musicfm/modules" / filename
    specification = asset_specification()[f"source/musicfm/modules/{filename}"]
    check_asset(path, specification)
    module_spec = importlib.util.spec_from_file_location(f"keyloom_mania_musicfm_{path.stem}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def modernize_weight_norm_keys(state: dict) -> dict:
    state = dict(state)
    for legacy, current in (("weight_g", "parametrizations.weight.original0"), ("weight_v", "parametrizations.weight.original1")):
        source = f"conformer.pos_conv_embed.conv.{legacy}"
        target = f"conformer.pos_conv_embed.conv.{current}"
        if source not in state or target in state:
            raise ValueError("Unexpected or ambiguous MusicFM weight-normalization checkpoint layout.")
        state[target] = state.pop(source)
    return state


def build_encoder(asset_root: Path, device: str = "cuda"):
    import torch
    from torch import nn
    from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import Wav2Vec2ConformerConfig, Wav2Vec2ConformerEncoder

    for name, specification in asset_specification().items():
        check_asset(asset_root / name, specification)
    config = Wav2Vec2ConformerConfig.from_json_file(str(asset_root / "conformer-config.json"))
    config.num_hidden_layers = 12
    config.hidden_size = 1024
    statistics = json.loads((asset_root / "fma_stats.json").read_text(encoding="utf-8"))
    mean, std = statistics["melspec_2048_mean"], statistics["melspec_2048_std"]
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("MusicFM normalization statistics must be finite with positive standard deviation.")
    frontend = load_source_module(asset_root, "features.py")
    convolution = load_source_module(asset_root, "conv.py")

    class FrozenMusicFM(nn.Module):
        def __init__(self):
            super().__init__()
            self.preprocessor_melspec_2048 = frontend.MelSTFT(n_fft=2048, hop_length=240, is_db=True)
            self.conv = convolution.Conv2dSubsampling(1, 512, 1024, strides=[2, 2], n_bands=128)
            self.conformer = Wav2Vec2ConformerEncoder(config)

        def forward(self, waveform, layer_index=7):
            with torch.autocast(device_type=waveform.device.type, enabled=False):
                mel = self.preprocessor_melspec_2048(waveform.float())[..., :-1]
                mel = (mel - mean) / std
            hidden = self.conv(mel)
            return self.conformer(hidden, output_hidden_states=True).hidden_states[layer_index]

    with torch.random.fork_rng(devices=[]):
        model = FrozenMusicFM()
    checkpoint = torch.load(asset_root / "pretrained_fma.pt", map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["state_dict"]
    if not state or any(not name.startswith("model.") for name in state):
        raise ValueError("Unexpected MusicFM checkpoint namespace.")
    state = modernize_weight_norm_keys({name[6:]: tensor for name, tensor in state.items()})
    expected = set(model.state_dict())
    extras = set(state) - expected
    allowed_extras = {"cls_token", "linear.weight", "linear.bias", "quantizer_melspec_2048_0.random_projection", "quantizer_melspec_2048_0.codebook"}
    if extras != allowed_extras:
        raise ValueError(f"Unexpected inference-head checkpoint keys: {sorted(extras)}")
    model.load_state_dict({name: tensor for name, tensor in state.items() if name in expected}, strict=True)
    model.requires_grad_(False).eval().to(device)
    return model, {"parameter_count": sum(parameter.numel() for parameter in model.parameters()), "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "omitted_checkpoint_keys": sorted(extras), "strict_trunk_load": True, "weight_norm_adaptation": "legacy weight_g/weight_v renamed to parametrizations.weight.original0/original1 without changing tensor values"}


def precision_context(device: str, precision: str):
    import torch

    if precision == "float32":
        return contextlib.nullcontext()
    if device != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("MusicFM bfloat16 inference requires a supported CUDA device.")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def extract_embeddings(model, waveform: np.ndarray, config: MusicFeatureConfig, device: str = "cuda") -> tuple[np.ndarray, dict]:
    import torch

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("MusicFM expects finite nonempty mono audio at 24 kHz.")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The audio encoder must be frozen and in evaluation mode.")
    plan = chunk_plan(int(waveform.size), config)
    chunks = []
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    with torch.inference_mode():
        for chunk in plan:
            samples = np.zeros(chunk["padded_samples"], dtype=np.float32)
            count = chunk["sample_end"] - chunk["sample_start"]
            samples[:count] = waveform[chunk["sample_start"]:chunk["sample_end"]]
            with precision_context(device, config.precision):
                encoded = model(torch.from_numpy(samples).unsqueeze(0).to(device), config.layer_index)
            expected_frames = chunk["padded_samples"] // FRAME_HOP
            if encoded.ndim != 3 or encoded.shape[0] != 1 or encoded.shape[1] != expected_frames:
                raise ValueError(f"Unexpected MusicFM output shape: {tuple(encoded.shape)}")
            trim_start = chunk["core_start_frame"] - chunk["input_start_frame"]
            trim_end = chunk["core_end_frame"] - chunk["input_start_frame"]
            chunks.append(encoded[0, trim_start:trim_end].float().cpu().numpy().copy())
            del encoded
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    result = np.concatenate(chunks)
    if not np.isfinite(result).all() or result.shape[0] != math.ceil(waveform.size / FRAME_HOP):
        raise ValueError("MusicFM produced invalid or incomplete features.")
    performance = {"inference_seconds": elapsed, "audio_seconds": waveform.size / SAMPLE_RATE, "audio_seconds_per_inference_second": waveform.size / SAMPLE_RATE / elapsed, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None, "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None, "chunk_count": len(plan), "chunks": plan}
    return result, performance
