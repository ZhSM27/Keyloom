from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from ..arrangement_model import ArrangementPlanner, ArrangementPlannerConfig
from ..features import AcousticConfig, decode_waveform, hash_file, logmel_features
from ..local_rhythm import OFFSETS_MS, rhythm_descriptors
from ..role_style_plan import ROLE_STYLE_VERSION, RoleStylePolicy, active_style_plan, validate_style_selection
from ..density_coordination import COORDINATION_OPTIONS, COORDINATION_VERSION, validate_coordination, density_execution_trace
from ..density_runtime import infer_density, resolve_density_model
from ..hierarchical_planner import (HIERARCHY_VERSION, validate_structure_id,
    section_phrase_structure, shape_hierarchical_density, json_hierarchy)
from ..generator_density import (BUDGET_VERSION, validate_budget, generator_capabilities,
    prepare_density_budget, native_density_condition, ExecutionLedger, execution_diagnostics)
from ..role_planner_runtime import infer_arrangement, resolve_arrangement_model
from ..rice_data import window_grid_from_section
from ..segment_labels import LABELS
from ..timing import BeatClock, TimingSection


from .config import (WEB_ROOT, PLANNER_CHECKPOINT, PLAN_ARTIFACT, PLAN_ANALYSIS,
    PLANNER_NORMALIZATION, GENERATOR_NORMALIZATION, STYLE_STATISTICS, DEFAULT_MUSICFM_ASSETS,
    MUSICFM_REQUIRED_ASSETS, OUTPUT_ROOT, MAX_UPLOAD_BYTES, ALLOWED_EXTENSIONS,
    BAR_BEATS, CANDIDATES_PER_BEAT, STUDIO_RUNTIME_VERSION, STUDIO_PORT, STUDIO_URL)
from .registry import resolve_generator_model, _read_json
from .loader import _generated_plan_model
from .contracts import PLAN_SCHEMA, validate_plan, validate_settings, option_id

def pool_feature_rows(frames: np.ndarray, starts_ms: np.ndarray, ends_ms: np.ndarray, hop_ms: float) -> np.ndarray:
    values = np.asarray(frames)
    if values.ndim != 2 or not len(values) or hop_ms <= 0:
        raise ValueError("Feature pooling requires nonempty time-major features and a positive frame hop.")
    starts = np.asarray(starts_ms, dtype=np.float64)
    ends = np.asarray(ends_ms, dtype=np.float64)
    result = np.zeros((len(starts), values.shape[1]), dtype=np.float32)
    for index, (start, end) in enumerate(zip(starts, ends)):
        lower = max(0, math.floor(start / hop_ms))
        upper = min(len(values), math.ceil(end / hop_ms))
        if upper > lower:
            result[index] = values[lower:upper].astype(np.float32, copy=False).mean(axis=0)
    return result


def sample_clamped(frames: np.ndarray, times_ms: np.ndarray, hop_ms: float) -> np.ndarray:
    values = np.asarray(frames)
    times = np.asarray(times_ms, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(times).all() or hop_ms <= 0:
        raise ValueError("Sampling requires finite timestamps and nonempty time-major features.")
    coordinate = np.clip(times / hop_ms, 0.0, len(values) - 1)
    lower = np.floor(coordinate).astype(np.int64)
    upper = np.minimum(lower + 1, len(values) - 1)
    alpha = (coordinate - lower)[..., None].astype(np.float32)
    return values[lower].astype(np.float32) * (1 - alpha) + values[upper].astype(np.float32) * alpha


def _normalize_features(values: np.ndarray, statistics: dict, name: str) -> np.ndarray:
    mean = np.asarray(statistics[name]["mean"], dtype=np.float32)
    std = np.asarray(statistics[name]["std"], dtype=np.float32)
    result = (np.asarray(values, dtype=np.float32) - mean) / std
    if result.shape[-1] != len(mean) or not np.isfinite(result).all():
        raise ValueError(f"The {name} audio features do not match the frozen training normalizer.")
    return result.astype(np.float32, copy=False)


def _timing_inputs(bpm: float, first_beat_ms: float, duration_ms: float):
    beat_length = 60_000.0 / bpm
    bar_count = max(1, math.ceil(max(0.0, duration_ms - first_beat_ms) / (BAR_BEATS * beat_length)))
    if bar_count > 906:
        raise ValueError("This chart is longer than the Planner's supported 906 bars.")
    starts = first_beat_ms + np.arange(bar_count, dtype=np.float64) * BAR_BEATS * beat_length
    ends = starts + BAR_BEATS * beat_length
    return TimingSection(float(first_beat_ms), float(beat_length), 4), starts, ends


def _planner_phase(bar_count: int, beat_length_ms: float) -> np.ndarray:
    bars = np.arange(bar_count, dtype=np.float32)
    return np.stack(
        (
            np.sin(bars % 4 * np.pi / 2),
            np.cos(bars % 4 * np.pi / 2),
            np.sin(bars % 8 * np.pi / 4),
            np.cos(bars % 8 * np.pi / 4),
            np.full(bar_count, beat_length_ms / 1000, dtype=np.float32),
            (bars == 0).astype(np.float32),
            bars / max(1, bar_count),
        ),
        axis=-1,
    ).astype(np.float32)


def _generate_song(audio_path: Path, settings: dict, progress, *, device="cuda") -> dict:
    import torch

    job_output = audio_path.parent

    from ..cadence_generator import CadenceGenerator

    from ..lane_control import LaneControlConfig, LaneControlState, absolute_beats
    from ..music_features import MusicFeatureConfig, build_encoder, extract_embeddings
    from ..rhythm_decode import RhythmDecodedGenerator, configure_rhythm_decoding
    from ..segment_labels import LABELS as PLAN_LABELS
    from ..style_sequence import StyleSequenceGenerator
    from ..playstyle_control import DECODER_VERSION, studio_playstyle_controls
    from ..tiny_data import STYLE_KEYS, SEQUENCE_LENGTH
    from ..timing import BeatClock

    if device not in ('cpu', 'cuda'):
        raise ValueError('Offline generation supports explicit cpu/cuda devices.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("GPU inference is unavailable. Install a CUDA-enabled PyTorch build or select CPU offline generation.")
    from ..catalog import verify_shared_resources
    verify_shared_resources()
    generator_model = resolve_generator_model(settings.get("model_id"))
    density_model_id = option_id("planner/density", settings.get("density_model_id", "fixed"))
    density_model = resolve_density_model(density_model_id, PLANNER_CHECKPOINT)
    arrangement_model_id = option_id("planner/role", settings.get("arrangement_model_id", "rule"))
    arrangement_model = resolve_arrangement_model(arrangement_model_id)
    coordination_id = validate_coordination(settings.get('coordination_id','legacy'))
    structure_id = validate_structure_id(settings.get('structure_id', 'legacy'))
    budget_id = validate_budget(settings.get('budget_id', 'legacy'), for_studio=True,
                                generator_id=generator_model.model_id,
                                main_style=settings.get('main_style'))
    validate_settings(settings)
    bpm = float(settings["bpm"])
    first_beat_ms = float(settings["first_beat_ms"])
    target_star = float(settings["target_star"])
    primary_label = str(settings["main_style"])
    primary_label, secondary_labels = validate_style_selection(
        primary_label, settings.get("secondary_styles", settings.get("secondary_style"))
    )
    if not 30 <= bpm <= 400 or not -3000 <= first_beat_ms <= 60_000 or not 2 <= target_star <= 9 or primary_label not in PLAN_LABELS:
        raise ValueError("BPM, first-beat offset, difficulty, or main style is outside the supported range.")
    title = str(settings.get("title", audio_path.stem)).strip()[:120] or audio_path.stem
    artist = str(settings.get("artist", "Unknown artist")).strip()[:120] or "Unknown artist"
    section, bar_starts, bar_ends = _timing_inputs(bpm, first_beat_ms, 1.0)
    progress("decode", 0.04, "读取并校验音频")
    waveform, decoding = decode_waveform(audio_path, AcousticConfig())
    duration_ms = float(decoding["decoded_duration_seconds"] * 1000)
    if duration_ms < 5_000:
        raise ValueError("音频短于 5 秒，无法生成稳定的 4K 谱面。")
    section, bar_starts, bar_ends = _timing_inputs(bpm, first_beat_ms, duration_ms)
    progress("features", 0.11, "提取节奏与音乐特征")
    acoustic = logmel_features(waveform, AcousticConfig(), device=device)
    from ..local_rhythm import rhythm_descriptors

    rhythm = rhythm_descriptors(acoustic)
    music_model, _ = build_encoder(DEFAULT_MUSICFM_ASSETS, device=device)
    music_config = MusicFeatureConfig()
    music, performance = extract_embeddings(music_model, waveform, music_config, device=device)
    del music_model
    torch.cuda.empty_cache()
    recipe = _read_json(GENERATOR_NORMALIZATION)
    local_statistics = recipe["normalization"]
    arrangement_metadata = _read_json(PLANNER_NORMALIZATION)
    planner_statistics = arrangement_metadata["normalization"]
    raw_music = pool_feature_rows(music, bar_starts, bar_ends, 40.0)
    raw_acoustic = pool_feature_rows(acoustic, bar_starts, bar_ends, 5.0)
    raw_rhythm = pool_feature_rows(rhythm, bar_starts, bar_ends, 5.0)
    profile = np.zeros(len(PLAN_LABELS), dtype=np.float32)
    profile[PLAN_LABELS.index(primary_label)] = 1.0
    for secondary_label in secondary_labels:
        profile[PLAN_LABELS.index(secondary_label)] = 0.35 / max(1, len(secondary_labels))
    planner_condition = np.concatenate(([(target_star - 2.0) / 7.0], profile, [0.0])).astype(np.float32)
    phase = _planner_phase(len(bar_starts), section.beat_length_ms)
    progress("planner", 0.31, "Planner 正在规划全曲段落")
    planner_checkpoint = torch.load(PLANNER_CHECKPOINT, map_location="cpu", weights_only=True)
    planner = ArrangementPlanner(ArrangementPlannerConfig(**planner_checkpoint["model_config"])).to(device).eval()
    planner.load_state_dict(planner_checkpoint["state_dict"], strict=True)
    plan_artifact = np.load(PLAN_ARTIFACT, allow_pickle=False)
    thresholds = np.asarray(plan_artifact["thresholds"], dtype=np.float32)
    compiler_config = _read_json(PLAN_ANALYSIS)["config"]
    del plan_artifact, planner_checkpoint
    planner_inputs = [
        _normalize_features(raw_music, planner_statistics, "music"),
        _normalize_features(raw_acoustic, planner_statistics, "acoustic"),
        _normalize_features(raw_rhythm, planner_statistics, "rhythm"),
    ]
    with torch.inference_mode():
        output = planner(
            torch.from_numpy(planner_inputs[0][None]).to(device),
            torch.from_numpy(planner_inputs[1][None]).to(device),
            torch.from_numpy(planner_inputs[2][None]).to(device),
            torch.from_numpy(phase[None]).to(device),
            torch.from_numpy(np.repeat(planner_condition[None, None], len(bar_starts), axis=1)).to(device),
            torch.zeros((1, len(bar_starts)), dtype=torch.bool, device=device),
            return_memory=density_model is not None,
        )
    density = torch.expm1(output["density_log"].float().clamp(0, 4.55))[0].cpu().numpy()
    pattern_probability = output["pattern_logits"].float().sigmoid()[0].cpu().numpy()
    planner_memory = output["memory"][0].float().cpu().numpy() if density_model is not None else None
    del planner, output
    torch.cuda.empty_cache()
    target_notes = density[:, 0].clip(0, 24)
    target_rows = density[:, 1].clip(0, 12)
    owner_scores, style_evidence = None, None
    if arrangement_model is not None:
        frozen_quiet = (target_notes <= .35) & (target_rows <= .20)
        owner_scores, style_evidence = infer_arrangement(arrangement_model, raw_music, raw_acoustic, raw_rhythm,
                                                        phase, target_star, primary_label, secondary_labels, frozen_quiet, device=device)
    plan = active_style_plan(pattern_probability, target_notes, target_rows, thresholds, compiler_config,
                             primary_label, secondary_labels, music=raw_music, acoustic=raw_acoustic, rhythm=raw_rhythm,
                             owner_scores=owner_scores, style_evidence=style_evidence)
    base_density = np.stack((target_notes, target_rows), axis=-1)
    beat_density = np.repeat(base_density[:, None], 4, axis=1)
    if density_model is not None:
        from ..phrase_quality import audio_beat_summary
        beat_activity = np.asarray([[audio_beat_summary(rhythm, start + beat * section.beat_length_ms, section.beat_length_ms)
                                     for beat in range(4)] for start in bar_starts], dtype=np.float32)
        beat_density = infer_density(density_model, PLANNER_CHECKPOINT, planner_memory, base_density,
                                     plan["segment_ids"], plan["quiet"].astype(bool), beat_activity, device=device)
        target_notes, target_rows = beat_density.mean(axis=1).T
        plan["target_note_budget"], plan["target_row_budget"] = target_notes, target_rows
    if coordination_id == 'coordinated':
        # Keep the frozen Planner/adapter context and silence fixed. Coordinate
        # final roles against the density actually handed to the generator.
        plan = active_style_plan(pattern_probability,base_density[:,0],base_density[:,1],thresholds,compiler_config,
            primary_label,secondary_labels,policy=RoleStylePolicy(energy_density_weight=2.),
            music=raw_music,acoustic=raw_acoustic,rhythm=raw_rhythm,owner_scores=owner_scores,style_evidence=style_evidence,
            density_reference=beat_density.mean(axis=1))
        plan['target_note_budget'],plan['target_row_budget'] = target_notes,target_rows
    unshaped_beat_density = beat_density.copy()
    hierarchy_report = None
    if structure_id == 'hierarchical':
        hierarchy = section_phrase_structure(plan['quiet'], music=raw_music, acoustic=raw_acoustic, rhythm=raw_rhythm)
        beat_density, density_shape = shape_hierarchical_density(beat_density, hierarchy)
        target_notes, target_rows = beat_density.mean(1).T
        plan = active_style_plan(pattern_probability, base_density[:, 0], base_density[:, 1], thresholds, compiler_config,
            primary_label, secondary_labels, policy=RoleStylePolicy(energy_density_weight=2. if coordination_id == 'coordinated' else 0.),
            music=raw_music, acoustic=raw_acoustic, rhythm=raw_rhythm, owner_scores=owner_scores, style_evidence=style_evidence,
            density_reference=beat_density.mean(1), structure_override=hierarchy)
        plan['target_note_budget'], plan['target_row_budget'] = target_notes, target_rows
        hierarchy_report = {'version': HIERARCHY_VERSION, 'structure': json_hierarchy(plan['hierarchy']),
            'density': density_shape, 'adapter_context': 'frozen_legacy_segments',
            'role_network_context': 'frozen_legacy_segments', 'new_training': False}
        (job_output / 'hierarchy.json').write_text(json.dumps(hierarchy_report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    validate_plan(plan, beat_density, len(bar_starts))
    model = _generated_plan_model(generator_model.checkpoint, device=device)
    effective_inference_policy = model.checkpoint_inference_policy
    count_density_policy = None
    if budget_id == 'count-reference':
        from ..count_density_budget import enable_count_density_reference
        count_density_policy = enable_count_density_reference(model, enabled=primary_label == 'Jumpstream')
        effective_inference_policy = {'checkpoint_policy': model.checkpoint_inference_policy,
                                      'runtime_count_density_reference': count_density_policy}
    inference_policy_sha256 = hashlib.sha256(json.dumps(effective_inference_policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    capabilities = generator_capabilities(model, {
        'generator_id': generator_model.model_id, 'checkpoint_sha256': generator_model.checkpoint_sha256,
        'inference_policy_sha256': inference_policy_sha256, 'bpm': bpm})
    generator_density, density_budget = prepare_density_budget(
        beat_density, plan['style_ids'], plan['quiet'], capabilities, budget_id)
    effective_density = native_density_condition(model, generator_density, plan['patterns'])
    # Preserve the music objective and original density arrays for comparison.
    # Explicit research structure/budget options can change neural conditions;
    # unshaped_beat_density keeps the original musical objective inspectable.
    del planner_memory
    np.savez_compressed(job_output / "planner-inputs.npz", probabilities=pattern_probability,
                        notes=target_notes, rows=target_rows, thresholds=thresholds,
                        music=raw_music, acoustic=raw_acoustic, rhythm=raw_rhythm,
                        base_density=base_density, beat_density=beat_density, generator_density=generator_density,
                        unshaped_beat_density=unshaped_beat_density,
                        owner_scores=pattern_probability if owner_scores is None else owner_scores,
                        style_evidence=pattern_probability if style_evidence is None else style_evidence)
    arrangement = {"schema_version": ROLE_STYLE_VERSION, "role_source": plan["role_source"],
                   "structure_id": structure_id, "hierarchy_version": HIERARCHY_VERSION if hierarchy_report else None,
                   "sections": plan.get('sections', []),
                   "planner_checkpoint": "planner/arrangement-r1.pt", "planner_sha256": hash_file(PLANNER_CHECKPOINT),
                   "compiler_config": compiler_config, "style_policy": plan["style_policy"],
                   "structure_policy": plan["structure_policy"], "segments": plan["segments"],
                   "unused_secondary_styles": plan["unused_secondary_styles"],
                   "density_policy": density_model_id,
                   "budget_id": budget_id, "budget_version": BUDGET_VERSION,
                   "coordination_id": coordination_id, "coordination_version": COORDINATION_VERSION,
                   "density_checkpoint": density_model["checkpoint"] if density_model else None,
                   "density_sha256": density_model["sha256"] if density_model else None,
                   "arrangement_model_id": arrangement_model_id,
                   "arrangement_checkpoint": arrangement_model["checkpoint"] if arrangement_model else None,
                   "arrangement_sha256": arrangement_model["sha256"] if arrangement_model else None,
                   "boundary": "Audio structure proxies; not verified verse/chorus labels or final chart adherence."}
    (job_output / "arrangement.json").write_text(json.dumps(arrangement, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    timing_section = TimingSection(section.offset_ms, section.beat_length_ms, 4)
    clock = BeatClock([timing_section])
    from .export import beat_plan_window, controlled_window_starts, section_budget_schedule, write_full_osu
    from .export import safe_name
    from ..main_style import STYLE_NAMES as GENERATOR_STYLE_NAMES

    if len(GENERATOR_STYLE_NAMES) - 1 != len(PLAN_LABELS):
        raise ValueError("The Planner and local generator use different 4K style vocabularies.")
    beat_counts = max(0, math.floor((duration_ms - first_beat_ms) / section.beat_length_ms))
    starts = controlled_window_starts(beat_counts, context_beats=2) if beat_counts >= 8 else [0]
    plan_lookup = {}
    for bar in range(len(plan["patterns"])):
        plan_lookup[(0, bar)] = {
            "pattern": plan["patterns"][bar],
            "rows": float(plan["target_row_budget"][bar]),
            "notes": float(plan["target_note_budget"][bar]),
            "quiet": bool(plan["quiet"][bar]),
        }
    pattern_lookup = {key: value["pattern"] for key, value in plan_lookup.items()}
    beat_budget = section_budget_schedule(plan_lookup, 0, max(8, beat_counts), 1.0)
    bar_plan = [
        {
            "bar": index + 1,
            "start_ms": round(float(bar_starts[index]), 1),
            "end_ms": round(float(min(bar_ends[index], duration_ms)), 1),
            "style": None if plan["quiet"][index] else LABELS[int(plan["style_ids"][index]) - 1],
            "requested_style": None if plan["quiet"][index] else LABELS[int(plan.get("owner_style_ids", plan["style_ids"])[index]) - 1],
            "role": str(plan["roles"][index]),
            "segment_id": int(plan["segment_ids"][index]),
            "repeat_group": int(plan["repeat_group_ids"][index]),
            "boundary_confidence": round(float(plan["boundary_scores"][index]), 4),
            **({'section_id': int(plan['section_ids'][index]), 'phrase_id': int(plan['phrase_ids'][index]),
                'phrase_function': next((p['function_cue'] for p in plan['segments'] if p['id'] == int(plan['phrase_ids'][index])), 'quiet')}
               if hierarchy_report else {}),
            "quiet": bool(plan["quiet"][index]),
            "notes_per_beat": round(float(target_notes[index]), 2),
            "rows_per_beat": round(float(target_rows[index]), 2),
        }
        for index in range(len(bar_starts))
    ]
    # Match MainStyleCorpusDataset: legacy global style dimensions stay zero;
    # generator style intent is supplied by the per-bar local plan.
    condition = np.zeros((1, 1 + len(STYLE_KEYS)), dtype=np.float32)
    condition[0, 0] = target_star / 10
    selected_styles = {primary_label, *secondary_labels}
    # R1 probability calibration damaged continuous JS/HS structures. Keep the
    # learned onset decisions unchanged; only coordinate the selected roles.
    model.density_coordination = False
    model_architecture=model.configuration().get('architecture')
    generator_runtime_policy={name:getattr(model,name,None) for name in (
        'expression_decoding','stream_repetition_strength','count_policy','count_temperature','count_seed',
        'music_soft_style_prior','music_style_prior_penalty','load_adapter_enabled','hand_load_strength','longjack_sampling_enabled','natural_release_enabled','natural_release_schema','cross_double_repeat_enabled','cross_double_schema','music_chord_support_enabled','music_chord_support_schema','learned_music_counts','learned_count_lookahead_enabled')}
    generator_runtime_policy.update({name:model.configuration().get(name) for name in ('short_pressure_schema','pressure_windows_beats','ordered_history_bins','protected_cut_layout_bank')})
    from ..natural_generator import NaturalLayoutGenerator
    from ..motion_state import MotionHistory
    motion_state = MotionHistory() if isinstance(model,NaturalLayoutGenerator) else None
    style_limits = tuple(tuple(row) for row in _read_json(STYLE_STATISTICS)['style_run_limits'])
    from ..playstyle_control import PlaystyleControlConfig
    lane_state, phrase_state = studio_playstyle_controls(device, selected_styles, style_limits,
        PlaystyleControlConfig(cut_phrase_context=bool(getattr(model,'expression_decoding',False))))
    previous_end_ms = None
    previous_start = None
    history_masks: dict[int, int] = {}
    notes = set()
    generated_rows = 0
    music_count_summary=None
    density_alignment_summary=None
    expression_summary=None
    music_planned_notes=0
    music_plan_digest=hashlib.sha256()
    music_planned_rows=0
    execution_ledger = ExecutionLedger()
    normalized_generator = {
        "music": _normalize_features(music, local_statistics, "music"),
        "acoustic": _normalize_features(acoustic, local_statistics, "acoustic"),
        "rhythm": rhythm,
    }
    del waveform
    for window_index, start_beat in enumerate(starts):
        progress("generation", 0.56 + 0.37 * window_index / max(1, len(starts)), f"局部生成器 · 窗口 {window_index + 1}/{len(starts)}")
        times, phase_window = window_grid_from_section(vars(section), int(start_beat))
        times = np.asarray(times, dtype=np.float64)
        phase_window = np.asarray(phase_window, dtype=np.float32)
        if times.ndim == 3:
            times = times[0]
        if phase_window.ndim == 3:
            phase_window = phase_window[0]
        acoustic_window = sample_clamped(normalized_generator["acoustic"], times, 5.0)
        music_window = sample_clamped(normalized_generator["music"], times, 40.0)
        local_raw = sample_clamped(normalized_generator["rhythm"], times[:, None] + np.asarray(OFFSETS_MS, dtype=np.float64)[None, :], 5.0).reshape(len(times), -1)
        local_window = _normalize_features(local_raw, local_statistics, "local")
        if phase_window.shape[1] != 5:
            raise ValueError("The local 8-beat phase features do not match the trained generator.")
        plan_window = beat_plan_window(pattern_lookup, 0, int(start_beat))
        batch = {
            "music": torch.from_numpy(music_window[None]).to(device),
            "acoustic": torch.from_numpy(acoustic_window[None]).to(device),
            "local": torch.from_numpy(local_window[None]).to(device),
            "phase": torch.from_numpy(phase_window[None]).to(device),
            "condition": torch.from_numpy(condition).to(device),
            "plan": torch.from_numpy(plan_window[None]).to(device),
        }
        if hasattr(model, "density_projection"):
            bar_indices = (np.arange(start_beat, start_beat + 8) // 4).clip(0, len(bar_plan) - 1)
            per_beat = generator_density[bar_indices, np.arange(start_beat, start_beat + 8) % 4].astype(np.float32)
            per_slot = np.repeat(per_beat, CANDIDATES_PER_BEAT, axis=0)
            batch["density"] = torch.from_numpy(per_slot[None]).to(device)
        prefix_slots = 0
        if previous_start is not None and previous_end_ms is not None and start_beat > previous_start:
            prefix_slots = int(np.searchsorted(times, previous_end_ms - 1e-5))
            prefix = np.asarray([history_masks.get(int(round(value)), 0) for value in times[:prefix_slots]], dtype=np.int64)
        else:
            prefix = np.empty(0, dtype=np.int64)
        if prefix_slots >= len(times):
            raise ValueError("Continuation window has no new positions to generate.")
        beats = absolute_beats(clock, times)
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                count_plan_enabled=True,
                prefix_masks=torch.from_numpy(prefix[None]).to(device),
                candidate_beats=torch.from_numpy(beats[None]).to(device),
                lane_state=lane_state,
                phrase_state=phrase_state,
                fast_empty=True,
                **({'motion_state':motion_state} if motion_state is not None else {}),
            )
        masks = generated["masks"][0].detach().cpu().numpy()
        musical_counts=generated['music_counts'][0].detach().cpu().numpy()
        prepared_counts=generated['counts'][0].detach().cpu().numpy()
        if 'density_alignment' in generated:
            detail=generated['density_alignment']
            if density_alignment_summary is None:density_alignment_summary=dict(detail)
            else:
                for key in ('calibrated_beats','bounded_beats','expected_rows_before','expected_rows_after'):
                    density_alignment_summary[key]+=detail[key]
        if 'phrase_expression' in generated:
            if expression_summary is None:expression_summary=dict(generated['phrase_expression'])
            else:
                for key in ('viability_scored_rows','stream_scored_rows','music_sampled_rows','music_nonmodal_rows','sampling_log_regret_sum','density_guarded_rows','density_infeasible_rows','protected_count_changes','protected_music_rows','hand_pressure_scored_rows','supported_double_restorations','learned_music_count_changed_rows','learned_music_count_net_notes','learned_count_lookahead_changed_rows'):
                    expression_summary[key]=expression_summary.get(key,0)+generated['phrase_expression'].get(key,0)
        if 'music_count_control' in generated:
            detail=generated['music_count_control']
            if music_count_summary is None:music_count_summary=dict(detail)
            else:
                for key in ('elective_changed_rows','elective_removed_notes','elective_added_notes',
                            'adapter_argmax_changed_rows','strong_count_changed_rows','final_changed_rows','final_net_notes_delta'):
                    if key not in detail:continue
                    music_count_summary[key]+=detail[key]
                music_count_summary['maximum_log_prior_regret']=max(music_count_summary['maximum_log_prior_regret'],detail['maximum_log_prior_regret'])
        for position, (time_ms, mask) in enumerate(zip(times, masks)):
            if position < prefix_slots:
                continue
            absolute_time = int(round(time_ms))
            history_masks[absolute_time] = int(mask)
            if time_ms < 0 or time_ms > duration_ms:
                continue
            execution_ledger.record(absolute_time, musical_counts[position], prepared_counts[position], mask)
            music_plan_digest.update(f'{absolute_time}:{int(musical_counts[position])};'.encode('ascii'))
            music_planned_notes+=int(musical_counts[position])
            music_planned_rows+=int(musical_counts[position]>0)
            if int(mask):
                generated_rows += 1
            for column in range(4):
                if int(mask) & (1 << column):
                    notes.add((absolute_time, column))
        previous_start = int(start_beat)
        previous_end_ms = float(times[0] + 8 * section.beat_length_ms)
        del batch, generated, masks, acoustic_window, music_window, local_window, local_raw
    count_density_windows = getattr(model, 'count_density_windows', [])
    del model
    if count_density_policy is not None:
        import gc
        gc.collect()
    torch.cuda.empty_cache()
    if len(notes) < 16:
        raise ValueError("The preview generated fewer than 16 notes. Check the timing alignment or choose a denser main style.")
    progress("packaging", 0.95, "校验并打包 osu!mania 4K 谱面")
    audio_suffix = audio_path.suffix.lower()
    if audio_suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("The audio format is no longer supported.")
    job_output = audio_path.parent
    style_suffix = primary_label if not secondary_labels else f"{primary_label} + {' + '.join(secondary_labels)}"
    package_name = f"{safe_name(title)} - {target_star:.1f}★ {safe_name(style_suffix)}.osz"
    package_path = job_output / package_name
    audio_name = f"audio{audio_suffix}"
    source_chart = {"metadata": {"Title": title, "Artist": artist, "Version": f"{target_star:.1f}★ {primary_label}"}}
    osu_path = job_output / "generated.osu"
    write_full_osu(osu_path, source_chart, sorted(notes), clock, audio_name, "Keyloom generator/" + generator_model.model_id)
    exported_clock = BeatClock([TimingSection(section.offset_ms, section.beat_length_ms, 4)])
    if exported_clock.sections[0].offset_ms != first_beat_ms or not osu_path.is_file():
        raise ValueError("The exported timing section failed validation.")
    from ..timing import read_chart
    verify_clock, verify_notes = read_chart(osu_path)
    if len(verify_notes) != len(notes) or len(verify_clock.sections) != 1:
        raise ValueError("The generated `.osu` did not round-trip through the project chart parser.")
    from ..playstyle_audit import audit_minijack_export
    minijack_audit = audit_minijack_export(verify_clock, verify_notes, bar_plan)
    if not minijack_audit["passed"]:
        raise ValueError("导出短叠未通过同列三连或同手五行限制，请检查当前解码版本与局部计划。")
    import rosu_pp_py

    star_rating = float(rosu_pp_py.Difficulty().calculate(rosu_pp_py.Beatmap(path=str(osu_path))).stars)
    density_trace = density_execution_trace(beat_density,bar_plan,[note.time_ms for note in verify_notes],
                                            raw_rhythm[:,32:36].mean(-1),section.beat_length_ms)
    original_density_summary = None
    if hierarchy_report:
        original_density_summary = density_execution_trace(unshaped_beat_density, bar_plan,
            [note.time_ms for note in verify_notes], raw_rhythm[:, 32:36].mean(-1), section.beat_length_ms)['summary']
        density_trace['unshaped_reference_summary'] = original_density_summary
        for index, bar in enumerate(density_trace['bars']):
            bar['unshaped_target_beat_density'] = unshaped_beat_density[index].tolist()
    execution_trace = execution_diagnostics(beat_density, generator_density, effective_density, bar_plan,
        section.beat_length_ms, execution_ledger, [(note.time_ms, note.column) for note in verify_notes], density_budget)
    execution_trace['count_density_reference'] = {'policy': count_density_policy, 'windows': count_density_windows}
    model_identity = {"generator": f"generator/{generator_model.model_id}",
                      "planner": "planner/arrangement/structured-v2",
                      "arrangement": f"planner/role/{arrangement_model_id}",
                      "density": f"planner/density/{density_model_id}",
                      "checkpoint_sha256": generator_model.checkpoint_sha256}
    export_provenance = {"model_identity": model_identity, "plan_schema": PLAN_SCHEMA,
        "inference_policy_sha256": inference_policy_sha256, 'version':COORDINATION_VERSION,'settings':{**settings,'coordination_id':coordination_id},
        'source_sha256':hash_file(audio_path),'generator_sha256':generator_model.checkpoint_sha256,
        'planner_sha256':hash_file(PLANNER_CHECKPOINT),'arrangement_sha256':arrangement['arrangement_sha256'],
        'density_sha256':arrangement['density_sha256'],'density_alignment':density_alignment_summary,
        'onset_policy':'preserve_original_model_decisions', 'budget_id': budget_id,
        'budget_version': BUDGET_VERSION, 'execution_diagnostic_version': execution_trace['version']}
    export_provenance['effective_inference_policy'] = effective_inference_policy
    export_provenance.update(structure_id=structure_id, hierarchy_version=HIERARCHY_VERSION if hierarchy_report else None)
    (job_output/'density-execution.json').write_text(json.dumps(execution_trace,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (job_output/'bar-density.json').write_text(json.dumps(density_trace,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (job_output/'actual-bars.json').write_text(json.dumps(bar_plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(package_path, "w") as archive:
        archive.write(osu_path, arcname=osu_path.name, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        archive.write(audio_path, arcname=audio_name, compress_type=zipfile.ZIP_STORED)
        archive.writestr('export-provenance.json',json.dumps(export_provenance,ensure_ascii=False),compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr('bar-density.json',json.dumps(density_trace,ensure_ascii=False),compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr('density-execution.json',json.dumps(execution_trace,ensure_ascii=False),compress_type=zipfile.ZIP_DEFLATED)
        if hierarchy_report:
            archive.writestr('hierarchy.json', json.dumps(hierarchy_report, ensure_ascii=False), compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(package_path) as archive:
        invalid_entry = archive.testzip()
        entries = archive.namelist()
        expected_entries = {osu_path.name, audio_name,'export-provenance.json','bar-density.json','density-execution.json'}
        if hierarchy_report:
            expected_entries.add('hierarchy.json')
        if invalid_entry or set(entries) != expected_entries:
            raise ValueError("The `.osz` package is corrupt or contains unexpected entries.")
        chart_text = archive.read(osu_path.name).decode("utf-8")
        if f"AudioFilename: {audio_name}" not in chart_text or "Mode: 3" not in chart_text or "CircleSize:4" not in chart_text:
            raise ValueError("The `.osz` manifest does not identify a four-key osu!mania chart and audio asset.")
    manifest = {
        "status": "completed",
        "title": title,
        "artist": artist,
        "bpm": round(bpm, 3),
        "first_beat_ms": round(first_beat_ms, 3),
        "target_star": target_star,
        "actual_star": round(star_rating, 4),
        "main_style": primary_label,
        "secondary_styles": list(secondary_labels),
        "main_style_bars": int(np.sum(plan["main_section"])),
        "secondary_style_bars": int(np.sum(plan["secondary_section"])),
        "style_plan_policy": plan["style_policy"],
        "style_plan_role_source": plan["role_source"],
        "arrangement_model_id": arrangement_model_id,
        "structure_id": structure_id, "hierarchy_version": HIERARCHY_VERSION if hierarchy_report else None,
        "hierarchy_sections": len(plan.get('sections', [])),
        "hierarchy_phrases": len(plan['segments']) if hierarchy_report else 0,
        "hierarchy_density": {key: value for key, value in hierarchy_report['density'].items()
                              if key not in ('original_beat_density', 'shaped_beat_density', 'multipliers')} if hierarchy_report else None,
        "arrangement_model_label": arrangement_model["label"] if arrangement_model else "音乐分段规则编排",
        "arrangement_checkpoint": arrangement_model["checkpoint"] if arrangement_model else None,
        "arrangement_sha256": arrangement_model["sha256"] if arrangement_model else None,
        "density_model_id": density_model_id,
        "budget_id": budget_id, "budget_version": BUDGET_VERSION,
        "density_budget": {key: value for key, value in density_budget.items() if key != 'bars'},
        "count_density_reference": count_density_policy,
        "execution_diagnostics": execution_trace['summary'],
        "coordination_id":coordination_id,
        "coordination_label":next(item['label'] for item in COORDINATION_OPTIONS if item['id']==coordination_id),
        "coordination_version":COORDINATION_VERSION,
        "coordination_onset_policy":"preserve_original_model_decisions",
        "density_alignment":density_alignment_summary,"density_execution":density_trace['summary'],
        "density_execution_original_reference": original_density_summary,
        "inference_device": device,
        "density_model_label": density_model["label"] if density_model else "原 Planner 密度",
        "density_checkpoint": density_model["checkpoint"] if density_model else None,
        "density_sha256": density_model["sha256"] if density_model else None,
        "arrangement_segments": plan["segments"],
        "unused_secondary_styles": plan["unused_secondary_styles"],
        "bars": len(bar_plan),
        "notes": len(notes),
        "rows": generated_rows,
        "quiet_bars": int(np.sum(plan["quiet"])),
        "audio_duration_ms": round(duration_ms, 2),
        "upload_sha256": hash_file(audio_path),
        "package_sha256": hash_file(package_path),
        "package_size_bytes": package_path.stat().st_size,
        "model_id": generator_model.model_id,
        "model_label": generator_model.label,
        "model_status": generator_model.status,
        "architecture": model_architecture,
        "export_name": generator_model.export_name,
        "plan_schema": PLAN_SCHEMA,
        "inference_policy_sha256": inference_policy_sha256,
        "model_architecture":model_architecture,
        "music_count_control":music_count_summary,
        "phrase_expression":expression_summary,
        "generation_runtime_version":STUDIO_RUNTIME_VERSION,
        "model_identity": model_identity,
        "generator_runtime_policy":generator_runtime_policy,
        "music_planned_notes":music_planned_notes,
        "music_planned_rows":music_planned_rows,
        "music_plan_sha256":music_plan_digest.hexdigest(),
        "generator_checkpoint": f"generator/{generator_model.model_id}-r{generator_model.revision}.pt",
        "generator_sha256": generator_model.checkpoint_sha256,
        "generator_condition": condition[0].tolist(),
        "decoder_version": DECODER_VERSION,
        "minijack_export_audit": minijack_audit,
        "rhythm_decoding": "learned-grid",
        "budget_policy": "neural",
        "lane_control": lane_state.summary(),
        "phrase_control": getattr(phrase_state,"_natural_release_proxy",phrase_state).summary(),
        "music_feature_seconds_per_audio_second": round(float(performance["audio_seconds_per_inference_second"]), 3),
        "playable_format": "osu!mania 4K (.osz)",
        "timing_note": "BPM and beat zero are user-adjustable estimates; this export uses one constant 4/4 timing section.",
    }
    (job_output / "generation.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        osu_path.unlink()
    except OSError:
        pass
    progress("complete", 1.0, "谱面已生成并通过导出校验", manifest=manifest, filename=package_name, bars=bar_plan, notes_preview=sorted(notes)[:240])
    return manifest
