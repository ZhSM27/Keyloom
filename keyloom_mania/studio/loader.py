"""Native generator architecture loading and reviewed inference policies."""
from pathlib import Path
from ..features import hash_file
from .config import GENERATOR_CHECKPOINT
from ..catalog import verify_asset

def _generated_plan_model(checkpoint_path: Path = GENERATOR_CHECKPOINT, *, device="cuda"):
    import torch

    from ..expression_generator import ExpressionGenerator
    from ..music_count_natural import MusicCountNaturalGenerator
    from ..count_continuation import CountContinuationGenerator
    from ..main_style import STYLE_NAMES as ALL_STYLE_NAMES
    from ..rhythm_decode import RhythmDecodedGenerator, configure_rhythm_decoding
    from ..tiny_model import TinyModelConfig

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = dict(checkpoint["model_config"])
    architecture = config.pop("architecture", "factorized")
    accepted = {name: value for name, value in config.items() if name in TinyModelConfig.__dataclass_fields__}
    constructors = {"music-count-v1": MusicCountNaturalGenerator, "count-context-v1": CountContinuationGenerator}
    if architecture not in constructors:
        raise ValueError(f"Unsupported generator architecture: {architecture}")
    model_class = constructors[architecture]
    extras={key:config.get(key,False) for key in ('reference_count_fallback','reference_layout_fallback')} if issubclass(model_class,ExpressionGenerator) else {}
    if device not in ('cuda', 'cpu'):
        raise ValueError('Native generator loading supports explicit cuda or cpu devices.')
    model = model_class(TinyModelConfig(**accepted),**extras).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if isinstance(model,ExpressionGenerator):
        policy=checkpoint.get('inference_policy',{})
        model.expression_decoding=policy.get('expression_decoding',False)
        if architecture=='count-context-v1':model.learned_count_lookahead_enabled=policy.get('learned_count_lookahead',False)
        if policy.get('natural_release',False):
            from ..natural_release import enable_natural_release
            enable_natural_release(model,policy.get('natural_release_reference'),policy.get('natural_release_strength',.35))
        if policy.get('cross_double_repeat',False):
            from ..cross_double_control import enable_cross_double_control
            enable_cross_double_control(model,policy.get('natural_release_reference'),policy.get('natural_release_strength',.35),policy.get('cross_double_repeat_strength',2.5))
        if policy.get('music_chord_support',False):
            from ..music_chord_support import enable_music_chord_support
            enable_music_chord_support(model)
        if policy.get('longjack_sampling',False):
            from ..phrase_decoding import enable_longjack_sampling
            enable_longjack_sampling(model)
        from ..short_hand_dynamics import SCHEMA
        if config.get('short_pressure_schema')!=SCHEMA or config.get('pressure_windows_beats')!=[2.,4.] or policy.get('hand_load_strength')!=0. or policy.get('pressure_reranking') is not False:
            raise ValueError('Released generators require the finite 2/4-beat schema and no pressure reranking.')
        model.stream_repetition_strength=policy.get('stream_repetition_strength',0.)
        model.count_policy=policy.get('count_policy','argmax')
        model.count_temperature=policy.get('count_temperature',1.)
        model.count_seed=policy.get('count_seed',20261112)
        expected_reference=policy.get('phrase_reference_sha256')
        if expected_reference and hash_file(verify_asset('generator/phrase-reference.json'))!=expected_reference:
            raise ValueError('The reviewed concrete phrase reference has changed.')
    configure_rhythm_decoding(model, "learned-grid") if isinstance(model, RhythmDecodedGenerator) else None
    if len(ALL_STYLE_NAMES) != 16:
        raise ValueError("Unexpected 4K Planner style vocabulary.")
    model.checkpoint_inference_policy = checkpoint.get("inference_policy", {})
    del checkpoint
    return model
