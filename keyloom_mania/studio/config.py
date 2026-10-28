"""Portable workbench settings and the public model bundle layout."""
from pathlib import Path
from ..paths import APP_HOME, ASSET_ROOT, OUTPUT_ROOT

WEB_ROOT = Path(__file__).resolve().parents[1] / "generation_studio_web"
PLANNER_ROOT = ASSET_ROOT / "planner"
PLANNER_CHECKPOINT = PLANNER_ROOT / "arrangement-r1.pt"
PLAN_ARTIFACT = PLANNER_ROOT / "thresholds.npz"
PLAN_ANALYSIS = PLANNER_ROOT / "compiler.json"
PLANNER_NORMALIZATION = PLANNER_ROOT / "normalization.json"
GENERATOR_NORMALIZATION = ASSET_ROOT / "generator/normalization.json"
STYLE_STATISTICS = ASSET_ROOT / "generator/style-statistics.json"
PHRASE_REFERENCE = ASSET_ROOT / "generator/phrase-reference.json"
DEFAULT_GENERATOR_MODEL_ID = "music-base"
GENERATOR_CHECKPOINT = ASSET_ROOT / "generator/music-base-r1.pt"
DEFAULT_MUSICFM_ASSETS = ASSET_ROOT / "audio/musicfm"
MUSICFM_REQUIRED_ASSETS = tuple(DEFAULT_MUSICFM_ASSETS / name for name in (
    "pretrained_fma.pt", "conformer-config.json", "fma_stats.json",
    "source/musicfm/modules/features.py", "source/musicfm/modules/conv.py",
))
MAX_UPLOAD_BYTES = 180 * 1024 * 1024
ALLOWED_EXTENSIONS = frozenset({".mp3", ".ogg", ".wav", ".flac", ".m4a"})
BAR_BEATS = 4
CANDIDATES_PER_BEAT = 24
STUDIO_RUNTIME_VERSION = "keyloom-0.1.0"
STUDIO_PORT = 8788
STUDIO_URL = f"http://127.0.0.1:{STUDIO_PORT}/"
