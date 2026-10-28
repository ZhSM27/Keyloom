"""Pinned upstream MusicFM assets and integrity validation."""
from pathlib import Path
from .paths import ASSET_ROOT
from .features import hash_file as digest_file
DEFAULT_ASSET_ROOT = ASSET_ROOT / "audio/musicfm"

SOURCE_REVISION = "b83ebedb401bcef639b26b05c0c8bee1dc2dfe71"

WEIGHTS_REVISION = "4513b38bc25ad1d227b1980819b9691ba97f4d87"

CONFIG_REVISION = "6b36ef01c6443c67ae7ed0822876d091ab50e4aa"

WEIGHTS_SHA256 = "68392eee13d34c2941b3761934abb6b1e67b2e9df498695bda2ea5c1087d4b96"

WEIGHTS_BYTES = 1316802154

SOURCE_HASHES = {
    "README.md": "e98d747d6219a84f65f57f3775ed21cea800b37662136d99f3d129e06645bf74",
    "model/__init__.py": "75a11da44c802486bc6f65640aa48a730f0f684c5c07a42ba3cd1735eb3fb070",
    "model/musicfm_25hz.py": "9645e51938e7a73689a0792ecb15ee957e0723582c8f040949eadd25036ec804",
    "modules/__init__.py": "75a11da44c802486bc6f65640aa48a730f0f684c5c07a42ba3cd1735eb3fb070",
    "modules/features.py": "aa8e99711f4a4eab522ff89fe86fc27e281e373ed0ae416743736da229f171da",
    "modules/conv.py": "585705a6450db374e8c411034e846491f7f48d4ed0516fbf8d3b84855ef0d7d1",
    "modules/random_quantizer.py": "2ffae390fbf7fefa8a62000ba0490af92ecc001d3b5715fb98f7a1e1578c2aa2",
    "LICENSE": "5684e11c103b652a5fc59a2cc930c4bb63b5d4aa497e8519aaeb147bc4d34877",
}

def asset_specification() -> dict:
    assets = {
        f"source/musicfm/{name}": {"sha256": checksum, "urls": [f"https://raw.githubusercontent.com/minzwon/musicfm/{SOURCE_REVISION}/{name}"]}
        for name, checksum in SOURCE_HASHES.items()
    }
    for name, repository, revision, filename, checksum in (
        ("fma_stats.json", "minzwon/MusicFM", WEIGHTS_REVISION, "fma_stats.json", "5416e468018bae68c6231d4cbb2b11f0d11c04e6437881505ae427a3f8344904"),
        ("conformer-config.json", "facebook/wav2vec2-conformer-rope-large-960h-ft", CONFIG_REVISION, "config.json", "7a63cb5706c9a37483f1973a3c226d54eb504ce15cf62cb52637019540c8a75d"),
    ):
        assets[name] = {"sha256": checksum, "urls": [f"https://{host}/{repository}/raw/{revision}/{filename}" for host in ("huggingface.co",)]}
    assets["pretrained_fma.pt"] = {"sha256": WEIGHTS_SHA256, "bytes": WEIGHTS_BYTES, "urls": [f"https://{host}/minzwon/MusicFM/resolve/{WEIGHTS_REVISION}/pretrained_fma.pt?download=true" for host in ("huggingface.co",)]}
    return assets

def check_asset(path: Path, specification: dict):
    if not path.is_file() or digest_file(path) != specification["sha256"]:
        raise ValueError(f"MusicFM asset is missing or has a different SHA-256: {path}")
    if "bytes" in specification and path.stat().st_size != specification["bytes"]:
        raise ValueError(f"MusicFM asset has a different byte count: {path}")
