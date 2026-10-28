"""Read-only release catalog binding weights to their inference resources."""
import hashlib
import json
from pathlib import Path, PurePosixPath
from .paths import ASSET_ROOT

CATALOG_PATH = Path(__file__).with_name("resources") / "catalog.json"


def catalog():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def asset_path(name, root=None):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError("Asset paths must be portable relative paths.")
    root = ASSET_ROOT if root is None else Path(root)
    resolved = (root / name).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Asset path leaves the configured bundle.")
    return resolved


def verify_asset(name, root=None):
    expected = catalog()["files"].get(name)
    if expected is None:
        raise ValueError(f"Unknown release asset: {name}")
    path = asset_path(name, root)
    if not path.is_file():
        raise ValueError(f"Missing model asset: {name}. Install the release model bundle.")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected["sha256"]:
        raise ValueError(f"Model asset checksum mismatch: {name}")
    return path


def verify_shared_resources():
    for name in catalog()["shared_resources"]:
        verify_asset(name)
