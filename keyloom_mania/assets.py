"""Install only catalogued assets; verify before replacing existing resources."""
import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from .catalog import asset_path, catalog
from .paths import ASSET_ROOT


def install_bundle(bundle, root=None):
    root = Path(root or ASSET_ROOT).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    expected = catalog()["files"]
    with tempfile.TemporaryDirectory(prefix="assets-", dir=root.parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(bundle) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise ValueError("Bundle must contain exactly the catalogued resource files.")
            for entry in entries:
                specification = expected[entry.filename]
                destination = asset_path(entry.filename, root=staging)
                if entry.file_size != specification["bytes"]:
                    raise ValueError(f"Incorrect asset size: {entry.filename}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with archive.open(entry) as source, destination.open("wb") as target:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
                        target.write(block)
                if digest.hexdigest() != specification["sha256"]:
                    raise ValueError(f"Incorrect asset checksum: {entry.filename}")
        # Validation is complete before touching the installed resource set.
        for name in expected:
            destination = asset_path(name, root=root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / name, destination)
    return len(expected)


def download_asset(destination, specification):
    from .features import hash_file
    destination = Path(destination)
    if destination.is_file() and hash_file(destination) == specification["sha256"]:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in specification["urls"]:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as target:
                temporary = Path(target.name)
                with urlopen(url, timeout=60) as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            if hash_file(temporary) != specification["sha256"]:
                raise ValueError("Downloaded resource has an unexpected SHA-256.")
            os.replace(temporary, destination)
            return
        except Exception as error:
            errors.append(str(error))
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
    raise RuntimeError("Could not download " + destination.name + ": " + "; ".join(errors))


def fetch_audio(include_beats=False):
    from .audio_assets import DEFAULT_ASSET_ROOT, asset_specification
    for name, specification in asset_specification().items():
        print("Checking MusicFM: " + name, flush=True)
        download_asset(DEFAULT_ASSET_ROOT / name, specification)
    if include_beats:
        from .beat_timing import CHECKPOINT, CHECKPOINT_SHA256
        download_asset(CHECKPOINT, {"sha256": CHECKPOINT_SHA256, "urls": [
            "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
        ]})
