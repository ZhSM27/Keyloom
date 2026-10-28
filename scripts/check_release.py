"""Check the public source allowlist and built distributions for accidental local data."""
import re
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("pyproject.toml", "MANIFEST.in", "LICENSE", "README.md", "README.en.md",
         "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md",
         "AGENTS.md", "studio.bat", ".gitignore", ".gitattributes")
DIRECTORIES = ("keyloom_mania", "tests", "scripts", "docs", ".github")
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".html", ".js", ".css", ".bat", ".txt", ".in"}
LOCAL_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", re.I)
SECRETS = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)")


def source_files(root=ROOT):
    for name in FILES:
        yield root / name
    for name in DIRECTORIES:
        for path in sorted((root / name).rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_file():
                if path.is_symlink():
                    raise ValueError(f"Symlink is not a public source file: {path.name}")
                yield path


def inspect(name, content):
    path = Path(name)
    if path.suffix in {".pt", ".pth", ".ckpt", ".osz", ".osu", ".wav", ".mp3", ".log"}:
        raise ValueError(f"Private or binary asset included in source: {name}")
    if len(content) > 5 * 1024 * 1024:
        raise ValueError(f"Unexpected large source file: {name}")
    if path.suffix in TEXT_SUFFIXES or path.name in {"LICENSE", "PKG-INFO", "METADATA"}:
        text = content.decode("utf-8-sig")
        if LOCAL_PATH.search(text) or SECRETS.search(text):
            raise ValueError(f"Possible local path or credential in {name}")


def main():
    count = 0
    for path in source_files():
        inspect(path.relative_to(ROOT).as_posix(), path.read_bytes())
        count += 1
    if len(sys.argv) > 1:
        for path in Path(sys.argv[1]).iterdir():
            if path.suffix in {".whl", ".zip"}:
                with zipfile.ZipFile(path) as archive:
                    for name in archive.namelist():
                        inspect(name, archive.read(name))
            elif path.name.endswith(".tar.gz"):
                with tarfile.open(path) as archive:
                    for entry in archive:
                        if entry.isfile():
                            inspect(entry.name, archive.extractfile(entry).read())
    print(f"Release source check passed: {count} public files.")


if __name__ == "__main__":
    main()
