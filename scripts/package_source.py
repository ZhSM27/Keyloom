"""Make a source archive using the same explicit allowlist as release checks."""
import zipfile
from check_release import ROOT, source_files, inspect


def main():
    destination = ROOT / "dist/keyloom-0.1.0-source.zip"
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files():
            name = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            inspect(name, content)
            archive.writestr(name, content)
    print(destination.name)


if __name__ == "__main__":
    main()
