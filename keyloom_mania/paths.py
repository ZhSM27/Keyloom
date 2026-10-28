"""Platform-independent runtime locations; nothing is written into site-packages."""
import os
from pathlib import Path


def application_home():
    override = os.environ.get("KEYLOOM_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return base / "keyloom"


APP_HOME = application_home()
ASSET_ROOT = APP_HOME / "assets"
OUTPUT_ROOT = APP_HOME / "generations"
LOG_ROOT = APP_HOME / "logs"
