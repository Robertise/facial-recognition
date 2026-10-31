"""
env.py – read KEY=VALUE lines from the repo-root .env into os.environ.

Lets the same settings (INNOVATION_ENABLED, LIVENESS_CHECKPOINT, ...) work on
Windows shells, where `KEY=1 python ...` is not valid syntax. Variables that
are already set in the environment win over the file.
"""
import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded
